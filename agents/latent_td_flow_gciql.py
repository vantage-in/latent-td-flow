import copy
from typing import Any
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import FlowModel, GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue, RewardModel

class LatentTDFlowGCIQLAgent(flax.struct.PyTreeNode):
    """Latent TD-Flow Agent with GCIQL Policy.

    Learns a latent state representation using Flow Matching and concurrently trains a GCIQL policy on the latents.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def flow_loss(self, batch, z_s, z_g, z_next, grad_params, rng=None):
        """Compute the flow matching loss."""

        # Bootstrapping Target Construction
        is_success = 1.0 - batch['masks']
        
        # ODE Solver
        def ode_solve(cond_s, cond_g, start_z):
            # Integrate v_theta from t=0 to t=1 starting at Noise z(0), which sampled as start_z in advance 
            
            steps = 5 # Small number for trade-off
            dt = 1.0 / steps
            curr_z = start_z
            for i in range(steps):
                t = i * dt
                t_batch = jnp.full((curr_z.shape[0], 1), t)
                # condition: current flow state and goal
                v = self.network.select('flow_model')(curr_z, t_batch, cond_s, cond_g, params=grad_params)
                # v = jax.lax.stop_gradient(v)
                curr_z = curr_z + v * dt
            return curr_z

        # Random key for ODE solver
        rng, z0_rng = jax.random.split(rng)
        z0 = jax.random.normal(z0_rng, shape=z_s.shape) # z0 is sampled only once and used for both bootstrapping and flow loss calculation
        z_future_pred = ode_solve(z_next, z_g, z0)
        
        # Random choice for bootstrap
        rng, boot_rng = jax.random.split(rng)
        boot_mask = jax.random.bernoulli(boot_rng, p=self.config['discount'], shape=(z_s.shape[0], 1))
        
        target_bootstrap = jnp.where(boot_mask, z_future_pred, z_next)
        
        if self.config['use_absorbing_state']:
            Z = jnp.where(is_success[:, None], z_s,
                  jnp.where(batch['masks'][:, None] == 0, z_next, # if done/success at next step
                    target_bootstrap
                  )
                )
        else:
            Z = target_bootstrap
        
        # Stop gradient on the entire target Z
        Z = jax.lax.stop_gradient(Z)
        
        # Flow Matching Loss
        rng, t_rng = jax.random.split(rng, 2)
        t = jax.random.uniform(t_rng, shape=(z_s.shape[0], 1))
        # z0 is already available
        
        z_t = t * Z + (1 - t) * z0
        target_v = Z - z0
        pred_v = self.network.select('flow_model')(z_t, t, z_s, z_g, params=grad_params)
        
        # Masking flow loss (Hybrid Sampling)
        sq_diff = (pred_v - target_v)**2
        # Mean over dims
        pixel_loss = jnp.mean(sq_diff, axis=-1) # (B,)
        
        mask = batch.get('value_feasible_mask', jnp.ones(pixel_loss.shape))
        loss = jnp.sum(pixel_loss * mask) / (jnp.sum(mask) + 1e-6)
        
        return loss, {
            'flow_loss': loss,
            'z_norm': jnp.mean(jnp.linalg.norm(z_s, axis=-1)),
            'z_diff': jnp.mean(jnp.linalg.norm(z_s - z_g, axis=-1)),
        }

    def value_loss(self, batch, z_s, z_g, grad_params):
        """Compute the IQL value loss using latent representations."""
        # Q-target: Min over target critics
        q1, q2 = self.network.select('target_critic')(z_s, z_g, batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(z_s, z_g, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, z_s, z_g, z_next, grad_params):
        """Compute the IQL critic loss using latent representations."""
        # Target Value (from target encoder z_next)
        next_v = self.network.select('value')(z_next, z_g)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(z_s, z_g, batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, z_s, z_g, z_next, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(z_s, z_g)
            q1, q2 = self.network.select('critic')(z_s, z_g, batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(z_s, z_g, params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(z_s, z_g, params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            
            # Critic Q(s, pi(s))
            q1, q2 = self.network.select('critic')(z_s, z_g, q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss
            
            actor_info = {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
            return actor_loss, actor_info
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    def reward_loss(self, batch, z_s, z_g, params):
        """Compute the auxiliary reward prediction loss."""
        # Predict if s and g are the same state (success)
        targets = (1.0 - batch['masks']) # (B,)
        
        logits = self.network.select('reward_model')(z_s, z_g, params=params)
        loss = optax.sigmoid_binary_cross_entropy(logits, targets).mean()
        
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        accuracy = (preds == targets).mean()
        
        return loss, {
            'reward_loss': loss,
            'reward_acc': accuracy
        }

    def ortho_loss(self, z):
        """Compute the orthonormality regularization loss."""
        # z: (B, D)
        B = z.shape[0]
        # Gram matrix: C = Z * Z^T => (B, B)
        C = jnp.matmul(z, z.T)
        
        # Diagonal elements
        diag = jnp.diag(C) # (B,)
        l_diag = -jnp.mean(diag)
        
        # Off-diagonal elements
        off_diag_mask = 1.0 - jnp.eye(B)
        off_diag = (C ** 2) * off_diag_mask
        
        # Sum of off-diagonal squares, divided by B(B-1)
        l_off = jnp.sum(off_diag) / (B * (B - 1) + 1e-6)
        
        loss = l_diag + self.config['ortho_off_diag_coef'] * l_off
        
        return loss, {
            'ortho_loss': loss,
            'ortho_diag': l_diag,
            'ortho_off': l_off,
        }

    def contrastive_loss(self, z_s, z_g, feasible_mask, temperature=0.1):
        """
        Compute Contrastive (InfoNCE) loss.
        z_s: (B, D) - Encoded States
        z_g: (B, D) - Encoded Goals
        feasible_mask: (B,) - 1.0 if (s, g) is a trajectory pair, 0.0 if random
        """
        # 1. Cosine Similarity Matrix (B x B)
        # Normalize
        z_s_norm = z_s / (jnp.linalg.norm(z_s, axis=-1, keepdims=True) + 1e-6)
        z_g_norm = z_g / (jnp.linalg.norm(z_g, axis=-1, keepdims=True) + 1e-6)
        
        # logits[i][j] = sim(s_i, g_j)
        logits = jnp.matmul(z_s_norm, z_g_norm.T) / temperature 
        
        # 2. InfoNCE for Positive Anchors
        # Labels are 0, 1, ... B-1 (diagonal is positive pair)
        labels = jnp.arange(logits.shape[0])
        
        # Loss calculation (Cross Entropy)
        loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        
        # 3. Masking: Apply loss ONLY to feasible pairs
        masked_loss = loss_per_sample * feasible_mask
        
        # Average over feasible samples
        return jnp.sum(masked_loss) / (jnp.sum(feasible_mask) + 1e-6)

    @jax.jit
    def total_loss(self, grad_params, batch, rng=None):
        """Compute the total loss (Flow + GCIQL)."""
        info = {}
        rng = rng if rng is not None else self.rng
        
        # 1. Compute Latents
        z_s = self.network.select('encoder')(batch['observations'], params=grad_params)
        
        if self.config['separate_encoders']:
            z_g = self.network.select('goal_encoder')(batch['value_goals'], params=grad_params)
            z_g_actor = self.network.select('goal_encoder')(batch['actor_goals'], params=grad_params)
        else:
            z_g = self.network.select('encoder')(batch['value_goals'], params=grad_params)
            z_g_actor = self.network.select('encoder')(batch['actor_goals'], params=grad_params)
            
        if self.config['use_target_encoder']:
             z_next = self.network.select('target_encoder')(batch['next_observations'])
             z_next = jax.lax.stop_gradient(z_next)
        else:
             z_next = self.network.select('encoder')(batch['next_observations'], params=grad_params)
             z_next = jax.lax.stop_gradient(z_next)
        
        # 2. Flow Loss
        rng, flow_rng = jax.random.split(rng)
        
        flow_loss, flow_info = self.flow_loss(batch, z_s, z_g, z_next, grad_params, flow_rng)
        for k, v in flow_info.items():
            info[f'flow/{k}'] = v
            
        # 3. RL Loss
        # Detach for RL
        z_s_rl = jax.lax.stop_gradient(z_s)
        z_g_rl = jax.lax.stop_gradient(z_g)
        z_g_actor_rl = jax.lax.stop_gradient(z_g_actor)
        z_next_rl = jax.lax.stop_gradient(z_next)

        # Value Loss
        value_loss, value_info = self.value_loss(batch, z_s_rl, z_g_rl, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
            
        # Critic Loss
        critic_loss, critic_info = self.critic_loss(batch, z_s_rl, z_g_rl, z_next_rl, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
            
        # Actor Loss
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, z_s_rl, z_g_actor_rl, z_next_rl, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # 4. Aux Reward Loss (Updates Encoder)
        if self.config['aux_loss_coef'] > 0.0:
            reward_loss, reward_info = self.reward_loss(batch, z_s, z_g, grad_params)
            for k, v in reward_info.items():
                info[f'aux/{k}'] = v
        else:
            reward_loss = 0.0
            
        # 5. Ortho Loss
        if self.config['ortho_coef'] > 0.0:
            ortho_loss, ortho_info = self.ortho_loss(z_s)
            for k, v in ortho_info.items():
                info[f'aux/{k}'] = v
        else:
            ortho_loss = 0.0

        # 6. Contrastive Loss
        if self.config['contrastive_coef'] > 0.0:
            feasible_mask = batch.get('value_feasible_mask', jnp.ones(batch['masks'].shape))
            contrastive_loss = self.contrastive_loss(z_s, z_g, feasible_mask)
            info['aux/contrastive_loss'] = contrastive_loss
        else:
            contrastive_loss = 0.0
        
        # Total Loss
        loss = flow_loss + value_loss + critic_loss + actor_loss + \
               self.config['aux_loss_coef'] * reward_loss + \
               self.config['ortho_coef'] * ortho_loss + \
               self.config['contrastive_coef'] * contrastive_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network parameters with EMA."""
        # soft update: theta' = tau * theta + (1 - tau) * theta'
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Perform a single update step."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(grad_params, batch, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
        # Update Target Networks (Soft Update)
        self.target_update(new_network, 'critic')
        if self.config['use_target_encoder']:
             self.target_update(new_network, 'encoder')
            
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions."""
        z_s = self.network.select('encoder')(observations)
        z_g = self.network.select('encoder')(goals) if goals is not None else None
        
        dist = self.network.select('actor')(z_s, z_g, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def encode_batch(self, batch):
        """Encode a batch of observations."""
        return self.network.select('encoder')(batch['observations'])
    
    def visualize_latents(self, batch):
        """Visualize latents (same implementation as LatentTDFlowAgent)."""
        z_s = self.network.select('encoder')(batch['observations'])
        z_g = self.network.select('encoder')(batch['value_goals'])
        dists = jnp.linalg.norm(z_s - z_g, axis=-1)
        
        images = {}
        fig = plt.figure()
        plt.hist(dists, bins=20)
        plt.title('Latent Distance to Goal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)
        images['eval/latent_dist_hist'] = wandb.Image(image)
        return images

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoder_cls = encoder_modules[config['encoder']]
        final_tanh = config.get('final_tanh', False)
        final_norm = config.get('final_norm', 'tanh' if final_tanh else None)
        encoder_def = encoder_cls(final_norm=final_norm)
        # Initialize one to get latent dim
        latent_dim = encoder_def.mlp_hidden_dims[-1]
        
        # Flow Model
        flow_model_def = FlowModel(
            hidden_dims=config['flow_hidden_dims'], 
            latent_dim=latent_dim
        )
        
        reward_model_def = RewardModel(
             hidden_dims=config['flow_hidden_dims'],
             layer_norm=config['layer_norm']
        )
        
        # RL Networks
        # Note: We set gc_encoder=None because inputs are already encoded latents z_s, z_g.
        
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=None,
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=None,
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=None,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=None,
            )
            
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=None,
                const_std=config['const_std'],
                state_dependent_std=False,
            )

        # Initialize Parameters
        # We need to run init to get valid params.
        dummy_enc_params = encoder_def.init(init_rng, ex_observations)
        ex_latents = encoder_def.apply(dummy_enc_params, ex_observations)
        
        network_info = dict(
            encoder=(encoder_def, (ex_observations,)),
            target_encoder=(copy.deepcopy(encoder_def), (ex_observations,)),
            flow_model=(flow_model_def, (ex_latents, jnp.zeros((ex_latents.shape[0], 1)), ex_latents, ex_latents)),
            reward_model=(reward_model_def, (ex_latents, ex_latents)),
            
            value=(value_def, (ex_latents, ex_latents)),
            
            critic=(critic_def, (ex_latents, ex_latents, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_latents, ex_latents, ex_actions)),
            
            actor=(actor_def, (ex_latents, ex_latents)),
        )

        if config['separate_encoders']:
            goal_encoder_def = encoder_cls(final_norm=final_norm) 
            network_info['goal_encoder'] = (goal_encoder_def, (ex_observations,))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        
        # Optimizer Definition
        if config['lr_decay']:
            print(f"Initializing LR Decay with decay_steps={config['decay_steps']}")
            # Cosine decay for representation learning
            lr_rep = optax.cosine_decay_schedule(
                init_value=config['lr'],
                decay_steps=config['decay_steps'],
                alpha=config.get('decay_alpha', 0.0) # Decay to alpha * lr
            )
            lr_rl = config['lr'] # Constant for RL
            
            # Partition parameters
            def param_labels(params):
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: 'rep' if any(
                        key.key in ['modules_encoder', 'modules_flow_model', 'modules_reward_model', 'modules_goal_encoder'] 
                        for key in path
                    ) else 'rl',
                    params
                )
            
            optimizers = {
                'rep': optax.adam(learning_rate=lr_rep),
                'rl': optax.adam(learning_rate=lr_rl),
            }
            
            network_tx = optax.multi_transform(optimizers, param_labels)
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
            
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_encoder'] = params['modules_encoder']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='latent_td_flow_gciql',
            lr=3e-4,
            batch_size=256,
            
            flow_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='ddpgbc', 
            alpha=1.0,           
            const_std=True,
            discrete=False,
            
            encoder='impala_small',
            dataset_class='GCDataset',
            
            # Goal Sampling
            value_p_curgoal=0.2, 
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            
            gc_negative=True,
            p_aug=0.5,
            frame_stack=1,
            
            final_norm='layernorm', # 'tanh', 'layernorm', 'layernorm_tanh', 'l2', None
            aux_loss_coef=0.1, 
            ortho_coef=0.00001,
            ortho_off_diag_coef=0.5,
            contrastive_coef=0.05, # Default coefficient for contrastive loss
            separate_encoders=True,
            use_target_encoder=True,
            use_absorbing_state=False, # Whether to use absorbing state logic for target Z
            lr_decay=True,
            decay_steps=int(2e5),
            decay_alpha=0.1,
        )
    )
    return config
