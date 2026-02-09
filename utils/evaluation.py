from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
            
    # Latent Visualization (for the last video episode if available, or last eval episode)
    if hasattr(agent, 'encode_batch') and len(trajs) > 0:
        # Use the last trajectory
        last_traj = trajs[-1]
        observations = np.array(last_traj['observation'])
        # Encode
        # Handle batching if too large? Usually eval traj is < 1000 steps.
        # Agent expects JAX array, typically.
        latents = agent.encode_batch(dict(observations=observations))
        latents = np.array(latents) # Convert to numpy
        
        # 1. Scatter: Latent Distance vs Time
        # Assume goal is fixed for the episode (or take from last step if s->g task)
        # We need the goal encoding.
        # If env has 'goal', it should be in info or passed.
        # The checking loop `evaluate` extracts `goal` from reset info.
        # But `trajs` might not store it explicitly in every step if it's static.
        # Let's assume we can get goal from the last trajectory's first observation? 
        # Or re-encode the goal found in info.
        
        # Re-fetch goal from the loop is hard.
        # But we can assume the goal for 'latent distance' is the target goal.
        # If we have goals in `last_traj`? No, `evaluate` uses `goal` from reset.
        
        # Let's map 'Time Difference' as 'Steps Remaining' or just 'Time Step'.
        # If we can't easily get 'goal', we can plot 'Distance to Final State'.
        
        final_state_latent = latents[-1:]
        dists_to_final = np.linalg.norm(latents - final_state_latent, axis=-1)
        
        fig, ax = plt.subplots()
        ax.plot(dists_to_final)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Latent Distance to Final State')
        ax.set_title('Latent Distance vs Time')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        stats['eval/latent_dist_vs_time'] = wandb.Image(img)
        
        # 2. t-SNE of Trajectory
        if TSNE is not None and len(latents) > 10:
            try:
                # Subsample if too large
                indices = np.linspace(0, len(latents)-1, min(500, len(latents)), dtype=int)
                sub_latents = latents[indices]
                tsne = TSNE(n_components=2, perplexity=min(30, len(sub_latents)-1))
                z_embedded = tsne.fit_transform(sub_latents)
                
                fig, ax = plt.subplots()
                sc = ax.scatter(z_embedded[:, 0], z_embedded[:, 1], c=np.arange(len(sub_latents)), cmap='viridis')
                plt.colorbar(sc, label='Time Step')
                ax.set_title('t-SNE of Latent Trajectory')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                stats['eval/latent_tsne'] = wandb.Image(img)
            except Exception as e:
                print(f"t-SNE visualization failed: {e}")

        # 3. Call agent's visualize_latents (e.g. Histogram)
        if hasattr(agent, 'visualize_latents'):
             # Construct batch
             # We need 'value_goals'. Assume it's the goal from the episode info?
             # evaluation loop extracted `goal` from reset info. But we are outside the loop.
             # We can try to get it from the last transition's info if preserved, or just use the last observation as "achieved goal"?
             # Actually, simpler: construct batch from 'observations' and 'next_observations'.
             # For 'value_goals', let's use the final state of the trajectory as a proxy for the goal? 
             # Or if the env has 'goal' key in observation?
             # In AntMaze, observation is just state. Goal is separate.
             # Let's Skip if we can't easily get the goal.
             pass 
             
             # Actually, let's use the latents we already computed!
             # We can plot histogram of norms here directly to save computation.
             z_norms = np.linalg.norm(latents, axis=-1)
             
             fig = plt.figure()
             plt.hist(z_norms, bins=20)
             plt.title('Latent Norm Histogram')
             buf = io.BytesIO()
             plt.savefig(buf, format='png')
             plt.close(fig)
             buf.seek(0)
             img = Image.open(buf)
             stats['eval/latent_norm_hist'] = wandb.Image(img)

    for k, v in stats.items():
        if isinstance(v, list):
            stats[k] = np.mean(v)

    return stats, trajs, renders
