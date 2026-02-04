import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info and len(info['goal'].shape) == 3: # H, W, C라고 가정
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        if 'goal' in info and len(info['goal'].shape) == 3:
             info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, dataset_path, frame_stack = None, action_clip_eps=1e-5, dataset_only=False, cur_env=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the environment (dataset).
        dataset_path: Path to the dataset file.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).

    Returns:
        A tuple of the environment (if `dataset_only` is False), training dataset, and validation dataset.
    """
    if dataset_only:
        train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_path=dataset_path, compact_dataset=True, dataset_only=dataset_only, cur_env=cur_env
        )
    else:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_path=dataset_path, compact_dataset=True, dataset_only=dataset_only, cur_env=cur_env
        )
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
    eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    # 그냥 env는 원래 없었는데 꼭 넣어야 하나?
    env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    # Dataset 객체에 frame_stack 정보 주입 (Dataset 클래스에서 사용)
    if frame_stack is not None:
        train_dataset.frame_stack = frame_stack
        val_dataset.frame_stack = frame_stack

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps))
    )
    val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env.reset()
        eval_env.reset()
        return env, eval_env, train_dataset, val_dataset
