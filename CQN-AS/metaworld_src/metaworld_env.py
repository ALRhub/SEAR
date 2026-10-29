from typing import Any, NamedTuple
from collections import deque

import metaworld
import dm_env
import numpy as np
from dm_env import StepType, specs

from gymnasium import spaces


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class MetaWorldEnv(dm_env.Environment):
    def __init__(self, task_name, seed=None, render_mode='rgb_array'):
        # MetaWorld v3 uses v3 environments
        # Convert v2 task names to v3 if needed
        if task_name.endswith("-v2"):
            task_name = task_name.replace("-v2", "-v3")

        # Create MetaWorld benchmark - use ML1 for single task training
        if seed is None:
            seed = 0

        try:
            ml1 = metaworld.ML1(task_name, seed=seed)
            self._env = ml1.train_classes[task_name](render_mode=render_mode)
            self._env.set_task(ml1.train_tasks[0])
        except ValueError:
            raise ValueError(f"Unknown MetaWorld task: {task_name}. Use v3 task names like 'pick-place-v3'")

        self._max_episode_steps = self._env.max_path_length

    def observation_spec(self):
        shape = self._env.observation_space.shape
        assert len(shape) == 1
        return specs.Array((shape[0],), np.float32, "observation")

    def raw_observation_spec(self):
        shape = self._env.observation_space.shape
        assert len(shape) == 1
        return specs.Array((shape[0],), np.float32, "observation")

    def action_spec(self):
        # MetaWorld uses Box(-1, 1) for actions
        shape = self._env.action_space.shape
        return specs.BoundedArray(
            shape,
            np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action"
        )

    def reset(self):
        obs, info = self._env.reset()
        self._step_count = 0
        return TimeStep(
            observation=obs.astype(np.float32),
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        # MetaWorld episodes terminate on success or max steps
        done = terminated or truncated

        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        # Bootstrap on timeout (truncated), not on success (terminated)
        discount = float(1 - terminated)

        return TimeStep(
            observation=obs.astype(np.float32),
            step_type=step_type,
            reward=reward,
            discount=discount,
        )

    def render(self):
        return self._env.render()


def make(task_name, seed=None):
    env = MetaWorldEnv(task_name, seed=seed)
    env = ExtendedTimeStepWrapper(env)
    return env
