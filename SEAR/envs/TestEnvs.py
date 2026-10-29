import gymnasium as gym
import numpy as np


class Final100(gym.Env):
    """Environment that gives no reward for 9 steps, then reward of 100 at step 10 and terminates."""

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self._steps = 0

        # Dummy observation space - single float [0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

        # One-dimensional continuous action space (actions have no effect)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng = np.random.RandomState(seed)

        self._steps = 0
        return np.array([0.0]), {}

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        self._steps += 1

        # Dummy observation always [0]
        obs = np.array([self._steps])

        if self._steps == 10:
            reward = 100.0
        else:
            reward = 0.0

        if self._steps == 10:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = self._rng.random() > 0.9

        return obs, reward, terminated, truncated, {}


class Const1(gym.Env):
    """Environment that gives constant reward of 1, truncates after 10 steps, never terminates."""

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self._steps = 0

        # Dummy observation space - single float [0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

        # One-dimensional continuous action space (actions have no effect)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng = np.random.RandomState(seed)

        self._steps = 0
        return np.array([0.0]), {}

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        self._steps += 1

        # Dummy observation always [0]
        obs = np.array([0.0])

        # Constant reward of 1
        reward = 1.0

        # Never terminates, only truncates after 10 steps
        terminated = False
        truncated = self._steps >= 10

        return obs, reward, terminated, truncated, {}
