#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:45:07 2024


"""

import gymnasium as gym
import numpy as np
from typing import Any, Callable
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import typer
import yaml
import jsons
from pathlib import Path
import jax
import jax.numpy as jnp

# Import SAC from the same location as in the first file
# Note: You'll need to ensure this path is correct for your setup
from sac.sac import SAC
from sac.config import Config

app = typer.Typer()

ColorType = tuple[int, int, int]


class _Wall:

    def __init__(self, upper: float, lower: float, left: float, right: float):
        self._upper = upper
        self._lower = lower
        self._left = left
        self._right = right

    def render_coordinates(
        self, tf_fn: Callable[[float], float], contraction: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (tf_fn(self._left) + contraction, tf_fn(self._lower) + contraction),
            (tf_fn(self._right) - contraction, tf_fn(self._upper) - contraction),
        )

    def project_outside(
        self, pos: np.ndarray, vel: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Projects the point to the closest point outside of the wall"""
        if self._left < pos[0] < self._right and self._lower < pos[1] < self._upper:
            distances = {
                "left": pos[0] - self._left,
                "right": self._right - pos[0],
                "lower": pos[1] - self._lower,
                "upper": self._upper - pos[1],
            }
            closest_edge = min(distances, key=distances.get)
            if closest_edge == "left":
                return np.array([self._left, pos[1]]), (
                    None if vel is None else np.array([0, vel[1]])
                )
            elif closest_edge == "right":
                return np.array([self._right, pos[1]]), (
                    None if vel is None else np.array([0, vel[1]])
                )
            elif closest_edge == "lower":
                return np.array([pos[0], self._lower]), (
                    None if vel is None else np.array([vel[0], 0])
                )
            elif closest_edge == "upper":
                return np.array([pos[0], self._upper]), (
                    None if vel is None else np.array([vel[0], 0])
                )
        else:
            return pos, vel


class BaseEnv(gym.Env):
    _AGENT_COLOR = (200, 0, 0)
    _AGENT_SHADOW_COLOR = (230, 14, 165)
    _GOAL_COLOR = (0, 200, 0)
    _GOAL_SHADOW_COLOR = (10, 180, 10)
    _BG_COLOR = (220, 220, 220)
    _WALL_COLOR = (35, 35, 35)

    _HORIZON = 100

    _DT = 0.05
    _DAMPING = 0.9
    _ACC_SCALE = 4.0

    _BOARDER_SIZE = 6
    _INTERNAL_IMG_SIZE = 256 - 2 * _BOARDER_SIZE

    _AGENT_SIZE = 4

    _ACTION_SCALE = 0.05

    _SUCCESS_THRESHOLD = 0.05
    _SUCCESS_BONUS_SCALE = 0.2

    _START_LEFT = -0.95
    _START_RIGHT = -0.8

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ]
    }

    def __init__(
        self,
        seed: int,
        obs_keys: list | str,
        img_size: int = 128,
        velocity_control: bool = False,
        render_mode: str = None,
        fixed_start_goal: bool = False,
    ):
        self.render_mode = render_mode
        self.fixed_start_goal = fixed_start_goal
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        self._obs_keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
        self._img_size = img_size

        self._agent_pos_scale = self._INTERNAL_IMG_SIZE / 2 - self._AGENT_SIZE

        self._agent_pos = None
        self._agent_vel = None
        self._steps = 0
        self._initialized = False

        self._goal_pos = None
        self._velocity_control = velocity_control

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        state_dim = 6 if self._velocity_control else 4

        # obs_spaces = {}
        # for key in self._obs_keys:
        #     if key == "img":
        #         space = gym.spaces.Box(low=0, high=255, shape=(self._img_size, self._img_size, 3), dtype=np.uint8)
        #     elif key == "position":
        #         space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        #     elif key == "state":
        #         space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        #     else:
        #         raise AssertionError(f"Unknown obs key: {key}")
        #     obs_spaces[key] = space
        # self.observation_space = gym.spaces.Dict(spaces=obs_spaces)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,)
        )

    def _agent_reset(self):
        if not self.fixed_start_goal:
            self._agent_pos = self._rng.uniform(
                low=[self._START_LEFT, -0.95], high=[self._START_RIGHT, 0.95]
            )
        else:
            # self._agent_pos = np.array([self._START_LEFT, -0.0])
            # self._agent_pos = np.array([-0.25, -0.1])
            self._agent_pos = self._rng.uniform(
                low=[self._START_LEFT, -0.1], high=[self._START_RIGHT, 0.1]
            )
            # self._agent_pos = self._rng.uniform(low=[self._START_LEFT, -.95], high=[self._START_RIGHT, .95])

    def _draw_circle(
        self, draw: ImageDraw, pos: np.ndarray, color: ColorType, size: float
    ):
        pos = self._internal_to_img(pos)
        draw.ellipse(
            [pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size], fill=color
        )

    def _internal_to_img(self, pos: np.ndarray):
        return (pos + 1) * self._agent_pos_scale + (
            self._AGENT_SIZE + self._BOARDER_SIZE
        )

    def _get_var_dim(self, key: str):
        raise AssertionError(f"Unknown obs key: {key}")

    def _goal_reset(self):
        if not self.fixed_start_goal:
            self._goal_pos = self._rng.uniform(low=[0.8, -0.95], high=[0.95, 0.95])
        else:
            self._goal_pos = np.array([0.875, 0.0])

    # def _get_obs(self) -> dict[str, np.ndarray]:
    #
    #     dist_to_goal = (self._goal_pos - self._agent_pos) / 2
    #     obs_dict = {}
    #     for k in self._obs_keys:
    #
    #         if k == "state":
    #             if self._velocity_control:
    #                 o = np.concatenate((self._agent_pos, self._agent_vel, dist_to_goal), axis=0)
    #             else:
    #                 o = np.concatenate((self._agent_pos, dist_to_goal), axis=0)
    #         elif k == "position":
    #             o = np.concatenate((self._agent_pos, dist_to_goal), axis=0)
    #         else:  # img
    #             img, _ = self._render()
    #             img = img.resize((self._img_size, self._img_size))
    #             o = np.ascontiguousarray(np.transpose(np.array(img), (2, 0, 1)))
    #         obs_dict[k] = o
    #     return obs_dict

    def _get_obs(self) -> dict[str, np.ndarray]:

        dist_to_goal = (self._goal_pos - self._agent_pos) / 2
        if self._velocity_control:
            o = np.concatenate((self._agent_pos, self._agent_vel, dist_to_goal), axis=0)
        else:
            o = np.concatenate((self._agent_pos, dist_to_goal), axis=0)
        return o

    def _get_reward(self) -> float:
        norm = np.linalg.norm(self._agent_pos - self._goal_pos)
        norm_reward = 1 - norm / np.sqrt(8)
        success = float(self._get_success())
        return (
            self._SUCCESS_BONUS_SCALE * success
            + (1 - self._SUCCESS_BONUS_SCALE) * norm_reward
        )

    def _get_success(self) -> bool:
        return bool(
            np.linalg.norm(self._agent_pos - self._goal_pos) < self._SUCCESS_THRESHOLD
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.array(action, dtype=np.float32)
        assert self.action_space.contains(action)
        assert self._initialized, "Call reset() before step()"
        # print(f"action=[{action[0]:2.2f}, {action[1]:2.2f}]")

        self._step(action)
        # Collision with outside walls
        self._agent_pos = np.clip(self._agent_pos, -1, 1)

        self._steps += 1
        truncated = self._steps >= self._HORIZON

        return (
            self._get_obs(),
            self._get_reward(),
            False,
            truncated,
            {"is_success": self._get_success()},
        )

    def _step(self, action: np.ndarray):
        if self._velocity_control:
            self._agent_vel = (
                self._DAMPING * self._agent_vel + action * self._ACC_SCALE * self._DT
            )
            self._agent_pos += self._agent_vel * self._DT
        else:
            self._agent_pos += action * self._ACTION_SCALE

        # Collision with outside walls
        if self._agent_pos[0] < -1 or self._agent_pos[0] > 1:
            self._agent_pos[0] = np.clip(self._agent_pos[0], -1, 1)
            if self._agent_vel is not None:
                self._agent_vel[0] = 0.0

        if self._agent_pos[1] < -1 or self._agent_pos[1] > 1:
            self._agent_pos[1] = np.clip(self._agent_pos[1], -1, 1)
            if self._agent_vel is not None:
                self._agent_vel[1] = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._initialized = True
        self._agent_reset()
        if self._velocity_control:
            self._agent_vel = np.zeros(2)
        self._goal_reset()
        self._steps = 0
        return self._get_obs(), {"success": False}

    def _render(self):
        img = Image.new(
            "RGB",
            (self._INTERNAL_IMG_SIZE, self._INTERNAL_IMG_SIZE),
            color=self._BG_COLOR,
        )
        img = ImageOps.expand(img, border=self._BOARDER_SIZE, fill=self._WALL_COLOR)

        draw = ImageDraw.Draw(img)
        self._draw_circle(draw, self._agent_pos, self._AGENT_COLOR, self._AGENT_SIZE)
        self._draw_circle(draw, self._goal_pos, self._GOAL_COLOR, self._AGENT_SIZE)
        return img, draw


class SlitEnv(BaseEnv):
    _BAR_RAD = 0.1

    def __init__(
        self,
        seed: int,
        obs_keys: str,
        img_size: int = 128,
        slit_width: float = 0.2,
        velocity_control: bool = False,
        render_mode: str = None,
        fixed_start_goal: bool = False,
        sparse: bool = False,
    ):
        super().__init__(
            seed=seed,
            obs_keys=obs_keys,
            img_size=img_size,
            velocity_control=velocity_control,
            render_mode=render_mode,
            fixed_start_goal=fixed_start_goal,
        )
        self._slit_rad = slit_width / 2
        self._wall_1 = _Wall(
            upper=0.4, lower=-0.4, left=-self._BAR_RAD, right=self._BAR_RAD
        )
        self.fig = None
        self._wall_1_coordinates = self._wall_1.render_coordinates(
            tf_fn=self._internal_to_img, contraction=self._AGENT_SIZE
        )
        self.sparse = sparse

    def _step(self, raw_action: np.ndarray):
        super()._step(raw_action)

        # Collision with walls
        self._agent_pos, self._agent_vel = self._wall_1.project_outside(
            self._agent_pos, self._agent_vel
        )

    def _render(self):
        img, draw = super()._render()
        draw.rectangle(self._wall_1_coordinates, fill=self._WALL_COLOR)

        return img, draw

    def render(self, img_size: int = 256) -> np.ndarray:
        img, _ = self._render()
        img = np.array(img)
        if img_size != img.shape[0] or img_size != img.shape[1]:
            img = Image.fromarray(img)
            img = img.resize((img_size, img_size))
            img = np.array(img)
        if self.render_mode == "human":
            self.fig = plt.figure() if self.fig is None else self.fig
            plt.figure(self.fig.number)
            plt.cla()
            plt.clf()
            plt.imshow(img)
            plt.xlabel("x_0")
            plt.ylabel("x_1")
            plt.tick_params(
                axis="x", top=True, labeltop=True, bottom=False, labelbottom=False
            )
            plt.show()
            plt.pause(1.0e-12)
        return img


@app.command()
def random():
    """Run the environment with random actions and visualize the results."""
    # Initialize the environment
    env = SlitEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        slit_width=0.2,
        velocity_control=False,
        render_mode="human",
    )

    # Reset the environment to start
    obs, _ = env.reset()
    print("Initial Observation:", obs)

    # Run for 100 steps with random actions
    for step in range(100):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the current state of the environment
        img = env.render(img_size=256)  # You can adjust the size as needed
        plt.imshow(img)
        plt.title(f'Step {step}: Reward={reward:.3f}, Success={info["is_success"]}')
        plt.show()

        print(
            f"Step {step}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}"
        )

        # Check if the episode is over
        if terminated or truncated:
            break


@app.command()
def traj(policy: str, n: int, n_trajs: int = 20):
    """
    Display trajectories of the trained agent in the environment.

    Args:
        policy: Path to the saved policy
        n: Checkpoint number
        n_trajs: Number of trajectories to run
    """
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = SlitEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        slit_width=0.2,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=True,
    )

    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy, seed=42)
    agent.load_model(policy, n, n)

    trajectories = []
    success = []

    for i in range(n_trajs):
        obs, _ = env.reset(seed=i)
        current_traj = [obs[:2]]  # Only keep the agent position for visualization
        state = None

        exit = False
        for step in range(env._HORIZON):
            actions, state = agent.predict(obs, state=state, deterministic=False)
            agent.policy.noise_key, agent.policy.key = jax.random.split(
                agent.policy.key, 2
            )
            for action in np.split(actions, config.alg.actor_chunksize, axis=0):
                obs, _, terminated, truncated, info = env.step(action)
                current_traj.append(
                    obs[:2]
                )  # Only keep the agent position for visualization

                # Check if the episode is over
                if terminated or truncated or info["is_success"]:
                    exit = True
                    break
            if exit:
                trajectories.append(current_traj)
                success.append(info["is_success"])
                break

    # plot overview
    fig, ax = plt.subplots()
    # Draw environment boundaries
    ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], "k-")
    # Draw the wall/barrier in the middle
    ax.plot(
        [-0.1, 0.1, 0.1, -0.1, -0.1], [-0.4, -0.4, 0.4, 0.4, -0.4], "k-", linewidth=2
    )
    # Mark the goal position
    goal_pos = env._goal_pos
    ax.scatter(goal_pos[0], goal_pos[1], color="green", s=100, marker="*", label="Goal")

    # Plot trajectories
    for traj in trajectories:
        ax.plot(*np.array(traj).T, "o-", alpha=1 / n_trajs)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect("equal")
    ax.set_title(f"Trajectories ({np.mean(success)*100:.2f}% success)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.show()
    plt.show(block=True)


@app.command()
def rollout(policy: str, n: int):
    """
    Perform a rollout using the trained policy and visualize the agent's behavior.

    Args:
        policy: Path to the saved policy
        n: Checkpoint number
    """
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = SlitEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        slit_width=0.2,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=True,
    )
    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy)
    agent.load_model(policy, n, n)

    for i in range(5):
        plt.close()
        plt.ion()

        obs, _ = env.reset(seed=i)
        state = None
        print("Initial Observation:", obs)

        exit = False
        for step in range(env._HORIZON):
            actions, state = agent.predict(obs, state=state, deterministic=False)
            for action in np.split(actions, config.alg.actor_chunksize, axis=0):
                obs, reward, terminated, truncated, info = env.step(action)

                # Render the current state of the environment
                img = env.render(img_size=256)  # You can adjust the size as needed
                plt.clf()
                plt.imshow(img)
                plt.title(
                    f'Step {step}: Reward={reward:.3f}, Success={info["is_success"]}'
                )
                plt.draw()
                plt.pause(1 / 30)

                print(
                    f"Step {step}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}"
                )

                # Check if the episode is over
                if terminated or truncated or info["is_success"]:
                    exit = True
                    break
            if exit:
                break


@app.command()
def corr(policy: str, checkpoint: int, at: int = 0, n_trajs: int = 5):
    """
    Visualize the model's action distribution/correlation at a specific timestep.

    Args:
        policy: Path to the saved policy
        checkpoint: Checkpoint number
        at: Timestep to visualize the action distribution
        n_trajs: Number of sampled trajectories to visualize
    """
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = SlitEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        slit_width=0.2,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=True,
    )
    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy)
    agent.load_model(policy, checkpoint, checkpoint)
    action_key = jax.random.PRNGKey(42)

    for i in range(1):
        plt.close()
        plt.ion()

        obs, _ = env.reset(seed=0)
        past_obs = [obs[:2]]  # Only store agent position for visualization
        state = None
        print("Initial Observation:", obs)

        fig, ax = plt.subplots()
        ax.set_box_aspect(1)

        # Draw environment boundaries
        ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], "k-")
        # Draw the wall/barrier in the middle
        ax.plot(
            [-0.1, 0.1, 0.1, -0.1, -0.1],
            [-0.4, -0.4, 0.4, 0.4, -0.4],
            "k-",
            linewidth=2,
        )
        # Mark the goal position
        goal_pos = env._goal_pos
        ax.scatter(
            goal_pos[0], goal_pos[1], color="green", s=100, marker="*", label="Goal"
        )

        exit = False

        for step in range(100):
            action, state = agent.predict(obs, state, None)
            for i, action in enumerate(
                np.split(action, config.alg.actor_chunksize, axis=0)
            ):
                ax.clear()
                # Redraw environment boundaries
                ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], "k-")
                # Redraw the wall/barrier in the middle
                ax.plot(
                    [-0.1, 0.1, 0.1, -0.1, -0.1],
                    [-0.4, -0.4, 0.4, 0.4, -0.4],
                    "k-",
                    linewidth=2,
                )
                # Redraw goal position
                ax.scatter(
                    goal_pos[0],
                    goal_pos[1],
                    color="green",
                    s=100,
                    marker="*",
                    label="Goal",
                )

                obs, reward, terminated, truncated, info = env.step(action[:2])
                assert isinstance(obs, np.ndarray)
                obs = obs.astype(np.float32)
                past_obs.append(obs[:2])  # Only store agent position for visualization

                # Render the current state of the environment
                ax.scatter(*obs[:2], color="red")
                ax.set_title(
                    f'Step {step}: Reward={reward:.3f}, Success={info["is_success"]}'
                )
                ax.set_xlim([-1.1, 1.1])
                ax.set_ylim([-1.1, 1.1])
                plt.draw()
                plt.pause(1 / 30)

                if step + i == at:
                    # plot trajectory so far
                    ax.plot(
                        *np.array(past_obs).T, "r-", linewidth=2, label="Trajectory"
                    )

                    # Sample from the policy and visualize distribution
                    key, action_key = jax.random.split(action_key, 2)
                    _, _, cov, _ = agent.policy.actor_state.apply_fn(
                        {
                            "params": agent.policy.actor_state.params,
                            "batch_stats": agent.policy.actor_state.batch_stats,
                        },
                        jnp.array(np.expand_dims(obs, 0)),
                        train=False,
                        deterministic=True,
                        key=action_key,
                    )
                    chunks = [
                        agent.policy.actor_state.apply_fn(
                            {
                                "params": agent.policy.actor_state.params,
                                "batch_stats": agent.policy.actor_state.batch_stats,
                            },
                            jnp.array(np.expand_dims(obs, 0)),
                            train=False,
                            deterministic=True,
                            key=key,
                        )[0]
                        for key in jax.random.split(key, n_trajs)
                    ]

                    # Plot action samples starting from current position
                    for chunk in chunks:
                        traj = np.clip(
                            np.cumsum(
                                obs[:2] + chunk[:, :2] * env._ACTION_SCALE, axis=0
                            ).T,
                            -1,
                            1,
                        )
                        ax.plot(*traj, "o-", alpha=0.5)

                    ax.legend()
                    fig.show()

                    # Plot covariance matrix
                    fig_cov, ax_cov = plt.subplots()
                    max_cov = np.max(np.abs(cov))
                    img = ax_cov.matshow(
                        cov, vmin=-max_cov, vmax=max_cov, cmap="seismic"
                    )
                    fig_cov.colorbar(img)
                    ax_cov.set_title("Action Distribution Covariance Matrix")
                    fig_cov.show()
                    plt.draw_all()
                    plt.show(block=True)
                    exit = True
                    break

                print(
                    f"Step {step}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}"
                )

                # Check if the episode is over
                if terminated or truncated or info["is_success"]:
                    exit = True
                    break
            if exit:
                break


@app.command()
def vplot(policy: str, n: int, ticks: int = 200):
    """
    Generate a value function plot for the environment using the trained policy.

    Args:
        policy: Path to the saved policy
        n: Checkpoint number
        ticks: Resolution of the plot grid
    """
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = SlitEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        slit_width=0.2,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=False,
    )
    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy, seed=42)
    agent.load_model(policy, n, n)

    # Create a grid of positions
    ticks = np.linspace(-1, 1, ticks)
    x, y = np.meshgrid(ticks, ticks)
    positions = np.column_stack([x.ravel(), y.ravel()])

    # Filter out positions inside the wall
    wall_mask = ~(
        (positions[:, 0] >= -0.1)
        & (positions[:, 0] <= 0.1)
        & (positions[:, 1] >= -0.4)
        & (positions[:, 1] <= 0.4)
    )
    filtered_positions = positions[wall_mask]

    # Create full observations with goal information
    # Use the fixed goal position
    goal_pos = np.array([0.875, 0.0])
    dist_to_goal = (goal_pos - filtered_positions) / 2
    obs = np.concatenate([filtered_positions, dist_to_goal], axis=1)

    # Get actions from policy and compute Q-values
    actions, _ = agent.predict(obs, deterministic=True)
    actions = np.split(actions, config.alg.actor_chunksize, axis=1)[0]
    actions = np.expand_dims(actions, 1)

    # Get Q-values from critic
    qs = np.min(agent.predict_critic(obs, actions), axis=0)[:, 1]

    # Create a visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    if config.critic.distributional:
        qs = jax.vmap(config.critic.dist.expected_value)(qs)
        assert isinstance(qs, jax.Array)

    # Plot the value function
    scatter = ax.scatter(
        filtered_positions[:, 0],
        filtered_positions[:, 1],
        c=qs.flatten(),
        cmap="viridis",
        alpha=0.7,
        s=10,
    )

    # Add a colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label("Q-Value")

    # Draw environment boundaries
    ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], "k-", linewidth=2)

    # Draw the wall/barrier in the middle
    ax.fill_between(
        [-0.1, 0.1, 0.1, -0.1, -0.1],
        [-0.4, -0.4, 0.4, 0.4, -0.4],
        [0.4, 0.4, -0.4, -0.4, 0.4],
        color="black",
        alpha=0.7,
    )

    # Mark the goal position
    ax.scatter(goal_pos[0], goal_pos[1], color="green", s=200, marker="*", label="Goal")

    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Value Function Plot")
    ax.legend()

    fig.show()
    plt.show()


if __name__ == "__main__":
    app()
