from os import path
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from sac.sac import SAC
import typer
import yaml
import jsons
from sac.config import Config
from pathlib import Path
import jax
import jax.numpy as jnp

app = typer.Typer()

ColorType = tuple[int, int, int]


class BaseEnv(gym.Env):
    _AGENT_COLOR = (200, 0, 0)
    _AGENT_SHADOW_COLOR = (230, 14, 165)
    _GOAL_COLOR = (0, 200, 0)
    _GOAL_SHADOW_COLOR = (10, 180, 10)
    _BG_COLOR = (220, 220, 220)
    _WALL_COLOR = (35, 35, 35)

    # _HORIZON = 100
    _HORIZON = 500
    # _HORIZON = 2

    _DT = 0.05
    _DAMPING = 0.9
    _ACC_SCALE = 4.0

    _BOARDER_SIZE = 6
    _INTERNAL_IMG_SIZE = 256 - 2 * _BOARDER_SIZE

    _AGENT_SIZE = 4

    _ACTION_SCALE = 0.005

    _SUCCESS_THRESHOLD = 0.05
    _SUCCESS_BONUS_SCALE = 0.2

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
        fixed_start_goal: bool = True,
        sparse: bool = False,
        **kwargs,
    ):
        self.render_mode = render_mode
        self._rng = np.random.RandomState(seed)
        self.fixed_start_goal = fixed_start_goal
        self.sparse = sparse
        self._obs_keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
        self._img_size = img_size

        self._agent_pos_scale = self._INTERNAL_IMG_SIZE / 2 - self._AGENT_SIZE

        self._agent_pos = None
        self._agent_vel = None
        self._steps = 0
        self._initialized = False

        self._velocity_control = velocity_control

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        state_dim = 4 if self._velocity_control else 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,)
        )

        # Define the four goals: top, bottom, left, right
        self.goals = np.array(
            [
                [0.0, 1.0],  # Top
                [0.0, -1.0],  # Bottom
                [1.0, 0.0],  # Right
                [-1.0, 0.0],  # Left
            ]
        )
        self.goals = np.array(
            [
                # [-1.0, -1.0],  # TL
                [-1.0, 1.0],  # TR
                [1.0, -1.0],  # BL
                # [1.0, 1.0],  # BR
            ]
        )
        self.fig = None

    def _agent_reset(self):
        """Agent starts in the center of the environment."""
        #
        if not self.fixed_start_goal:
            # self._agent_pos = self._rng.uniform(low=[-0.1, -0.1], high=[0.1, 0.1])  # Always starts in the center
            self._agent_pos = self._rng.uniform(
                low=[-0.2, -0.2], high=[0.2, 0.2]
            )  # Always starts in the center
        else:
            self._agent_pos = np.array([0.0, 0.0])  # Always starts in the center

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

    def _get_obs(self) -> np.ndarray:
        """Observes agent position, velocity, and distance to the nearest goal."""
        # dist_to_goals = (self.goals - self._agent_pos) / 2
        # dist_to_nearest_goal = np.min(np.linalg.norm(dist_to_goals, axis=1))  # Nearest goal

        if self._velocity_control:
            # o = np.concatenate((self._agent_pos, self._agent_vel, [dist_to_nearest_goal]), axis=0)
            o = np.concatenate((self._agent_pos, self._agent_vel), axis=0)
        else:
            # o = np.concatenate((self._agent_pos, [dist_to_nearest_goal]), axis=0)
            o = self._agent_pos
        return o

    def _get_reward(self) -> float:
        """Calculates the reward based on the nearest goal."""
        # Calculate the distance to the nearest goal
        distances_to_goals = np.linalg.norm(self._agent_pos - self.goals, axis=1)
        nearest_goal_distance = np.min(distances_to_goals)

        # Normalize the reward
        # norm_reward = 1 - nearest_goal_distance / np.sqrt(8)
        norm_reward = -nearest_goal_distance
        success = float(self._get_success())
        # Normalize the reward1 -
        # return self._SUCCESS_BONUS_SCALE * success + (1 - self._SUCCESS_BONUS_SCALE) * norm_reward
        # return norm_reward
        return 10 * norm_reward

    def _get_success(self) -> bool:
        """Determines if the agent reached any of the goals."""
        distances_to_goals = np.linalg.norm(self._agent_pos - self.goals, axis=1)
        return bool(np.any(distances_to_goals < self._SUCCESS_THRESHOLD))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        assert self._initialized, "Call reset() before step()"

        self._step(action)
        self._agent_pos = np.clip(self._agent_pos, -1, 1)

        self._steps += 1
        truncated = self._steps >= self._HORIZON
        reward = self._get_reward()

        if self.sparse:
            reward *= 10
            reward *= truncated

        return (
            self._get_obs(),
            reward,
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
            # self._agent_pos += action * self._ACTION_SCALE
            self._agent_pos = (
                self._agent_pos + action * self._ACTION_SCALE
                if self._HORIZON > 5
                else action
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng = np.random.RandomState(seed)
        self._initialized = True
        self._agent_reset()
        if self._velocity_control:
            self._agent_vel = np.zeros(2)
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

        # Draw all four goals
        for goal in self.goals:
            self._draw_circle(draw, goal, self._GOAL_COLOR, self._AGENT_SIZE)

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
    # Initialize the environment
    env = BaseEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
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
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = BaseEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=True,
    )
    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy)
    agent.load_model(policy, n, n)

    trajectories = []
    success = []

    for i in range(n_trajs):
        obs, _ = env.reset(seed=i)
        current_traj = [obs]
        state = None

        exit = False
        for step in range(env._HORIZON):
            actions, state = agent.predict(obs, state=state, deterministic=False)
            for action in np.split(actions, config.alg.actor_chunksize, axis=0):
                obs, _, terminated, truncated, info = env.step(action)
                current_traj.append(obs)

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
    ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1])
    for traj in trajectories:
        ax.plot(*np.array(traj).T, "o-", alpha=1 / n_trajs)
    ax.set_title(f"Trajectories ({np.mean(success)*100:.2f}% success)")
    fig.show()
    plt.show(block=True)


@app.command()
def rollout(policy: str, n: int):
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = BaseEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
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
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = BaseEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
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
        past_obs = [obs]
        state = None
        print("Initial Observation:", obs)

        fig, ax = plt.subplots()
        ax.set_box_aspect(1)

        exit = False

        for step in range(100):
            action, state = agent.predict(obs, state, None)
            for i, action in enumerate(
                np.split(action, config.alg.actor_chunksize, axis=0)
            ):
                ax.clear()
                ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1])

                obs, reward, terminated, truncated, info = env.step(action[:2])
                past_obs.append(obs)

                # Render the current state of the environment
                ax.scatter(*obs)
                ax.set_title(
                    f'Step {step}: Reward={reward:.3f}, Success={info["is_success"]}'
                )
                plt.draw()
                plt.pause(1 / 30)

                if step + i == at:
                    # plot rollouts
                    ax.plot(*np.array(past_obs).T)
                    key, action_key = jax.random.split(action_key, 2)
                    dist, _, cov, _ = agent.policy.actor_state.apply_fn(
                        {
                            "params": agent.policy.actor_state.params,
                            "batch_stats": agent.policy.actor_state.batch_stats,
                        },
                        np.expand_dims(obs, 0),
                        train=False,
                    )
                    chunks = [
                        dist.sample(seed=key).reshape(
                            (agent.config.alg.actor_chunksize, -1)
                        )
                        for key in jax.random.split(key, n_trajs)
                    ]
                    for chunk in chunks:
                        traj = np.clip(
                            np.cumsum(obs + chunk * env._ACTION_SCALE, axis=0).T, -1, 1
                        )
                        ax.plot(*traj, "o-", alpha=0.5)

                    fig.show()
                    # plot cov
                    fig_cov, ax_cov = plt.subplots()
                    max_cov = np.max(np.abs(cov))
                    img = ax_cov.matshow(
                        cov, vmin=-max_cov, vmax=max_cov, cmap="seismic"
                    )
                    fig_cov.colorbar(img)
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
    with open(Path(policy) / "config.yaml", "r") as f:
        config = jsons.load(yaml.load(f, yaml.Loader), Config)

    # Initialize the environment
    env = BaseEnv(
        seed=0,
        obs_keys="state",
        img_size=128,
        velocity_control=False,
        render_mode="human",
        fixed_start_goal=False,
    )
    agent = SAC("MlpPolicy", env=env, config=config, model_save_path=policy)
    agent.load_model(policy, n, n)

    ticks = np.linspace(-1, 1, ticks)
    obs = np.array(np.meshgrid(ticks, ticks)).reshape(2, -1).T

    actions, _ = agent.predict(obs, deterministic=True)
    actions = np.split(actions, config.alg.actor_chunksize, axis=1)[0]
    actions = np.expand_dims(actions, 1)

    qs = np.min(agent.predict_critic(obs, actions), axis=0)[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(*obs.T, c=qs.flatten())
    ax.set_box_aspect(1)
    fig.show()
    plt.show()


if __name__ == "__main__":
    app()
