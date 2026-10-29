from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
    assert_never,
)

import flax
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
from flax.training.train_state import TrainState
from gymnasium import Env, spaces
from jaxtyping import Array, Float, jaxtyped
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    TrainFreq,
    TrainFrequencyUnit,
    RolloutReturn,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from typeguard import typechecked as typechecker

from common.off_policy_algorithm import OffPolicyAlgorithmJax
from common.type_aliases import (
    ActorTrainState,
    OGBenchDataset,
    ReplayBufferSamplesNp,
    RLTrainState,
)
from sac.config import ActorKind, Config, CriticKind
from sac.policies import SACPolicy
from sac.policy_diffusion import DiffPol
from sac.utils import nan_to_num_dict, Floaty
from sac.chunky_buffer import ChunkyBuffer
from sac.distributional import project

if TYPE_CHECKING:

    def dummy(x, *args, **kwargs):
        _ = args
        _ = kwargs
        return x

    jax.jit = dummy


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda _: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda _: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init * 1.0  # force to float


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        "MultiInputPolicy": SACPolicy,
    }

    action_space: spaces.Box  # type: ignore[assignment]
    offline_data: None | ChunkyBuffer

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        config: Config,
        model_save_path: str | None,
        dataset: Optional[OGBenchDataset] = None,
        save_every_n_steps: int = 20000,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: Floaty = 0.005,
        gamma: float = 0.0,
        train_freq: Union[int, Tuple[int, str]] = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = ChunkyBuffer,  # type: ignore
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        stats_window_size: int = 100,
    ) -> None:
        assert gamma == 0, "Please set gamma via the config, not the constructor"
        gamma = config.alg.gamma
        # Use config values as defaults, but allow overriding with method arguments
        learning_rate = config.optimizer.actor_lr
        qf_learning_rate = config.optimizer.critic_lr
        buffer_size = config.alg.buffer_size
        learning_starts = learning_starts or 100
        batch_size = batch_size or 256
        tau = config.alg.tau
        gradient_steps = config.alg.utd
        self.config = config
        assert not (
            config.alg.single_action_v and config.critic.kind != CriticKind.TRANSFORMER
        ), "single_action_v only works with a transformer critic"
        if dataset:
            assert isinstance(env, Env)
            offline_buffer_kwargs = {
                "buffer_size": max(
                    dataset["observations"].shape[0], config.alg.buffer_size
                ),
                "obs_dim": env.observation_space,
                "action_dim": env.action_space,
                "device": "cpu",
                "n_envs": 1,
                "optimize_memory_usage": False,
            }
            self.offline_data = ChunkyBuffer.from_ogbench(
                dataset, offline_buffer_kwargs
            )
        else:
            self.offline_data = None

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=dict(),
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )

        self.policy_delay = config.alg.policy_delay
        self.crossq_style = config.alg.crossq_style
        self.use_bnstats_from_live_net = False
        self.policy_q_reduce_fn = jnp.min  # TODO
        self.save_every_n_steps = save_every_n_steps
        self.model_save_path = model_save_path
        self.critic_chunksize = config.alg.critic_chunksize
        self.actor_chunksize = config.alg.actor_chunksize
        self.chunky_critic = config.alg.chunky_critic
        self.scale_chunks = config.alg.scale_chunks

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # pytype: disable=not-instantiable
            assert isinstance(self.action_space, spaces.Box)
            chunky_action_space = spaces.Box(
                np.concatenate(self.actor_chunksize * [self.action_space.low]),
                np.concatenate(self.actor_chunksize * [self.action_space.high]),
            )

            self.actor: nn.Module | None

            if self.config.actor.kind == ActorKind.DIFFUSION:
                self.policy = DiffPol(  # type: ignore[assignment]
                    observation_space=self.observation_space,
                    action_space=chunky_action_space,
                    config=self.config,
                )
                self.key = self.policy.build(self.key)
                self.actor = None
            else:
                self.policy = SACPolicy(  # type: ignore[assignment]
                    self.config,
                    observation_space=self.observation_space,
                    action_space=chunky_action_space,
                )
                self.key = self.policy.build(self.key)
                self.actor = self.policy.actor  # type: ignore[assignment]
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if self.config.alg.ent_coef == "auto":
                # Default initial value of ent_coef when learned
                ent_coef_init = self.config.alg.ent_coef_init * 1.0
                assert (
                    ent_coef_init > 0.0
                ), "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                self.ent_coef = ConstantEntropyCoef(self.config.alg.ent_coef * 1.0)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,  # type: ignore
                ),
            )

        # automatically set target entropy if needed
        if self.config.alg.target_ent == "auto":
            self.target_entropy = float(
                -np.prod(self.action_space.shape).astype(np.float32)
                * self.actor_chunksize
            )
        else:
            self.target_entropy = float(self.config.alg.target_ent)

    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert (
            self.env is not None
        ), "You must set the environment before calling learn()"
        assert self.replay_buffer is not None, "You can't learn without a replay buffer"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()
        assert isinstance(self, SAC)

        # SAC Flow: Offline pre-training phase
        if self.config.alg.offline_steps > 0:
            if self.offline_data is None:
                raise ValueError(
                    "SAC Flow mode (offline_steps > 0) requires offline data. "
                    "Ensure a dataset is provided (e.g., via OGBench environment)."
                )
            self.replay_buffer = self.offline_data

        while self.num_timesteps < total_timesteps:
            last_chunk_bias_factor = max(
                0,
                (self.config.critic.last_chunk_bias_decay - self.num_timesteps)
                / self.config.critic.last_chunk_bias_decay,
            )

            last_chunk_bias = (
                last_chunk_bias_factor * self.config.critic.last_chunk_bias
            )

            if self.num_timesteps >= self.config.alg.offline_steps:
                # ONLINE PHASE: collect rollouts and train
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,  # type: ignore
                    log_interval=log_interval,
                )

                if not rollout.continue_training:
                    break

                # Training logic for online phase
                if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = (
                        self.gradient_steps
                        if self.gradient_steps >= 0
                        else rollout.episode_timesteps
                    )
                    # Special case when the user passes `gradient_steps=0`
                    if gradient_steps > 0:
                        self.train(
                            batch_size=self.batch_size,
                            gradient_steps=gradient_steps,
                            last_chunk_bias=last_chunk_bias,
                        )
            else:
                # OFFLINE PHASE: train on offline data without collecting rollouts
                # Increment num_timesteps by 1 (represents 1 gradient step)
                self.num_timesteps += 1

                # Perform 1 gradient step
                if self.num_timesteps > self.learning_starts:
                    self.train(
                        batch_size=self.batch_size,
                        gradient_steps=1,
                        last_chunk_bias=last_chunk_bias,
                    )

                # Manually trigger callbacks to enable logging and evaluation
                callback.update_locals(locals())
                if not callback.on_step():
                    break

                # Log training metrics periodically during offline phase
                if log_interval is not None and self.num_timesteps % log_interval == 0:
                    self._dump_logs()

        callback.on_training_end()

        return self

    def train(self, gradient_steps, batch_size, last_chunk_bias: float = 0):
        critic_chunksize = self.critic_chunksize
        # Sample all at once for efficiency (so we can jit the for loop)
        assert self.replay_buffer is not None, "You can't train without a replay buffer"
        n_samples = batch_size * gradient_steps
        if self.offline_data and self.config.alg.rlpd:
            n_samples = n_samples // 2

        self.replay_buffer: ChunkyBuffer
        data_chunky = self.replay_buffer.sample_chunk(
            critic_chunksize,
            n_samples,
            env=self._vec_normalize_env,
            bias=last_chunk_bias,
        )

        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {
            i: True
            for i in range(gradient_steps)
            if ((self._n_updates + i + 1) % self.policy_delay) == 0
        }
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if self.offline_data and self.config.alg.rlpd:
            data_offline = self.offline_data.sample_chunk(
                critic_chunksize,
                n_samples,
                env=self._vec_normalize_env,
            )
            data_chunky = ReplayBufferSamplesNp.combine(data_chunky, data_offline)

        if self.config.actor.kind == ActorKind.DIFFUSION:
            self.policy: DiffPol
            sampler = self.policy.sampler
        else:
            sampler = None

        # Check if we're in offline phase with skip_rl mode
        skip_rl = (
            self.config.alg.skip_rl
            and self.num_timesteps <= self.config.alg.offline_steps
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train_with_chunky_critic(
            self.crossq_style,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data_chunky,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            config=self.config,
            sampler=sampler,
            critic_chunksize=critic_chunksize,
            skip_rl=skip_rl,
        )

        if self.model_save_path is not None:
            if (self.num_timesteps % self.save_every_n_steps == 0) or (
                self.num_timesteps == (self.learning_starts + 1)
            ):
                self._save_model()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/last_chunk_bias", last_chunk_bias)
        for k, v in log_metrics.items():
            self.logger.record(f"train/{k}", v.item())

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        if self.use_sde and self.actor is not None:
            self.actor.reset_noise(env.num_envs)  # type: ignore

        callback.on_rollout_start()
        continue_training = True
        is_start_of_episode = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                if self.actor is not None:
                    self.actor.reset_noise(env.num_envs)  # type: ignore

            # Select action randomly or according to policy
            chunky_actions, chunky_buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            if self.config.alg.random_chunks or (
                self.config.alg.random_init_chunk and is_start_of_episode
            ):
                self.key, rollout_key = jax.random.split(self.key, 2)
                rollout_len = jax.random.randint(
                    rollout_key, (1,), 1, self.config.alg.actor_chunksize
                )[0]
            elif self.config.alg.temporl:
                # Evaluate Q-values for action chunk and select prefix with highest Q-value
                assert (
                    self.config.critic.distributional
                ), "TempoRL currently only works with dist critic (because of q-val dim)"
                # calculate some shape constants
                batch_size = 1
                assert self.observation_space.shape is not None
                obs_dim = sum(self.observation_space.shape)
                action_dim = self.action_space.shape[0]

                chunky_buffer_actions_jax = jnp.array(chunky_buffer_actions).reshape(
                    (batch_size, self.config.alg.actor_chunksize, action_dim)
                )

                current_obs = jnp.array(self._last_obs)
                assert current_obs.shape == (batch_size, obs_dim)

                # Get Q-values for the action chunk
                q_dist = self.policy.qf_state.apply_fn(
                    {
                        "params": self.policy.qf_state.params,
                        "batch_stats": self.policy.qf_state.batch_stats,
                    },
                    current_obs,
                    chunky_buffer_actions_jax,
                    train=False,
                )

                n_critics = self.config.critic.n_critics

                assert q_dist.shape == (
                    n_critics,
                    batch_size,
                    self.config.alg.critic_chunksize,
                    self.config.critic.dist.n_atoms,
                )

                q_values = jax.vmap(
                    jax.vmap(jax.vmap(self.config.critic.dist.expected_value))
                )(
                    q_dist
                )  # type: ignore
                q_values: Array
                assert q_values.shape == (
                    n_critics,
                    batch_size,
                    self.config.alg.critic_chunksize,
                )

                # pop n_critics dim
                q_values_min = jnp.mean(q_values, axis=0)
                assert q_values_min.shape == (
                    batch_size,
                    self.config.alg.critic_chunksize,
                )

                rollout_len = 1 + int(jnp.argmax(q_values_min[0]))

                # Log the chosen chunk size to wandb
                if hasattr(self, "logger") and self.logger:
                    self.logger.record("rollout/temporl_chunk_size", rollout_len)
            else:
                rollout_len = self.config.alg.actor_chunksize

            is_start_of_episode = False

            for actions, buffer_actions in zip(
                np.split(chunky_actions, self.actor_chunksize, axis=1)[:rollout_len],
                np.split(chunky_buffer_actions, self.actor_chunksize, axis=1)[
                    :rollout_len
                ],
            ):
                assert actions.shape == buffer_actions.shape
                # Rescale and perform action
                new_obs, rewards, dones, infos = env.step(actions)

                self.num_timesteps += env.num_envs
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if not callback.on_step():
                    return RolloutReturn(
                        episode_timesteps=num_collected_steps * env.num_envs,
                        n_episodes=num_collected_episodes,
                        continue_training=False,
                    )

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

                self._update_current_progress_remaining(
                    self.num_timesteps, self._total_timesteps
                )

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is dones as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                for idx, done in enumerate(dones):
                    if done:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1
                        # Mark that the next chunk will be at the start of a new episode
                        is_start_of_episode = True

                        if action_noise is not None:
                            kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                            action_noise.reset(**kwargs)

                        # Log training infos
                        if (
                            log_interval is not None
                            and self._episode_num % log_interval == 0
                        ):
                            self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            episode_timesteps=num_collected_steps * env.num_envs,
            n_episodes=num_collected_episodes,
            continue_training=continue_training,
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "crossq_style",
            "use_bnstats_from_live_net",
        ],
    )
    def update_critic(
        crossq_style: bool,
        use_bnstats_from_live_net: bool,
        gamma: Floaty,
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key,
        actor_chunksize: int,
    ):
        (
            key,
            noise_key,
            dropout_key_target,
            dropout_key_current,
            redq_key,
        ) = jax.random.split(key, 5)
        # sample action from the actor
        dist = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            next_observations,
            train=False,
        )
        next_state_actions_chunk = dist.sample(
            seed=noise_key
        )  # TODO this is kinda stupid

        next_log_prob = dist.log_prob(
            next_state_actions_chunk
        )  # TODO, this should be a marginal likelihood

        next_state_actions = jnp.split(
            next_state_actions_chunk, actor_chunksize, axis=1
        )[0]

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        # for next_state_actions in jnp.split(next_state_actions_chunk, critic_chunksize, axis=1):
        # TODO can we do more than one step here?

        def mse_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params,
                        "batch_stats": (
                            qf_state.target_batch_stats
                            if not use_bnstats_from_live_net
                            else batch_stats
                        ),
                    },
                    next_observations,
                    next_state_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False,
                )

                # shape is (n_critics, batch_size, 1)
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    observations,
                    actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([observations, next_observations], axis=0),
                    jnp.concatenate([actions, next_state_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
            assert isinstance(current_q_values, Float[Array, "ncritics batch 1"])  # type: ignore

            if next_q_values.shape[0] > 2:  # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2
                next_q_values = jax.random.choice(
                    redq_key, next_q_values, (m_critics,), replace=False, axis=0
                )

            next_q_values = jnp.min(next_q_values, axis=0)
            next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(
                -1, 1
            )
            target_q_values = (
                rewards.reshape(-1, 1)
                + (1 - dones.reshape(-1, 1)) * gamma * next_q_values
            )
            assert isinstance(target_q_values, Float[Array, "batch 1"])  # type: ignore

            loss = (
                0.5
                * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2)
                .mean(axis=1)
                .sum()
            )
            return loss, (state_updates, current_q_values, next_q_values)

        (
            qf_loss_value,
            (state_updates, current_q_values, next_q_values),
        ), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats, dropout_key_current
        )

        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            "critic_loss": qf_loss_value,
            "ent_coef": ent_coef_value,
            "current_q_values": current_q_values.mean(),
            "next_q_values": next_q_values.mean(),
        }

        return (qf_state, metrics, key)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @partial(
        jax.jit,
        static_argnames=[
            "chunky_critic",
            "config",
            "critic_chunksize",
            "skip_rl",
        ],
    )
    def update_actor(
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        chunky_observations: Float[Array, "batch chunkyobs"],
        chunky_actions: Float[Array, "batch chunkyaction"],
        key,
        critic_chunksize: int,
        chunky_critic: bool,
        config: Config,
        skip_rl: bool = False,
    ):
        batch_size, chunky_action = chunky_actions.shape
        (
            key,
            dropout_key,
        ) = jax.random.split(key, 2)

        observations = jnp.split(chunky_observations, critic_chunksize, axis=1)

        def actor_loss(params, batch_stats):
            (
                actor_actions,
                log_probs,
                _,
                cov_bias,
            ), state_updates = actor_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                observations[0],
                mutable=["batch_stats"],
                train=True,
                deterministic=False,
                key=key,
            )
            assert isinstance(actor_actions, Float[Array, "batch chunkyaction"])  # type: ignore
            proj_loss = 0
            off_diag_cov = 0
            actor_loss = jnp.zeros((batch_size, 1))
            bc_loss = 0
            cum_log_prob = jnp.zeros((batch_size, 1))
            nonfinite_action_logprobs = 0.0

            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            if config.alg.train_bc:
                # BC loss: compute MSE between predicted and actual actions
                assert (
                    chunky_actions.shape == actor_actions.shape
                )  # = (batch, chunkyaction)
                # Compute squared L2 norm per sample: ||a^θ - a||_2^2
                bc_diff_squared = jnp.sum((chunky_actions - actor_actions) ** 2, axis=1)
                assert bc_diff_squared.shape == (batch_size,)
                bc_loss_per_sample = bc_diff_squared.reshape(batch_size, 1)
                actor_loss = config.actor.bc_coef * bc_loss_per_sample
                bc_loss = bc_diff_squared.mean()

            if not skip_rl:
                # RL loss: Q-value maximization + entropy regularization
                if config.alg.marginalize_actor_logprob:
                    discount = config.alg.gamma ** jnp.arange(
                        config.alg.actor_chunksize
                    )
                    assert log_probs.shape == (batch_size, config.alg.actor_chunksize)
                    cum_log_prob = discount * log_probs
                    assert cum_log_prob.shape == (
                        batch_size,
                        config.alg.actor_chunksize,
                    )
                    nonfinite_action_logprobs = 1 - jnp.mean(jnp.isfinite(cum_log_prob))
                    # jax.debug.print("Nonfinite action logprobs: {}", nonfinite_action_logprobs)
                    cum_log_prob = jnp.nan_to_num(
                        cum_log_prob, nan=0, posinf=0, neginf=0
                    )
                    cum_log_prob = cum_log_prob.sum(axis=1).reshape(batch_size, 1)
                else:
                    raise NotImplementedError()

                qf_pi = qf_state.apply_fn(
                    {
                        "params": qf_state.params,
                        "batch_stats": qf_state.batch_stats,
                    },
                    observations[0],
                    actor_actions.reshape(batch_size, config.alg.actor_chunksize, -1),
                    rngs={"dropout": dropout_key},
                    train=False,
                )
                if config.critic.distributional:
                    if config.actor.target_quartile is None:
                        qf_pi = jax.vmap(
                            jax.vmap(jax.vmap(config.critic.dist.expected_value))
                        )(qf_pi)
                    else:
                        qf_pi = jax.vmap(
                            jax.vmap(
                                jax.vmap(
                                    partial(
                                        config.critic.dist.quartile,
                                        q=config.actor.target_quartile,
                                    )
                                )
                            )
                        )(qf_pi)

                    assert isinstance(qf_pi, Array)
                else:
                    qf_pi = qf_pi[:, :, 1:]

                assert isinstance(qf_pi, Float[Array, "ncritics batch chunk"])  # type: ignore
                collate_fn = config.actor.update_collate_fn.jax_fn()
                qf_pi = collate_fn(qf_pi, axis=0)
                assert isinstance(qf_pi, Float[Array, "batch chunk"])  # type: ignore
                if config.actor.only_fit_last_nstep:
                    actor_loss -= qf_pi[:, -1].reshape(batch_size, 1)
                else:
                    actor_loss -= qf_pi.mean(axis=1).reshape(batch_size, 1)

                actor_loss += ent_coef_value * cum_log_prob

            actor_loss += proj_loss
            assert actor_loss.shape == (batch_size, 1)

            return actor_loss.mean(), (
                -cum_log_prob.mean(),
                bc_loss,
                proj_loss,
                off_diag_cov,
                nonfinite_action_logprobs,
                state_updates,
                cov_bias,
            )

        (
            actor_loss_value,
            (
                entropy,
                bc_loss,
                proj_loss,
                off_diag_cov,
                nonfinite_action_logprobs,
                state_updates,
                cov_bias,
            ),
        ), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params, actor_state.batch_stats
        )
        actor_state = actor_state.replace(old_params=actor_state.params)
        actor_state = actor_state.apply_gradients(grads=nan_to_num_dict(grads))
        actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])

        return (
            actor_state,
            qf_state,
            actor_loss_value,
            key,
            entropy,
            bc_loss,
            proj_loss,
            off_diag_cov,
            nonfinite_action_logprobs,
            cov_bias,
            optax.global_norm(grads),
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @partial(
        jax.jit,
        static_argnames=[
            "crossq_style",
            "use_bnstats_from_live_net",
            "sampler",
            "config",
        ],
    )
    def update_chunky_crossq_critic(
        crossq_style: bool,
        use_bnstats_from_live_net: bool,
        gamma: Floaty,
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        chunky_observations: Float[Array, "batch chunkyobs"],
        chunky_actions: Float[Array, "batch chunkyaction"],
        chunky_rewards: Float[Array, "batch chunk"],
        chunky_dones: Float[Array, "batch chunk"],
        chunky_truncated: Float[Array, "batch chunk"],
        chunky_next_observations: Float[Array, "batch chunkyobs"],
        key,
        sampler,
        config: Config,
    ):
        assert config.alg.apply_next_state_ent_bonus, "Not implemented"
        batch_size, chunk_size = chunky_rewards.shape
        critic_chunksize = chunk_size
        assert chunk_size == critic_chunksize
        action_size = chunky_actions.shape[1] // chunk_size
        obs_size = chunky_next_observations.shape[1] // chunk_size
        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
        n_atoms = config.critic.dist.n_atoms
        n_critics = config.critic.n_critics
        assert config.critic.distributional
        assert not config.alg.scale_chunks
        assert not config.alg.use_next_action_logprob
        assert not config.critic.only_fit_last_nstep

        key, next_action_key, current_dropout_key, next_dropout_key = jax.random.split(
            key, 4
        )

        chunky_observations = chunky_observations.reshape(
            (batch_size, chunk_size, obs_size)
        )
        first_observations = chunky_observations[:, 0, :]

        assert first_observations.shape == (batch_size, obs_size)
        chunky_next_observations = chunky_next_observations.reshape(
            (batch_size * chunk_size, obs_size)
        )

        all_observations = jnp.concatenate(
            [first_observations, chunky_next_observations], axis=0
        )
        assert all_observations.shape == ((1 + chunk_size) * batch_size, obs_size)

        chunky_actions = chunky_actions.reshape((batch_size, chunk_size, action_size))
        next_chunky_actions, next_chunky_logprobs = SAC.sample_action(
            chunky_next_observations,
            actor_state,
            key,
            sampler,
            config,
        )
        next_chunky_logprobs = next_chunky_logprobs.reshape((batch_size, chunk_size))
        next_chunky_actions = next_chunky_actions.reshape(
            (batch_size * chunk_size, chunk_size, action_size)
        )

        all_actions = jnp.concatenate([chunky_actions, next_chunky_actions], axis=0)

        assert all_actions.shape == (
            (1 + chunk_size) * batch_size,
            chunk_size,
            action_size,
        )

        dropout_key, key = jax.random.split(key, 2)

        def dist_loss(params, batch_stats):
            all_q, state_updates = qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                all_observations,
                all_actions,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
                train=True,
            )

            assert all_q.shape == (
                n_critics,
                (1 + chunk_size) * batch_size,
                chunk_size,
                n_atoms,
            )

            current_q = all_q[:, :batch_size, :, :]
            assert current_q.shape == (n_critics, batch_size, chunk_size, n_atoms)

            next_q = all_q[
                :,  # critics
                batch_size:,  # skip current_q
                -1,  # only use full-chunk value for next-state value
                :,  # atoms,
            ]
            assert next_q.shape == (n_critics, chunk_size * batch_size, n_atoms)
            next_q = next_q.reshape(n_critics, batch_size, chunk_size, n_atoms)

            # Move n_critics to the back: (batch, chunk, n_critics, n_atoms)
            next_q = jnp.transpose(next_q, (1, 2, 0, 3))

            def min_dist(
                critic_dists: Float[Array, "n_critics n_atoms"],
            ) -> Float[Array, "n_atoms"]:
                """Select distribution with minimum expected value"""
                expected_values = jax.vmap(config.critic.dist.expected_value)(
                    critic_dists
                )
                min_idx = jnp.argmin(expected_values)
                return critic_dists[min_idx]

            # Apply min_dist over batch and chunk dimensions
            next_q = jax.vmap(jax.vmap(min_dist))(next_q)
            assert next_q.shape == (batch_size, chunk_size, n_atoms)

            discounted_reward = gamma ** jnp.arange(chunk_size) * chunky_rewards
            discounted_accumulated_reward = jnp.cumsum(discounted_reward, axis=1)

            next_state_v_mask = 1.0 - jnp.logical_and(
                chunky_dones, jnp.logical_not(chunky_truncated)
            )
            next_state_v_scale = next_state_v_mask * gamma ** jnp.arange(
                1, critic_chunksize + 1
            )
            assert (
                (batch_size, critic_chunksize)
                == next_state_v_scale.shape
                == discounted_accumulated_reward.shape
                == next_chunky_logprobs.shape
            )

            batched_project = jax.vmap(
                project,
                in_axes=[
                    0,  # weights
                    0,  # add
                    0,  # scale
                    None,  # dist
                ],
            )
            chunky_project = jax.vmap(
                batched_project,
                in_axes=[
                    0,  # weights
                    0,  # add
                    0,  # scale
                    None,  # dist
                ],
            )
            # project(Q) return (scale * Q) + add
            # the critic target is reward + gamma * (Q + alpha * H)
            # we can rearrange this into (gamma * Q) + (reward + gamma * alpha * H)
            # Note that in N-step returns, reward is the N-step reward and gamma is actually
            # gamma**N, which we can access as next_action_scale in this function. next_action_scale
            # also takes care of masking away the next-state V on termination.
            q_target = chunky_project(
                next_q,  # weights
                discounted_accumulated_reward
                - ent_coef_value * next_state_v_scale * next_chunky_logprobs,  # add
                next_state_v_scale,  # scale
                config.critic.dist,  # dist
            )
            q_target = jax.lax.stop_gradient(q_target)

            q_target_expanded = jnp.broadcast_to(
                q_target,
                (config.critic.n_critics, batch_size, critic_chunksize, n_atoms),
            )
            assert q_target_expanded.shape == current_q.shape

            if config.critic.use_cramer_von_mises:
                diff = SAC.batch_cramer_von_mises(q_target_expanded, current_q)
            else:
                diff = SAC.batch_cross_entropy(q_target_expanded, current_q)

            assert isinstance(diff, Array)
            assert diff.shape == (config.critic.n_critics, batch_size, chunk_size)

            loss = (
                diff.mean(axis=2)  # over chunk
                .mean(axis=1)  # over batch
                .sum()  # over n_critics
            )

            # calculate logging values
            current_q = jnp.mean(
                jax.vmap(jax.vmap(jax.vmap(config.critic.dist.expected_value)))(
                    current_q
                )
            )
            next_q = jnp.mean(
                jax.vmap(jax.vmap(config.critic.dist.expected_value))(q_target)
            )

            # calculate q_target statistics for logging
            q_target_values = jax.vmap(jax.vmap(config.critic.dist.expected_value))(
                q_target
            )
            q_target_min = jnp.min(q_target_values)
            q_target_max = jnp.max(q_target_values)
            q_target_mean = jnp.mean(q_target_values)

            return loss, (
                state_updates,
                current_q,
                next_q,
                q_target_min,
                q_target_max,
                q_target_mean,
            )

        (
            qf_loss_value,
            (
                state_updates,
                current_q_values,
                next_q_values,
                q_target_min,
                q_target_max,
                q_target_mean,
            ),
        ), grads = jax.value_and_grad(dist_loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats
        )

        qf_state = qf_state.apply_gradients(grads=nan_to_num_dict(grads))
        if config.critic.simbav2:
            qf_state = qf_state.l2normalize()
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            "critic_grad_norm": optax.global_norm(grads),
            "critic_param_norm": optax.global_norm(qf_state.params),
            "critic_loss": qf_loss_value,
            "ent_coef": ent_coef_value,
            "current_q_values": current_q_values.mean(),
            "next_q_values": next_q_values.mean(),
            "q_target_min": q_target_min,
            "q_target_max": q_target_max,
            "q_target_mean": q_target_mean,
            "nonfinite_next_state_logprobs": 0,
        }

        return (qf_state, metrics, key)

    @staticmethod
    def sample_action(
        next_observation: Float[Array, "batch obs"],
        actor_state: ActorTrainState,
        key,
        sampler,
        config: Config,
    ) -> Tuple[
        Float[Array, " batch chunk action"],  # actions
        Float[Array, " batch chunk"],  # logprobs
    ]:
        batch_size = next_observation.shape[0]

        discount = jnp.mean(
            config.alg.gamma ** jnp.arange(0, config.alg.actor_chunksize)
        )
        if config.actor.kind == ActorKind.DIFFUSION:
            next_chunky_action, run_costs, sto_costs, terminal_costs, _, _ = (
                DiffPol.sample_action(
                    actor_state,
                    actor_state.params,
                    next_observation,
                    key,
                    sampler,
                )
            )
            # as we can't marginalize the log probs for each step, we just use an average
            discount = discount.mean(axis=1)
            next_action_logprob = discount * jnp.sum(
                run_costs + sto_costs + terminal_costs, axis=1
            )
        else:
            (
                next_chunky_action,
                next_chunky_logprobs,
                _,
                _,
            ), _ = actor_state.apply_fn(  # TODO: Update state
                {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                # unroll the observations for the actor
                next_observation,
                mutable=["batch_stats"],
                train=False,
                deterministic=False,
                key=key,
            )
            next_action_logprob = jnp.sum(next_chunky_logprobs * discount, axis=1)
        assert next_action_logprob.shape == (batch_size,)
        return next_chunky_action, next_action_logprob

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0))  # over n_critics
    @partial(jax.vmap, in_axes=(0, 0))  # over batch
    @partial(jax.vmap, in_axes=(0, 0))  # over chunk
    @jaxtyped(typechecker=typechecker)
    def batch_cross_entropy(
        m: Float[Array, " n_atoms"], p: Float[Array, " n_atoms"]
    ) -> Floaty:
        assert m.shape == p.shape
        return -jnp.sum(m * jnp.log(p + 1e-15))

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0))  # over n_critics
    @partial(jax.vmap, in_axes=(0, 0))  # over batch
    @partial(jax.vmap, in_axes=(0, 0))  # over chunk
    @jaxtyped(typechecker=typechecker)
    def batch_cramer_von_mises(
        m: Float[Array, " n_atoms"], p: Float[Array, " n_atoms"]
    ) -> Floaty:
        m_cdf = jnp.cumsum(m)
        p_cdf = jnp.cumsum(p)
        diff = m_cdf - p_cdf
        return jnp.sum(diff**2)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @partial(
        jax.jit,
        static_argnames=[
            "crossq_style",
            "use_bnstats_from_live_net",
            "sampler",
            "config",
        ],
    )
    def update_chunky_critic(
        crossq_style: bool,
        use_bnstats_from_live_net: bool,
        gamma: Floaty,
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        chunky_observations: Float[Array, "batch chunkyobs"],
        chunky_actions: Float[Array, "batch chunkyaction"],
        chunky_rewards: Float[Array, "batch chunk"],
        chunky_dones: Float[Array, "batch chunk"],
        chunky_truncated: Float[Array, "batch chunk"],
        chunky_next_observations: Float[Array, "batch chunkyobs"],
        key,
        sampler,
        config: Config,
    ):
        """
        implements a chunky version of the critic update, following Eq (7) and (8) of the TOP-ERL paper.

        """
        batch_size, chunk_size = chunky_rewards.shape
        critic_chunksize = chunk_size
        assert chunk_size == critic_chunksize
        action_size = chunky_actions.shape[1] // chunk_size
        obs_size = chunky_next_observations.shape[1] // chunk_size
        assert not config.alg.scale_chunks
        assert not config.alg.use_next_action_logprob

        key, next_action_key, current_dropout_key, next_dropout_key = jax.random.split(
            key, 4
        )

        first_observation = jnp.split(chunky_observations, chunk_size, axis=1)[0]
        chunky_actions = chunky_actions.reshape((batch_size, chunk_size, action_size))
        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        next_state_v_mask = jnp.logical_not(chunky_dones)
        next_state_v_scale = next_state_v_mask * gamma ** jnp.arange(
            1, critic_chunksize + 1
        )
        assert next_state_v_scale.shape == (batch_size, critic_chunksize)

        discounted_reward = gamma ** jnp.arange(chunk_size) * chunky_rewards

        discounted_accumulated_reward = jnp.cumsum(discounted_reward, axis=1)

        nonfinite_next_state_logprobs = 0
        chunky_next_observations = chunky_next_observations.reshape(
            (batch_size, critic_chunksize, obs_size)
        )
        if config.critic.only_fit_last_nstep:
            chunky_next_observations = chunky_next_observations[:, -1, :].reshape(
                (batch_size, 1, obs_size)
            )
            next_state_v_scale = next_state_v_scale[:, -1].reshape((batch_size, 1))
            discounted_accumulated_reward = discounted_accumulated_reward[
                :, -1
            ].reshape((batch_size, 1))

        q_target = SAC.calculate_q_target(
            chunky_next_observations,
            next_state_v_scale,
            discounted_accumulated_reward,
            actor_state,
            qf_state,
            ent_coef_value if config.alg.apply_next_state_ent_bonus else 0.0,
            next_action_key,
            use_bnstats_from_live_net,
            sampler,
            action_size,
            config,
        )
        assert isinstance(q_target, Array)

        if config.critic.distributional:
            n_atoms = config.critic.dist.n_atoms

            def dist_loss(params, batch_stats, dropout_key):
                nonlocal critic_chunksize
                q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    first_observation,
                    chunky_actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                if config.critic.only_fit_last_nstep:
                    q_values = q_values[:, :, -1, :]
                    q_values = q_values.reshape(
                        (
                            config.critic.n_critics,
                            batch_size,
                            1,
                            n_atoms,
                        )
                    )
                    critic_chunksize = 1  # <-- keep in mind for the following asserts

                assert isinstance(q_target, Array)
                assert q_target.shape == (batch_size, critic_chunksize, n_atoms)
                assert q_values.shape == (
                    config.critic.n_critics,
                    batch_size,
                    critic_chunksize,
                    n_atoms,
                )

                # Expand q_target to match all critics
                q_target_expanded = jnp.expand_dims(q_target, axis=0)
                q_target_expanded = jnp.broadcast_to(
                    q_target_expanded,
                    (config.critic.n_critics, batch_size, critic_chunksize, n_atoms),
                )
                assert q_target_expanded.shape == q_values.shape

                if config.critic.use_cramer_von_mises:
                    diff = SAC.batch_cramer_von_mises(q_target_expanded, q_values)
                else:
                    diff = SAC.batch_cross_entropy(q_target_expanded, q_values)

                assert isinstance(diff, Array)
                assert diff.shape == (
                    config.critic.n_critics,
                    batch_size,
                    critic_chunksize,
                )

                loss = (
                    diff.mean(axis=2)  # over chunk
                    .mean(axis=1)  # over batch
                    .sum()  # over n_critics
                )

                # calculate logging values
                current_q = jax.vmap(
                    jax.vmap(jax.vmap(config.critic.dist.expected_value))
                )(q_values)
                assert isinstance(q_target, Array)
                next_q = jax.vmap(jax.vmap(config.critic.dist.expected_value))(q_target)
                if config.critic.only_fit_last_nstep:
                    assert isinstance(current_q, Array)
                    assert isinstance(next_q, Array)
                    assert current_q.shape == (
                        config.critic.n_critics,
                        batch_size,
                        critic_chunksize,
                    )
                    assert next_q.shape == (batch_size, critic_chunksize)
                    current_q = current_q[:, :, -1]
                    next_q = next_q[:, -1]

                return loss, (state_updates, current_q, next_q)

            loss = dist_loss

        else:  # non-distributional critic

            def mse_loss(params, batch_stats, dropout_key):
                nonlocal critic_chunksize
                nonlocal q_target
                q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    first_observation,
                    chunky_actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                if config.critic.only_fit_last_nstep:
                    q_values = q_values[:, :, -1].reshape(
                        (config.critic.n_critics, batch_size, 1)
                    )
                    critic_chunksize = 1  # <-- keep in mind for the following asserts
                else:
                    q_values = q_values[:, :, 1:]  # Drop V

                assert q_values.shape == (
                    config.critic.n_critics,
                    batch_size,
                    critic_chunksize,
                )

                diff = q_target - q_values
                assert isinstance(q_target, Array)
                assert q_target.shape == (batch_size, critic_chunksize)
                assert diff.shape == (
                    config.critic.n_critics,
                    batch_size,
                    critic_chunksize,
                )

                loss = (
                    (diff**2)
                    .mean(axis=2)  # over chunk
                    .mean(axis=1)  # over batch
                    .mean()  # over n_critics
                )
                assert loss.size == 1
                return loss, (state_updates, q_values[:, 1:], q_target)

            loss = mse_loss

        (
            qf_loss_value,
            (state_updates, current_q_values, next_q_values),
        ), grads = jax.value_and_grad(loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats, current_dropout_key
        )

        qf_state = qf_state.apply_gradients(grads=nan_to_num_dict(grads))
        if config.critic.simbav2:
            qf_state = qf_state.l2normalize()
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            "critic_grad_norm": optax.global_norm(grads),
            "critic_param_norm": optax.global_norm(qf_state.params),
            "critic_loss": qf_loss_value,
            "ent_coef": ent_coef_value,
            "current_q_values": current_q_values.mean(),
            "next_q_values": next_q_values.mean(),
            "nonfinite_next_state_logprobs": nonfinite_next_state_logprobs,
        }

        return (qf_state, metrics, key)

    @staticmethod
    @partial(
        jax.vmap, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None)
    )  # map over batch
    @partial(
        jax.vmap, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None)
    )  # map over chunk
    def calculate_q_target(
        next_observation: Float[Array, " obs"],
        next_state_v_scale: Floaty,
        reward: Floaty,
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_value: Floaty,
        key,
        use_bnstats_from_live_net: bool,
        sampler,
        action_dim: int,
        config: Config,
    ) -> Union[Floaty, Float[Array, " n_atoms"]]:
        actor_chunksize = config.alg.actor_chunksize
        gamma = config.alg.gamma
        n_critics = config.critic.n_critics
        n_atoms = config.critic.dist.n_atoms

        assert len(next_observation.shape) == 1
        next_observation = jnp.expand_dims(
            next_observation, 0
        )  # flax expects a batch dim

        # calculate next-state action and its (approximate) logprob

        next_action_key, key = jax.random.split(key, 2)
        if config.actor.kind == ActorKind.DIFFUSION:
            next_chunky_action, run_costs, sto_costs, terminal_costs, _, _ = (
                DiffPol.sample_action(
                    actor_state,
                    actor_state.params,
                    next_observation,
                    next_action_key,
                    sampler,
                )
            )
            # as we can't marginalize the log probs for each step, we just use an average
            discount = jnp.mean(gamma ** jnp.arange(0, actor_chunksize))
            next_action_logprob = discount * jnp.sum(
                run_costs + sto_costs + terminal_costs
            )
        else:
            (
                next_chunky_action,
                next_chunky_logprobs,
                _,
                _,
            ), _ = actor_state.apply_fn(  # TODO: Update state
                {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                # unroll the observations for the actor
                next_observation,
                mutable=["batch_stats"],
                train=False,
                deterministic=False,
                key=next_action_key,
                single_action=config.alg.single_action_v,
            )
            discount = gamma ** jnp.arange(0, actor_chunksize)
            next_action_logprob = jnp.sum(next_chunky_logprobs * discount)

        # calculate the q-value of the next-state action

        next_action = next_chunky_action.reshape(1, actor_chunksize, action_dim)
        if config.alg.single_action_v:
            next_action = next_action[0, 0, :].reshape(1, 1, action_dim)
            actor_chunksize = 1

        dropout_key, key = jax.random.split(key, 2)
        q_value = qf_state.apply_fn(
            {
                "params": qf_state.target_params,
                "batch_stats": (
                    qf_state.target_batch_stats
                    if not use_bnstats_from_live_net
                    else qf_state.batch_stats
                ),
            },
            next_observation,
            next_action,
            rngs={"dropout": dropout_key},
            train=False,
        )
        q_value = jnp.squeeze(q_value, 1)  # remove batch axis
        if config.critic.distributional:
            assert q_value.shape == (n_critics, actor_chunksize, n_atoms)
        else:
            assert q_value.shape == (n_critics, actor_chunksize + 1)

        # choose a RLPD-style random subset of the critics
        if n_critics != config.critic.update_ensemble_size:
            subset_key, key = jax.random.split(key, 2)
            critic_i = jax.random.choice(
                key=subset_key,
                a=jnp.arange(n_critics),
                shape=(config.critic.update_ensemble_size,),
                replace=False,
            )
            q_value = q_value[critic_i]

        if config.critic.distributional:
            assert q_value.shape == (
                config.critic.update_ensemble_size,
                actor_chunksize,
                n_atoms,
            )

            # Collate the last-action q dist according to config.critic.collate_fn
            q_value = q_value[
                :,  # all critics
                -1,  # last action
                :,  # complete dist
            ]

            assert q_value.shape == (config.critic.update_ensemble_size, n_atoms)
            q_expected = jax.vmap(config.critic.dist.expected_value)(q_value).flatten()  # type: ignore
            q_expected: Array
            assert q_expected.shape == (config.critic.update_ensemble_size,)

            if config.critic.collate_fn == "mean":
                q_value = q_value.mean(axis=0)
            elif config.critic.collate_fn == "min":
                min_q_index = jnp.argmin(q_expected)
                assert min_q_index.shape == tuple()  # scalar
                q_value = q_value[min_q_index]
            elif config.critic.collate_fn == "max":
                max_q_index = jnp.argmax(q_expected)
                assert max_q_index.shape == tuple()  # scalar
                q_value = q_value[max_q_index]
            else:
                assert_never(config.critic.collate_fn)
            assert q_value.shape == (n_atoms,)

            # project(Q) return (scale * Q) + add
            # the critic target is reward + gamma * (Q + alpha * H)
            # we can rearrange this into (gamma * Q) + (reward + gamma * alpha * H)
            # Note that in N-step returns, reward is the N-step reward and gamma is actually
            # gamma**N, which we can access as next_action_scale in this function. next_action_scale
            # also takes care of masking away the next-state V on termination.
            return project(
                weights=q_value,
                add=reward + next_state_v_scale * -ent_coef_value * next_action_logprob,
                scale=next_state_v_scale,
                dist=config.critic.dist,
            )

        else:  # non-distributional critic
            assert q_value.shape == (
                config.critic.update_ensemble_size,
                actor_chunksize + 1,
            )

            if config.critic.collate_fn == "min":
                q_value = q_value.min(axis=0)
            elif config.critic.collate_fn == "mean":
                q_value = q_value.mean(axis=0)
            elif config.critic.collate_fn == "max":
                q_value = q_value.max(axis=0)
            else:
                assert_never(config.critic.collate_fn)

            assert q_value.shape == (1 + actor_chunksize,)
            # we estimate the next state value using the return of the whole chunk
            q_value = q_value[-1]

            next_state_v = q_value - ent_coef_value * next_action_logprob

            return reward + next_state_v_scale * next_state_v

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    @partial(
        jax.jit,
        static_argnames=["sampler", "config", "critic_chunksize"],
    )
    def update_diffusion_actor(
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        chunky_observations: Float[Array, "batch chunkyobs"],
        critic_chunksize: int,
        key,
        sampler,
        config: Config,
    ):
        batch_size = chunky_observations.shape[0]
        (
            key,
            dropout_key,
        ) = jax.random.split(key, 2)

        observations = jnp.split(chunky_observations, critic_chunksize, axis=1)
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(actor_state_in, actor_params):
            actions, run_costs, sto_costs, terminal_costs, latents, v_t = (
                DiffPol.sample_action(
                    actor_state_in, actor_params, observations[0], noise_key, sampler
                )
            )
            _ = v_t
            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats,
                },
                observations[0],
                actions.reshape(batch_size, config.alg.actor_chunksize, -1),
                rngs={"dropout": dropout_key},
                train=False,
            )
            q_reduce_fn = config.actor.update_collate_fn.jax_fn()
            qf_pi = q_reduce_fn(qf_pi, axis=0)
            if config.critic.distributional:
                assert isinstance(qf_pi, Float[Array, "batch chunk n_atoms"])  # type: ignore
                qf_pi = jax.vmap(jax.vmap(config.critic.dist.expected_value))(qf_pi)
                assert isinstance(qf_pi, Array)  # type: ignore
            else:
                qf_pi = qf_pi[:, 1:]

            assert isinstance(qf_pi, Float[Array, "batch chunk"])  # type: ignore
            if config.actor.only_fit_last_nstep:
                min_qf_pi = qf_pi[:, -1].reshape(batch_size, 1)
            else:
                min_qf_pi = qf_pi.mean(axis=1).reshape(batch_size, 1)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            actor_loss = (
                -min_qf_pi
                + ent_coef_value
                * (run_costs.squeeze() + sto_costs.squeeze() + terminal_costs.squeeze())
            ).mean()

            max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
            min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
            mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

            latent_acts = {
                "max_la": max_actions,
                "min_la": min_actions,
                "mean_la": mean_actions,
            }

            return actor_loss, (
                run_costs.mean(),
                sto_costs.mean(),
                terminal_costs.mean(),
                latent_acts,
            )

        outs = jax.value_and_grad(actor_loss, has_aux=True, argnums=1)(
            actor_state, actor_state.params
        )
        (
            act_loss_value,
            (run_costs_mean, sto_costs, terminal_costs, latent_acts),
        ), grads = outs
        _ = latent_acts
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_grad_norm = optax.global_norm(grads)
        metrics = {
            "entropy": 0.0,
            "run_costs": run_costs_mean,
            "sto_costs": sto_costs,
            "terminal_costs": terminal_costs,
        }
        return actor_state, qf_state, act_loss_value, key, actor_grad_norm, metrics

    @staticmethod
    @jax.jit
    def soft_update(tau: Floaty, qf_state: RLTrainState):
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(
                qf_state.params, qf_state.target_params, tau
            )
        )
        qf_state = qf_state.replace(
            target_batch_stats=optax.incremental_update(
                qf_state.batch_stats, qf_state.target_batch_stats, tau
            )
        )
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(
        target_entropy: Floaty, ent_coef_state: TrainState, entropy: Floaty
    ):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy)
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(
            ent_coef_state.params
        )
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "crossq_style",
            "use_bnstats_from_live_net",
            "gradient_steps",
            "config",
            "critic_chunksize",
            "sampler",
            "skip_rl",
        ],
    )
    def _train_with_chunky_critic(
        crossq_style: bool,
        use_bnstats_from_live_net: bool,
        gamma: Floaty,
        tau: Floaty,
        target_entropy: Floaty,
        gradient_steps: int,
        data_chunky: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        qf_state: RLTrainState,
        actor_state: ActorTrainState,
        ent_coef_state: TrainState,
        key,
        config: Config,
        critic_chunksize: int,
        sampler: Callable | None,
        skip_rl: bool,
    ):
        actor_loss_value = jnp.array(0)
        log_metrics_critic = dict()
        entropy = 0
        bc_loss = 0
        nonfinite_action_logprobs = 0
        cov_bias = jnp.array(0.0)
        actor_grad_norm = jnp.nan
        proj_loss = jnp.nan
        off_diag_cov = jnp.nan

        if config.alg.crossq_style:
            update = SAC.update_chunky_crossq_critic
        else:
            update = SAC.update_chunky_critic

        for i in range(gradient_steps):

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = update(
                crossq_style=crossq_style,
                use_bnstats_from_live_net=use_bnstats_from_live_net,
                gamma=gamma,
                actor_state=actor_state,
                qf_state=qf_state,
                ent_coef_state=ent_coef_state,
                chunky_observations=jnp.array(data_chunky.observations),
                chunky_actions=jnp.array(data_chunky.actions),
                chunky_rewards=jnp.array(data_chunky.rewards),
                chunky_dones=jnp.array(data_chunky.dones),
                chunky_truncated=jnp.array(data_chunky.truncated),
                chunky_next_observations=jnp.array(data_chunky.next_observations),
                key=key,
                sampler=sampler,
                config=config,
            )
            qf_state = SAC.soft_update(tau, qf_state)

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            # hack to be able to jit (n_updates % policy_delay == 0)
            if i in policy_delay_indices:
                if config.actor.kind == ActorKind.DIFFUSION:
                    assert sampler is not None
                    (
                        actor_state,
                        qf_state,
                        actor_loss_value,
                        key,
                        actor_grad_norm,
                        _,
                    ) = SAC.update_diffusion_actor(
                        actor_state,
                        qf_state,
                        ent_coef_state,
                        jnp.array(slice(data_chunky.observations)),
                        critic_chunksize,
                        key,
                        sampler=sampler,
                        config=config,
                    )
                else:
                    (
                        actor_state,
                        qf_state,
                        actor_loss_value,
                        key,
                        entropy,
                        bc_loss,
                        proj_loss,
                        off_diag_cov,
                        nonfinite_action_logprobs,
                        cov_bias,
                        actor_grad_norm,
                    ) = SAC.update_actor(
                        actor_state,
                        qf_state,
                        ent_coef_state,
                        jnp.array(slice(data_chunky.observations)),
                        jnp.array(slice(data_chunky.actions)),
                        key,
                        critic_chunksize=critic_chunksize,
                        chunky_critic=True,
                        config=config,
                        skip_rl=skip_rl,
                    )
                ent_coef_state, _ = SAC.update_temperature(
                    target_entropy, ent_coef_state, entropy
                )

        log_metrics = {
            "actor_proj_loss": proj_loss,
            "actor_off_diag_cov": off_diag_cov,
            "actor_grad_norm": actor_grad_norm,
            "actor_loss": actor_loss_value,
            "action_entropy": entropy,
            "bc_loss": bc_loss,
            "actor_nonfinite_action_logprobs": nonfinite_action_logprobs,
            "cov_bias_mean": cov_bias.mean(),
            **log_metrics_critic,
        }
        return qf_state, actor_state, ent_coef_state, key, log_metrics

    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)

    def current_entropy_coeff(self):
        return self.ent_coef_state.apply_fn({"params": self.ent_coef_state.params})

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.

        -- adapted from stable_baselines3/common_off_policy_algorithm.py

        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            # Changed here to use the chunked action space of the policy instead of the unchunked one of the SAC class
            unscaled_action = np.array(
                [self.policy.action_space.sample() for _ in range(n_envs)]
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _save_model(self):
        save_model_state(
            self.policy.actor_state,
            self.model_save_path,
            "actor_state",
            self.num_timesteps,
        )
        save_model_state(
            self.policy.qf_state,
            self.model_save_path,
            "critic_state",
            self.num_timesteps,
        )

    def load_model(self, path, n_steps_actor, n_steps_critic):
        actor_state = load_state(
            path, "actor_state", n_steps_actor, train_state=self.policy.actor_state
        )  # type: ignore
        actor_state: ActorTrainState
        self.policy.actor_state = actor_state

        critic_state = load_state(
            path, "critic_state", n_steps_critic, train_state=self.policy.qf_state
        )  # type: ignore
        critic_state: RLTrainState
        self.policy.qf_state = critic_state


# from build.py
# Save and load model
def save_model_state(train_state, path, name, n_steps):
    # Serialize the model parameters
    serialized_state = flax.serialization.to_bytes(train_state)  # pyright: ignore
    os.makedirs(path, exist_ok=True)
    extended_path = os.path.join(path, f"{name}_{n_steps}.msgpack")
    # Save the serialized parameters to a file
    with open(extended_path, "wb") as f:
        f.write(serialized_state)


def load_state(path, name, n_steps, train_state=None):
    extended_path = os.path.join(path, f"{name}_{n_steps}.msgpack")
    # Load the serialized parameters from a file
    with open(extended_path, "rb") as f:
        train_state_loaded = f.read()
    return flax.serialization.from_bytes(  # pyright: ignore
        train_state, train_state_loaded
    )
