from typing import Any, Optional, Sequence, Tuple, Type, Union, assert_never

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
import math
from gymnasium import spaces
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from common.policies import BaseJaxPolicy
from common.type_aliases import ActorTrainState, RLTrainState
from sac.config import ActorKind, Config, CriticKind, NormConfig, get_module
from sac.transformer_critic import SeqQFunc
from sac.utils import activation_fn
from sac.actor import Actor, SimbaV2Head
from sac.actor_autoregressive import ActorAutoregressive
from scale_rl.agents.simbaV2.simbaV2_layer import (
    HyperEmbedder,
    HyperLERPBlock,
)

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class Critic(nn.Module):
    out_dim: int
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    norm: NormConfig
    config: Config
    dropout_rate: Optional[float] = None

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch obs"],
        action: Float[Array, "batch chunk action"],
        train: bool,
    ) -> Union[Float[Array, "batch v_and_chunk"], Float[Array, "batch chunk n_atoms"]]:
        self.variable("batch_stats", "dummy", lambda _: jnp.zeros(1), 1)
        batch_size, critic_chunksize, action_dim = action.shape
        assert critic_chunksize == self.config.alg.critic_chunksize
        assert self.out_dim == critic_chunksize + 1
        assert not (self.config.critic.weight_norm)
        assert not self.dropout_rate
        n_atoms = self.config.critic.dist.n_atoms

        if self.config.critic.distributional:
            output_dimension = n_atoms * critic_chunksize
        else:
            assert self.out_dim == critic_chunksize + 1
            output_dimension = self.out_dim

        x = jnp.concatenate(
            [x, action.reshape((batch_size, critic_chunksize * action_dim))], 1
        )

        hidden_dim = self.net_arch[0]
        num_blocks = len(self.net_arch)

        if self.config.critic.simbav2:
            # Use SimbaV2 architecture
            # Default values based on scale_rl implementation
            scaler_init = math.sqrt(2) / math.sqrt(hidden_dim)
            scaler_scale = math.sqrt(2) / math.sqrt(hidden_dim)
            alpha_init = 1 / (num_blocks + 1) if num_blocks > 0 else 1.0
            alpha_scale = 1 / math.sqrt(hidden_dim)
            c_shift = 3.0

            # Embedder
            x = HyperEmbedder(
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                c_shift=c_shift,
            )(x)

            # Encoder blocks
            for _ in range(num_blocks):
                x = HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )(x)
            x = SimbaV2Head(output_dimension, hidden_dim=hidden_dim)(x)
        else:
            for n_units in self.net_arch:
                x = nn.Dense(n_units)(x)
                norm = get_module(self.norm, train)
                x = norm(x)
                x = self.activation_fn()(x)
            x = nn.Dense(output_dimension)(x)

        if self.config.critic.distributional:
            x = x.reshape((batch_size, critic_chunksize, n_atoms))
            x = jax.nn.softmax(x, axis=-1)

        if self.is_initializing():
            total_params = sum(
                p.size for p in jax.tree_util.tree_leaves(self.variables["params"])
            )
            print(f"Critic Parameters: {total_params:,}")

        return x


class TransformerCritic(nn.Module):
    out_dim: int
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    critic_chunksize: int
    actor_chunksize: int
    norm: NormConfig
    config: Config
    dropout_rate: Optional[float] = None
    n_critics: int = 2

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        obs: Float[Array, "batch obs"],
        action: Float[Array, "batch chunk action"],
        train: bool = True,
    ):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            SeqQFunc,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,  # type: ignore
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            n_embed=self.config.critic.n_embed,
            n_heads=self.config.critic.n_heads,
            n_layer=self.config.critic.n_layer,
            dropout_rate=0,
            block_size=max(self.actor_chunksize, self.critic_chunksize) + 2,
            relative_pos=True,
            norm=self.norm,
            config=self.config,
        )(obs, action, train)
        return q_values


class VectorCritic(nn.Module):
    out_dim: int
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    norm: NormConfig
    config: Config
    dropout_rate: Optional[float] = None
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,  # type: ignore
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            out_dim=self.out_dim,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            config=self.config,
        )(obs, action, train)
        return q_values


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        config: Config,
        observation_space: spaces.Space,
        action_space: spaces.Box,
    ):
        super().__init__(
            observation_space,
            action_space,
            None,
            None,
            optimizer_class=optax.adam,
            optimizer_kwargs=None,
            squash_output=True,
        )
        self.config = config
        self.critic_chunksize = config.alg.critic_chunksize
        self.actor_chunksize = config.alg.actor_chunksize
        self.chunky_critic = config.alg.chunky_critic
        self.dropout_rate = 0
        self.critic_norm = config.critic.norm
        self.actor_norm = config.actor.norm

        assert config.actor.activation == config.critic.activation
        self.activation_fn = activation_fn[config.actor.activation]
        self.activation_fn: type[nn.Module]

        self.net_arch_pi = config.actor.hs
        self.net_arch_qf = config.critic.hs
        self.n_critics = config.critic.n_critics
        self.use_sde = False

        self.key = self.noise_key = jax.random.key(0)

    def _get_critic_optimizer(self):
        """Get the optimizer for the critic based on configuration."""
        if self.config.optimizer.critic_use_adamw:
            return optax.adamw(
                learning_rate=self.config.optimizer.critic_lr,
                b1=self.config.optimizer.b1,
            )
        else:
            return optax.adam(
                learning_rate=self.config.optimizer.critic_lr,
                b1=self.config.optimizer.b1,
            )

    def build(self, key, skip_actor: bool = False):
        critic_chunksize = self.config.alg.critic_chunksize
        actor_chunksize = self.config.alg.actor_chunksize
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [
                    spaces.flatten(
                        self.observation_space, self.observation_space.sample()
                    )
                ]
            )
        else:
            obs = jnp.array([self.observation_space.sample()])

        if self.chunky_critic:
            action = jnp.array([self.action_space.sample()]).reshape(
                (1, actor_chunksize, -1)
            )
        else:
            action = jnp.split(
                jnp.array([self.action_space.sample()]), actor_chunksize, axis=1
            )[0]

        chunky_action_dim = int(np.prod(self.action_space.shape))
        single_action_dim = chunky_action_dim // actor_chunksize

        if not skip_actor:
            match self.config.actor.kind:
                case ActorKind.DIFFUSION:
                    raise ValueError(
                        "SACPolicy does not support diffusion, use sac.DiffusionPolicy instead"
                    )
                case ActorKind.AR:
                    self.actor = ActorAutoregressive(
                        action_dim=chunky_action_dim,
                        single_action_dim=single_action_dim,
                        config=self.config,
                    )
                case ActorKind.MLP:
                    self.actor = Actor(
                        action_dim=chunky_action_dim,
                        single_action_dim=single_action_dim,
                        config=self.config,
                    )
                case x:
                    assert_never(x)
            # Hack to make gSDE work without modifying internal SB3 code
            self.actor.reset_noise = self.reset_noise

            actor_init_variables = self.actor.init(
                {"params": actor_key, "batch_stats": bn_key},
                obs,
                train=False,
                deterministic=False,
                key=jax.random.key(42),
            )
            self.actor_state = ActorTrainState.create(
                apply_fn=self.actor.apply,
                params=actor_init_variables["params"],
                old_params=actor_init_variables["params"],
                batch_stats=actor_init_variables["batch_stats"],
                tx=self.optimizer_class(
                    learning_rate=self.config.optimizer.actor_lr,  # type: ignore[call-arg]
                    **self.optimizer_kwargs,
                ),
            )
            self.actor.apply = jax.jit(  # type: ignore[method-assign]
                self.actor.apply,
                static_argnames=(
                    "use_batch_norm",
                    "batch_norm_momentum",
                    "bn_mode",
                    "return_dist",
                ),
            )

        match self.config.critic.kind:
            case CriticKind.TRANSFORMER:
                self.qf = TransformerCritic(
                    out_dim=(critic_chunksize + 1) if self.chunky_critic else 1,
                    dropout_rate=self.dropout_rate,
                    norm=self.critic_norm,
                    net_arch=self.net_arch_qf,
                    activation_fn=self.activation_fn,
                    n_critics=self.n_critics,
                    critic_chunksize=critic_chunksize,
                    actor_chunksize=self.actor_chunksize,
                    config=self.config,
                )

            case CriticKind.MLP:
                self.qf = VectorCritic(
                    out_dim=(critic_chunksize + 1) if self.chunky_critic else 1,
                    dropout_rate=self.dropout_rate,
                    norm=self.critic_norm,
                    net_arch=self.net_arch_qf,
                    activation_fn=self.activation_fn,
                    n_critics=self.n_critics,
                    config=self.config,
                )

            case x:
                assert_never(x)

        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=self._get_critic_optimizer(),
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=(
                "dropout_rate",
                "use_layer_norm",
                "use_batch_norm",
                "batch_norm_momentum",
                "bn_mode",
            ),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        _ = batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    # type: ignore[override]
    def _predict(  # type: ignore[reportIncompatibleMethodOverride]
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(
            self.actor_state, observation, self.noise_key
        )

    def _predict_deterministic(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        assert not kwargs
        return BaseJaxPolicy.select_action(self.actor_state, observation)

    def predict_action_with_logprobs(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation, True)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()

        return BaseJaxPolicy.sample_action(
            self.actor_state, observation, self.noise_key, True
        )

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o,
                a,
                rngs={"dropout": dropout_key},
                train=False,
            )

        return jax.jit(Q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )
