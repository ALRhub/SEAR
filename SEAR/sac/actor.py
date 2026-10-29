from typing import Any, Sequence, Tuple, Union, Protocol

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability
import math
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from common.distributions import TanhTransformedDistribution
from sac.config import Config, get_module
from scale_rl.agents.simbaV2.simbaV2_layer import (
    HyperEmbedder,
    HyperLERPBlock,
    HyperDense,
    Scaler,
)

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class SimbaV2Head(nn.Module):
    # the logic of HyperNormalTanhPolicy's mean/cov generation
    # also found in HyperCategoricalValues.
    # extracted to make it work with our action generation
    out_dim: int
    hidden_dim: int

    def setup(self):
        scaler_init = 1
        scaler_scale = 1

        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(self.out_dim)
        self.bias = self.param("bias", nn.initializers.zeros, (self.out_dim,))

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> tfd.Distribution:
        x = self.w1(x)
        x = self.scaler(x)
        x = self.w2(x) + self.bias
        return x


class IActor(Protocol):
    config: Config
    action_dim: int
    single_action_dim: int


class Actor(nn.Module):
    action_dim: int
    single_action_dim: int
    config: Config

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    # type: ignore[name-defined]
    def __call__(
        self,
        x: Float[Array, "batch obs"],
        train: bool,
        deterministic: bool,
        key: Array,
        return_dist: bool = False,
        single_action: bool = False,
    ) -> Union[
        # return_dist == False
        tuple[
            Float[Array, "batch chunkyaction"],  # chunk action
            Float[Array, "batch chunk"],  # logprobs
            Float[Array, "action action"],  # debug_cov
            Float[Array, "batch 1"],  # cov_bias
        ],
        # return_dist == True
        tuple[
            Float[Array, "batch chunkyaction"],  # mean
            Float[Array, "batch chunkyaction chunkyaction"],  # cov
            Float[Array, "batch 1"],  # cov_bias
        ],
    ]:
        mean, cov, cov_bias_magnitude = self._get_action_dist(x, train)

        if return_dist:
            return mean, cov, cov_bias_magnitude
        debug_cov = cov.mean(axis=0)

        actions, logprobs = self._sample_action_and_logprobs(
            actor=self, mean=mean, cov=cov, deterministic=deterministic, key=key
        )

        return actions, logprobs, debug_cov, cov_bias_magnitude

    def _get_action_dist(self, x, train):
        batch_size = x.shape[0]
        self.variable("batch_stats", "dummy", lambda _: jnp.zeros(1), 1)

        if self.config.actor.simbav2:
            # Use SimbaV2 architecture
            hidden_dim = self.config.actor.hs[0] if self.config.actor.hs else 256
            num_blocks = len(self.config.actor.hs)

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

        else:  # normal MLP
            for n_units in self.config.actor.hs:
                x = nn.Dense(n_units)(x)
                x = nn.relu(x)
                norm = get_module(self.config.actor.norm, train)
                x = norm(x)

        mean, cov, cov_bias_magnitude = Actor._mean_and_cov(
            self, x, n_actions=self.config.alg.actor_chunksize, config=self.config
        )

        assert cov.shape == (batch_size, self.action_dim, self.action_dim)
        return mean, cov, cov_bias_magnitude

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def _mean_and_cov(
        actor: IActor,
        hidden_state: Float[Array, " batch hidden"],
        n_actions: int,
        config: Config,
    ) -> Tuple[
        Float[Array, " batch chunkyaction"],  # mean
        Float[Array, " batch chunkyaction chunkyaction"],  # cov
        Float[Array, " batch 1"],  # cov bias magnitude
    ]:
        action_dim = actor.single_action_dim * n_actions
        x = hidden_state
        batch_size = x.shape[0]
        simba_hs = config.actor.hs[0]
        if config.actor.simbav2:
            mean = SimbaV2Head(out_dim=action_dim, hidden_dim=simba_hs)(x)
        else:
            mean = nn.Dense(action_dim)(x)
        assert mean.shape == (batch_size, action_dim)

        # Calculate cov_bias
        if config.actor.simbav2:
            cov_bias_magnitude = SimbaV2Head(out_dim=1, hidden_dim=simba_hs)(x)
        else:
            cov_bias_magnitude = nn.Dense(1)(x)
        cov_bias_magnitude = (
            jax.nn.sigmoid(cov_bias_magnitude + 4) * actor.config.actor.bias
        )

        if actor.config.actor.bias_kind == "diag":

            def scaled_diag(scale: float) -> Array:
                return scale * jnp.eye(action_dim)

            cov_bias = jax.vmap(scaled_diag)(cov_bias_magnitude**2)
        else:
            raise NotImplementedError(
                f"Unknown bias_kind '{actor.config.actor.bias_kind}'"
            )
        # cov_bias is always a full NxN covariance matrix
        assert cov_bias.shape == (batch_size, action_dim, action_dim)

        # Calculate covariance matrix cov

        if not actor.config.actor.full_cov:
            if config.actor.simbav2:
                diag_log_std = SimbaV2Head(out_dim=action_dim, hidden_dim=simba_hs)(x)
            else:
                diag_log_std = nn.Dense(action_dim)(x)
            diag_log_std = jnp.clip(
                diag_log_std,
                actor.config.actor.log_std_min,
                actor.config.actor.log_std_max,
            )
            diag_std = jnp.exp(diag_log_std)
            std = jax.vmap(jnp.diag)(diag_std)
        else:
            n_nonzero = sum(
                range(action_dim + 1)
            )  # how many non-zero elements does the tril have?

            def triangular_matrix(nonzero_elements: jnp.ndarray):
                # This is a simplified version of the cov-generation of trpl.
                result = jnp.zeros((action_dim, action_dim))
                tril_indices = jnp.tril_indices_from(result)
                diag_indices = jnp.diag_indices_from(result)
                # This scaling factor works well empirically to get a somewhat uniform diagonal
                result = result.at[tril_indices].set(nonzero_elements * 0.003)
                # Also ensure the diag is always positive with a biased softplus (softplus(.5) ~ 1)
                result = result.at[diag_indices].set(
                    jax.nn.softplus(result[diag_indices] + 0.5) + 1e-5
                )
                return result

            if config.actor.simbav2:
                tril_values = SimbaV2Head(out_dim=n_nonzero, hidden_dim=simba_hs)(x)
            else:
                tril_values = nn.Dense(n_nonzero)(x)
            tril = jax.vmap(triangular_matrix)(tril_values)
            # std = jnp.clip(
            #     tril,
            #     jnp.log(actor.config.actor.log_std_min),
            #     jnp.log(actor.config.actor.log_std_max),
            # )
            std = tril

        cov = jax.vmap(lambda x: x @ x.T)(std)
        assert cov.shape == cov_bias.shape
        cov += cov_bias
        return mean, cov, cov_bias_magnitude

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def _sample_action_and_logprobs(
        actor: IActor,
        mean: Float[Array, "batch chunkyaction"],
        cov: Float[Array, "batch chunkyaction chunkyaction"],
        deterministic: bool,
        key: jax.Array,
    ) -> Tuple[
        Float[Array, " batch chunkyaction"],  # action
        Float[Array, " batch chunk"],  # logprobs
    ]:
        # Calculate logprobs based on the specified type
        logprob_type = actor.config.actor.logprob_type

        if logprob_type == "joint":
            return Actor._compute_joint_logprobs(actor, mean, cov, deterministic, key)
        elif logprob_type == "marginalized":
            # Marginalized probabilities: log p(a_i | s_0) for each action chunk
            return Actor._compute_marginalized_logprobs(
                actor, mean, cov, deterministic, key
            )
        else:
            raise ValueError(f"Unknown logprob_type: {logprob_type}")

    @staticmethod
    def _compute_joint_logprobs(actor: IActor, mean, cov, deterministic, key):
        """Compute joint log probability for all actions together."""
        n_actions = mean.shape[1] // actor.single_action_dim
        if deterministic:
            dist = TanhTransformedDistribution(
                tfd.MultivariateNormalFullCovariance(mean, cov)
            )
            actions = dist.mode()
            joint_logprob = dist.log_prob(actions)
        else:
            dist = distrax.Transformed(
                distrax.MultivariateNormalFullCovariance(
                    loc=mean, covariance_matrix=cov
                ),
                distrax.Block(distrax.Tanh(), ndims=1),
            )

            actions, joint_logprob = dist.sample_and_log_prob(seed=key)

        # Repeat for each chunk to maintain consistent output shape
        logprobs = (
            jnp.tile(joint_logprob[:, None], (1, n_actions))  # type: ignore
            / actor.config.alg.actor_chunksize
        )
        return actions, logprobs

    @staticmethod
    def _compute_marginalized_logprobs(actor: IActor, mean, cov, deterministic, key):
        """Compute marginalized log probabilities for each action chunk independently."""
        batch = mean.shape[0]
        single_action = actor.single_action_dim
        n_actions = mean.shape[1] // single_action

        actions = jnp.empty_like(mean)
        logprobs = jnp.empty((batch, n_actions))

        for i in range(n_actions):
            mean_ = mean[:, i * single_action : (i + 1) * single_action]
            std_ = cov[
                :,
                i * single_action : (i + 1) * single_action,
                i * single_action : (i + 1) * single_action,
            ]
            dist = TanhTransformedDistribution(
                tfd.MultivariateNormalFullCovariance(mean_, std_)
            )
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample(seed=key)
            actions = actions.at[:, i * single_action : (i + 1) * single_action].set(
                action
            )
            logprobs = logprobs.at[:, i].set(dist.log_prob(action))
        return actions, logprobs
