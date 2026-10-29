from typing import Any, List, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from sac.config import Config, get_module
from sac.transformer_critic import Block
from sac.actor import Actor


tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class ActorTransformer(nn.Module):
    """Transformer backbone for the autoregressive actor."""

    n_embed: int
    n_heads: int
    n_layer: int
    dropout_rate: float
    max_seq_len: int
    config: Config

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        obs: Float[Array, "batch obs_dim"],
        action_history: Float[Array, "batch seq_len action_dim"],
        training: bool,
    ) -> Float[Array, "batch output_seq_len n_embed"]:
        """
        Args:
            obs: [batch, obs_dim] - observation
            action_history: [batch, seq_len, action_dim] - history of actions (can be empty)
            training: whether in training mode

        Returns:
            [batch, 1+seq_len, n_embed] - embeddings for obs token + action tokens
        """
        seq_len = action_history.shape[1]

        # Observation embedding
        if getattr(self.config.actor, "weight_norm", False):
            obs_embed = nn.WeightNorm(
                nn.Dense(self.n_embed, use_bias=False, name="ObsEmbedding")
            )(obs)
        else:
            obs_embed = nn.Dense(self.n_embed, use_bias=False, name="ObsEmbedding")(obs)
        obs_embed = jnp.expand_dims(obs_embed, 1)  # [batch, 1, n_embed]

        if seq_len > 0:
            # Action embeddings
            if getattr(self.config.actor, "weight_norm", False):
                action_embed = nn.WeightNorm(
                    nn.Dense(self.n_embed, use_bias=False, name="ActionEmbedding")
                )(action_history)
            else:
                action_embed = nn.Dense(
                    self.n_embed, use_bias=False, name="ActionEmbedding"
                )(action_history)
            # Concatenate obs and action history
            seq_embed = jnp.concatenate(
                [obs_embed, action_embed], axis=1
            )  # [batch, 1+seq_len, n_embed]
        else:
            seq_embed = obs_embed

        # Positional embeddings
        pos_indices = jnp.arange(seq_embed.shape[1])
        pos_embed = nn.Embed(self.max_seq_len, self.n_embed, name="PositionEmbedding")(
            pos_indices
        )

        # Add positional embeddings
        x = seq_embed + pos_embed

        # Transformer blocks
        for _ in range(self.n_layer):
            x = Block(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                norm=self.config.actor.norm,
                weight_norm=getattr(self.config.actor, "weight_norm", False),
            )(x, training=training)

        # Layer norm
        norm = get_module(self.config.actor.norm, train=training)
        x = norm(x)

        # Return all tokens (caller extracts what they need)
        return x  # [batch, 1+seq_len, n_embed]


class ActorAutoregressive(nn.Module):
    action_dim: int
    single_action_dim: int
    config: Config

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch obs"],
        train: bool,
        deterministic: bool,
        key: Array,
        teacher_forcing_actions: Float[Array, "batch chunkyaction"] | None = None,
        single_action: bool = False,
    ) -> tuple[
        Float[Array, "batch chunkyaction"],  # chunk action
        Float[Array, "batch chunk"],  # logprobs
        Float[Array, "action action"],  # debug_cov
        Float[Array, "batch 1"],  # cov_bias
    ]:
        self.variable("batch_stats", "dummy", lambda _: jnp.zeros(1), 1)

        batch_size = x.shape[0]
        chunk_size = self.config.alg.actor_chunksize
        assert x.ndim == 2, f"Expected 2D observation, got shape {x.shape}"
        assert chunk_size > 0, f"actor_chunksize must be positive, got {chunk_size}"

        # When single_action=True, only generate one action (for critic updates with single_action_v)
        effective_chunk_size = 1 if single_action else chunk_size

        # Initialize transformer
        transformer = ActorTransformer(
            n_embed=self.config.actor.n_embed,
            n_heads=self.config.actor.n_heads,
            n_layer=self.config.actor.n_layer,
            dropout_rate=self.config.actor.dropout_rate,
            max_seq_len=chunk_size,
            config=self.config,
        )

        # Generate actions step by step with proper conditioning
        actions, logprobs, step_cov_bias_mags = self._generate_autoregressive(
            x,
            train,
            transformer,
            deterministic,
            key,
            batch_size,
            effective_chunk_size,
            teacher_forcing_actions,
        )

        return self._format_for_sampling(
            actions, logprobs, step_cov_bias_mags, batch_size, chunk_size
        )

    @jaxtyped(typechecker=typechecker)
    def _generate_autoregressive(
        self,
        x: Float[Array, "batch obs"],
        train: bool,
        transformer: "ActorTransformer",
        deterministic: bool,
        key: Array,
        batch_size: int,
        chunk_size: int,
        teacher_forcing_actions: Float[Array, "batch chunkyaction"] | None = None,
    ) -> Tuple[
        List[Float[Array, "batch single_act_dim"]],  # actions  # noqa: F821
        List[Float[Array, "batch"]],  # logprobs  # noqa: F821
        List[Float[Array, "batch"]],  # step_cov_bias_mags  # noqa: F821
    ]:
        """Generate actions autoregressively."""
        actions = []
        logprobs = []
        step_cov_bias_mags = []
        sampled_actions = []  # For autoregressive conditioning

        # Teacher forcing mode check
        use_teacher_forcing = teacher_forcing_actions is not None

        # Split keys for sampling
        keys = jax.random.split(key, chunk_size)

        for step, step_key in enumerate(keys):
            # Create action history
            if step == 0:
                # First step: no conditioning on previous actions
                action_history = jnp.zeros((batch_size, 0, self.single_action_dim))
            else:
                if use_teacher_forcing:
                    # Use ground truth actions for conditioning (no stop_gradient needed)
                    action_history = teacher_forcing_actions[
                        :, : step * self.single_action_dim
                    ].reshape(batch_size, step, self.single_action_dim)
                else:
                    # Subsequent steps in sampling mode: condition on actual sampled actions
                    # We stop the gradient here to prevent deep gradient chains through the transformer
                    action_history = jax.lax.stop_gradient(
                        jnp.stack(sampled_actions, axis=1)
                    )

            # Get transformer representation
            h_all = transformer(x, action_history, train)  # [batch, 1+step, n_embed]
            h = h_all[:, -1, :]  # Extract last token [batch, n_embed]
            assert h.shape == (
                batch_size,
                self.config.actor.n_embed,
            ), f"Expected transformer output shape {(batch_size, self.config.actor.n_embed)}, got {h.shape}"

            # Generate action distribution for this step
            mean, cov, cov_bias_mag = Actor._mean_and_cov(
                self, hidden_state=h, n_actions=1, config=self.config
            )
            assert mean.shape == (
                batch_size,
                self.single_action_dim,
            ), f"Expected mean shape {(batch_size, self.single_action_dim)}, got {mean.shape}"
            assert cov.shape == (
                batch_size,
                self.single_action_dim,
                self.single_action_dim,
            ), f"Expected cov shape {(batch_size, self.single_action_dim, self.single_action_dim)}, got {cov.shape}"

            step_cov_bias_mags.append(cov_bias_mag.flatten())

            # Sample action
            action, logprob = Actor._sample_action_and_logprobs(
                self, mean=mean, cov=cov, deterministic=deterministic, key=step_key
            )
            actions.append(action)
            logprobs.append(logprob.flatten())  # flatten [batch, 1] -> [batch]
            # Store sampled action for next step conditioning
            sampled_actions.append(action)

        return actions, logprobs, step_cov_bias_mags

    @jaxtyped(typechecker=typechecker)
    def _format_for_sampling(
        self,
        actions: List[Float[Array, "batch single_act_dim"]],  # noqa: F821
        logprobs: List[Float[Array, "batch"]],  # noqa: F821
        step_cov_bias_mags: List[Float[Array, "batch"]],  # noqa: F821
        batch_size: int,
        chunk_size: int,
    ) -> Tuple[
        Float[Array, "batch chunky_act"],  # chunk action
        Float[Array, "batch chunk"],  # logprobs
        Float[Array, "single_act_dim single_act_dim"],  # debug_cov
        Float[Array, "batch 1"],  # cov_bias
    ]:
        """Format sampled actions and logprobs for output."""
        # Combine all actions and logprobs
        sample = jnp.concatenate(
            actions, axis=1
        )  # [batch, chunkyaction or single_action_dim]
        logprobs_concat = jnp.stack(logprobs, axis=1)  # [batch, effective_chunk_size]

        effective_chunk_size = len(actions)

        # If we only generated one action but need to return chunk_size, pad with zeros
        if effective_chunk_size < chunk_size:
            # Pad actions with zeros to match expected output shape
            pad_width = chunk_size - effective_chunk_size
            sample = jnp.pad(
                sample,
                ((0, 0), (0, pad_width * self.single_action_dim)),
                mode="constant",
            )
            # Pad logprobs with zeros
            logprobs_concat = jnp.pad(
                logprobs_concat, ((0, 0), (0, pad_width)), mode="constant"
            )

        assert sample.shape == (
            batch_size,
            self.action_dim,
        ), f"Expected sample shape {(batch_size, self.action_dim)}, got {sample.shape}"
        assert logprobs_concat.shape == (
            batch_size,
            chunk_size,
        ), f"Expected logprobs shape {(batch_size, chunk_size)}, got {logprobs_concat.shape}"

        # debug cov makes little sense in this setting
        debug_cov = 42 * jnp.ones((self.single_action_dim, self.single_action_dim))

        cov_bias_mag_avg = jnp.mean(jnp.array(step_cov_bias_mags).T, axis=1).reshape(
            batch_size, 1
        )

        return sample, logprobs_concat, debug_cov, cov_bias_mag_avg
