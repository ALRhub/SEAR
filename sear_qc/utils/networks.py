from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims, activate_final=True, layer_norm=self.layer_norm
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class MultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking."""

    num_heads: int
    n_embed: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training):
        from flax.linen.attention import make_causal_mask

        mask = make_causal_mask(x[..., 0])
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, mask=mask)
        x = nn.Dense(self.n_embed)(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward layers."""

    n_embed: int
    n_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training: bool):
        sa = MultiHeadAttention(
            num_heads=self.n_heads,
            n_embed=self.n_embed,
            dropout_rate=self.dropout_rate,
        )
        x = x + sa(nn.LayerNorm()(x), training=training)

        ff = nn.Sequential(
            [
                nn.Dense(4 * self.n_embed),
                nn.PReLU(),
                nn.Dense(self.n_embed),
                nn.Dropout(self.dropout_rate, deterministic=not training),
            ]
        )
        x = x + ff(nn.LayerNorm()(x))
        return x


class _TransformerCriticBackbone(nn.Module):
    """Inner transformer backbone, designed to be ensemblized."""

    output_dim: int
    horizon_length: int
    n_embed: int
    n_heads: int
    n_layer: int
    dropout_rate: float

    @nn.compact
    def __call__(self, seq_embed, training=False):
        # Positional embedding
        seq_len = seq_embed.shape[1]
        block_size = self.horizon_length + 2
        pos = jnp.arange(0, seq_len)
        pos_embed = nn.Embed(block_size, self.n_embed, name="PositionEmbedding")(pos)

        x = seq_embed + pos_embed

        for _ in range(self.n_layer):
            x = TransformerBlock(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
            )(x, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class TransformerValue(nn.Module):
    """Transformer-based critic where each action in the chunk is a token.

    Ensembling is handled internally (like Value), so this module should NOT
    be wrapped with ensemblize.

    Attributes:
        output_dim: Number of output values per token (typically 1).
        horizon_length: Number of actions in the chunk.
        action_dim: Dimension of each individual action.
        n_embed: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout_rate: Dropout rate.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module.
    """

    output_dim: int
    horizon_length: int
    action_dim: int
    n_embed: int = 512
    n_heads: int = 16
    n_layer: int = 2
    dropout_rate: float = 0.0
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        backbone_cls = _TransformerCriticBackbone
        if self.num_ensembles > 1:
            backbone_cls = ensemblize(backbone_cls, self.num_ensembles)
        self.backbone = backbone_cls(
            output_dim=self.output_dim,
            horizon_length=self.horizon_length,
            n_embed=self.n_embed,
            n_heads=self.n_heads,
            n_layer=self.n_layer,
            dropout_rate=self.dropout_rate,
        )
        self.state_embed = nn.Dense(self.n_embed, use_bias=False, name="StateEmbedding")
        self.action_embed = nn.Dense(
            self.n_embed, use_bias=False, name="ActionEmbedding"
        )

    def __call__(self, observations, actions=None):
        # The rest of the code is written in a PyTorch-Style to support a variable number of leading axis.
        # The TransformerValue is not. This method is explicitly handling leading dims, so the main implementation
        # in call_without_leading_dims can work with consistent shapes.
        leading_dims = observations.shape[:-1]
        if actions is not None:
            assert actions.shape[:-1] == leading_dims

        total_leading_dims = 1
        for d in leading_dims:
            total_leading_dims *= d

        batched_obs = observations.reshape(total_leading_dims, -1)
        if actions is not None:
            batched_actions = actions.reshape(total_leading_dims, -1)
        else:
            batched_actions = None

        batched_q = self.call_without_leading_dims(batched_obs, batched_actions)
        n_critics = batched_q.shape[0]
        end_shape = batched_q.shape[2:]  # skip n_critics and batch
        result_shape = (n_critics,) + leading_dims + end_shape

        return batched_q.reshape(*result_shape)

    def call_without_leading_dims(self, observations, actions=None):
        if self.encoder is not None:
            observations = self.encoder(observations)

        batch_size = observations.shape[0]

        # State embedding
        state_embed = self.state_embed(observations)

        if actions is not None:
            # Reshape flat actions (batch, chunk*action_dim) -> (batch, chunk, action_dim)
            actions_reshaped = jnp.reshape(
                actions, (batch_size, self.horizon_length, self.action_dim)
            )

            # Action embedding
            action_embed = self.action_embed(actions_reshaped)

            state_embed = jnp.expand_dims(state_embed, 1)
            seq_embed = jnp.concatenate([state_embed, action_embed], axis=1)
        else:
            seq_embed = jnp.expand_dims(state_embed, 1)

        x = self.backbone(seq_embed)

        # Drop state token, return (batch, chunk) for output_dim=1
        x = x[..., 1:, :]
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    output_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class(
            (*self.hidden_dims, self.output_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
