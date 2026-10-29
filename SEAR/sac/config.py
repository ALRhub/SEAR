from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Protocol, Optional, Union, Any, assert_never
import jax.numpy as jnp
from jaxtyping import Float, Array

from flax.linen.normalization import BatchNorm, LayerNorm, Module

from sac.utils import BatchRenorm, Floaty
import jax


class JaxReductionFunction(Protocol):
    def __call__(
        self,
        a: jax.Array,
        axis: Optional[Union[int, tuple]] = None,
    ) -> jax.Array: ...


@dataclass(frozen=True)
class NormConfig:
    norm: str
    bn_momentum: float
    bn_warmup: int


def get_module(norm_config, train: bool) -> Module:
    norm = norm_config.norm.lower()
    if norm == "none":
        return Dummy()
    elif norm == "bn":
        return BatchNorm(
            use_running_average=not train, momentum=norm_config.bn_momentum
        )
    elif norm == "brn":
        return BatchRenorm(
            use_running_average=not train,
            momentum=norm_config.bn_momentum,
            bn_warmup=norm_config.bn_warmup,
        )
    elif norm == "ln":
        return LayerNorm()
    raise NotImplementedError()


class Dummy(Module):
    def __call__(self, x):
        return x


class ActorKind(Enum):
    MLP = "mlp"
    DIFFUSION = "diffusion"
    AR = "autoregressive"


class CriticKind(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


class NoiseKind(Enum):
    NONE = "none"
    ADDITIVE_GAUSS = "additive_gauss"


class Fn(Enum):
    MIN = "min"
    MEAN = "mean"

    def jax_fn(self) -> JaxReductionFunction:
        match self.value:
            case "min":
                return jnp.min
            case "mean":
                return jnp.mean
            case x:
                assert_never(x)


@dataclass(frozen=True)
class ActorConfig:
    norm: NormConfig
    activation: str
    full_cov: bool
    logprob_type: str
    hs: tuple
    log_std_min: float
    log_std_max: float
    bias: float
    bias_kind: str
    kind: ActorKind
    only_fit_last_nstep: bool
    update_collate_fn: Fn
    simbav2: bool
    trpl: bool
    mean_bound: float
    cov_bound: float
    weight_norm: bool
    n_embed: int
    n_heads: int
    n_layer: int
    dropout_rate: float
    bc_coef: float
    target_quartile: None | float


@dataclass(frozen=True)
class DistConfig:
    n_atoms: int
    v_min: float
    v_max: float

    @property
    def delta_z(self) -> float:
        return (self.v_max - self.v_min) / (self.n_atoms - 1)

    @property
    def atoms(self) -> Float[Array, " n_atoms"]:
        return jnp.linspace(self.v_min, self.v_max, self.n_atoms)

    def expected_value(self, weights: Float[Array, " n_atoms"]) -> Floaty:
        return jnp.sum(weights * self.atoms)

    def quartile(self, weights: Float[Array, " n_atoms"], q: float) -> Floaty:
        # computes the given quartile, interpolating linearly between buckets
        assert 0 <= q <= 1
        cum = jnp.cumsum(weights)
        ri = jnp.searchsorted(cum, q)
        li = jnp.maximum(ri - 1, 0)
        cl = cum[li]
        cr = cum[ri]
        # cl < q <= cr
        d = cr - cl
        alpha = (q - cl) / d
        alpha = jnp.nan_to_num(alpha, nan=0)  # if li == ri
        # 0 < alpha <= 1
        return (1 - alpha) * self.atoms[li] + alpha * self.atoms[ri]


@dataclass(frozen=True)
class CriticConfig:
    norm: NormConfig
    activation: str
    n_critics: int
    hs: tuple
    simbav2: bool
    simba_blocks: int
    norm_qk: bool
    only_fit_last_nstep: bool
    update_ensemble_size: int
    kind: CriticKind
    distributional: bool
    use_cramer_von_mises: bool
    dist: DistConfig
    weight_norm: bool
    n_embed: int
    n_heads: int
    n_layer: int
    last_chunk_bias: float
    last_chunk_bias_decay: float
    collate_fn: Literal["mean", "min", "max"]


@dataclass(frozen=True)
class AlgConfig:
    chunky_critic: bool
    random_chunks: bool
    random_init_chunk: bool
    temporl: bool
    scale_chunks: bool
    crossq_style: bool
    use_next_action_logprob: bool
    marginalize_next_action_logprob: bool
    marginalize_actor_logprob: bool
    next_ep_critic_update: bool
    tau: float
    utd: int
    tot_time_steps: int
    policy_delay: int
    log_freq: int
    critic_chunksize: int
    actor_chunksize: int
    ent_coef: float | Literal["auto"]
    ent_coef_init: float
    target_ent: float | Literal["auto"]
    apply_next_state_ent_bonus: bool
    gamma: float
    batch_size: int
    offline_steps: int
    skip_rl: bool
    train_bc: bool
    buffer_size: int
    single_action_v: bool
    rlpd: bool


@dataclass(frozen=True)
class OptimizerConfig:
    actor_lr: float
    critic_lr: float
    b1: float
    critic_use_adamw: bool


@dataclass(frozen=True)
class WandbConfig:
    entity: str
    project: str
    group: str
    job_type: str
    activate: bool


@dataclass(frozen=True)
class RolloutConfig:
    buffer_len: int
    ratio_tolerance: float
    annealing_eps: float
    bias: float


@dataclass(frozen=True)
class NoiseConfig:
    kind: NoiseKind
    action: float
    obs: float


@dataclass(frozen=True)
class Config:
    exp_name: str
    seed: int
    use_jit: bool
    env_name: str

    alg: AlgConfig
    actor: ActorConfig
    critic: CriticConfig
    optimizer: OptimizerConfig
    wandb: WandbConfig
    rollout: RolloutConfig
    noise: NoiseConfig

    model_save_path: Optional[str]
    n_saved_models: int
