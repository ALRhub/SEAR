from dataclasses import asdict
from typing import Callable, NamedTuple
from flax.core import FrozenDict
from gymnasium.spaces.box import Box
import jax
import numpy as np
import jax.numpy as jnp

from functools import partial

from gymnasium import spaces
import optax
from DIME.diffusion.common.scheduler import get_cosine_schedule
from DIME.diffusion.common.utils import get_sampler_init
from DIME.diffusion.od.od_integrators import get_integrator as get_integrator_od
from DIME.diffusion.od.od_sampling import sample as sample_od
from common.policies import BaseJaxPolicy
from common.type_aliases import ActorTrainState
from sac.config import Config, OptimizerConfig
from sac.policies import SACPolicy


class _Optimizer(OptimizerConfig):
    actor_grad_clip: float = 1
    do_actor_grad_clip: bool = True


class _ScoreModel(NamedTuple):
    use_target_score: bool = False


class _Sampler(NamedTuple):
    name: str = "dis"
    integrator: str = "EM"
    init_std: float = 2.5
    friction: float = 1.0
    underdamped: bool = False
    learn_prior: bool = False
    learn_betas: bool = False
    learn_friction: bool = True
    learn_mass_matrix: bool = False
    score_model: _ScoreModel = _ScoreModel()
    dt_schedule: Callable = get_cosine_schedule(
        total_steps=16, min=0.001, s=0.008, pow=2
    )


class _Actor(NamedTuple):
    diff_steps: int = 16  # also used in _Sampler


class _Alg(NamedTuple):
    batch_size: int
    optimizer: _Optimizer
    actor: _Actor = _Actor()


class DimeCfg(NamedTuple):
    """Adapter for DIME-style methods that expect a 'cfg' parameter"""

    alg: _Alg
    step_size: float
    step_size_betas: float
    sampler: _Sampler = _Sampler()
    per_step_dt: bool = False
    dt: float = 0.1
    use_path_gradient: bool = False
    use_target_score: bool = False
    dt: float = 0.1
    learn_dt: bool = True
    per_step_dt: bool = False
    per_dim_friction: bool = True
    # Related to the learning rate scheduler (not used in DIME)
    use_step_size_scheduler: bool = False

    @classmethod
    def from_config(cls, config: Config) -> "DimeCfg":
        _ = config
        return cls(
            step_size=config.optimizer.actor_lr,
            step_size_betas=config.optimizer.actor_lr,
            alg=_Alg(
                batch_size=config.alg.batch_size,
                optimizer=_Optimizer(**asdict(config.optimizer)),
            ),
        )


class DiffPol(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        config: Config,
        squash_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            features_extractor_kwargs=None,
            optimizer_class=optax.adam,
            squash_output=squash_output,
        )
        self.use_sde = False
        self.config = config

    def build(self, key):
        (
            key,
            qf_key,
            dropout_key,
            bn_key,
        ) = jax.random.split(key, 4)
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

        assert isinstance(self.action_space, Box)
        a_dim = self.action_space.shape[0]
        obs_dim = obs.shape[1]

        sac_pol = SACPolicy(
            config=self.config,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        sac_pol.build(qf_key, skip_actor=True)
        self.qf = sac_pol.qf
        self.qf_state = sac_pol.qf_state

        # Initialize actor
        cfg = DimeCfg.from_config(self.config)
        key, diff_key = jax.random.split(key, 2)
        self.actor_model, actor_state = get_sampler_init(cfg.sampler.name)(
            diff_key, cfg, a_dim, obs_dim
        )
        self.actor_state = ActorTrainState(
            step=actor_state.step,
            apply_fn=actor_state.apply_fn,
            params=actor_state.params,
            old_params=actor_state.params,
            tx=actor_state.tx,
            opt_state=actor_state.opt_state,
            batch_stats=FrozenDict({"foo": jnp.zeros(1)}),
        )
        target_model_state = get_sampler_init(cfg.sampler.name)(
            diff_key, cfg, a_dim, obs_dim
        )
        self.actor_target_model, self.target_actor_state = target_model_state
        self.integrator = get_integrator_od(cfg, self.actor_model)
        self.target_integrator = get_integrator_od(cfg, self.actor_target_model)
        sampler = sample_od
        self.sampler = partial(
            sampler, integrator=self.integrator, diffusion_model=self.actor_model
        )
        self.target_sampler = partial(
            sampler,
            integrator=self.target_integrator,
            diffusion_model=self.actor_target_model,
        )
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "return_logprob"])
    def sample_action(
        actor_state, actor_params, observations, key, sampler, return_logprob=False
    ):
        out = sampler(key, actor_state, actor_params, observations, stop_grad=False)
        # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
        final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
        return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

    def _predict(  # type: ignore
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, *_ = DiffPol.sample_action(
            self.actor_state,
            self.actor_state.params,
            observation,
            self.noise_key,
            self.sampler,
        )
        return actions[0]

    def _predict2(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, _, _, _, la, _ = DiffPol.sample_action(
            self.actor_state,
            self.actor_state.params,
            observation,
            self.noise_key,
            self.sampler,
        )
        actions = (actions, la)
        return actions

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

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
