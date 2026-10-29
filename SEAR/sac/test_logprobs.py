import jax
import jax.numpy as jnp
import pytest

from sac.actor_autoregressive import ActorAutoregressive
from sac.config import (
    ActorConfig,
    ActorKind,
    AlgConfig,
    Config,
    CriticConfig,
    CriticKind,
    DistConfig,
    Fn,
    NoiseConfig,
    NoiseKind,
    NormConfig,
    OptimizerConfig,
    RolloutConfig,
    WandbConfig,
)
from sac.actor import Actor


def create_test_config(
    actor_chunksize: int,
    full_cov: bool,
    bias_kind: str,
    logprob_type: str,
    autoregressive: bool,
) -> Config:
    """Create a test configuration for the Actor."""
    norm_config = NormConfig(norm="none", bn_momentum=0.9, bn_warmup=1000)

    actor_config = ActorConfig(
        norm=norm_config,
        activation="relu",
        full_cov=full_cov,
        logprob_type=logprob_type,
        hs=(64, 64),
        log_std_min=-20.0,
        log_std_max=2.0,
        bias=0.001,
        bias_kind=bias_kind,
        kind=ActorKind.MLP,
        only_fit_last_nstep=False,
        update_collate_fn=Fn.MIN,
        simbav2=False,
        weight_norm=False,
        n_embed=128,
        n_heads=8,
        n_layer=2,
        dropout_rate=0.0,
        trpl=False,
        mean_bound=5,
        cov_bound=1,
        bc_coef=0,
        target_quartile=None,
    )

    critic_config = CriticConfig(
        norm=norm_config,
        activation="relu",
        n_critics=2,
        hs=(64, 64),
        simbav2=False,
        simba_blocks=2,
        norm_qk=False,
        only_fit_last_nstep=False,
        update_ensemble_size=2,
        kind=CriticKind.MLP,
        distributional=True,
        use_cramer_von_mises=False,
        dist=DistConfig(11, 0, 1),
        weight_norm=True,
        n_embed=128,
        n_heads=8,
        n_layer=2,
        last_chunk_bias=0,
        last_chunk_bias_decay=100,
        collate_fn="min",
    )

    alg_config = AlgConfig(
        chunky_critic=False,
        random_chunks=False,
        random_init_chunk=True,
        temporl=True,
        scale_chunks=False,
        crossq_style=False,
        use_next_action_logprob=False,
        marginalize_next_action_logprob=False,
        marginalize_actor_logprob=False,
        next_ep_critic_update=False,
        tau=0.005,
        utd=1,
        tot_time_steps=1000000,
        policy_delay=1,
        log_freq=1000,
        critic_chunksize=2,
        actor_chunksize=actor_chunksize,
        ent_coef=0.2,
        ent_coef_init=1,
        apply_next_state_ent_bonus=True,
        gamma=0.99,
        batch_size=256,
        offline_steps=0,
        skip_rl=False,
        buffer_size=1_000_000,
        single_action_v=True,
        target_ent=0,
        train_bc=True,
        rlpd=False,
    )

    optimizer_config = OptimizerConfig(
        actor_lr=3e-4, critic_lr=3e-4, b1=0.9, critic_use_adamw=False
    )
    wandb_config = WandbConfig(
        entity="test", project="test", group="test", job_type="test", activate=False
    )
    rollout_config = RolloutConfig(
        buffer_len=1000, ratio_tolerance=0.05, annealing_eps=0.001, bias=0
    )
    noise_config = NoiseConfig(kind=NoiseKind.NONE, action=0.1, obs=0.1)

    return Config(
        exp_name="test",
        seed=42,
        use_jit=False,
        env_name="test",
        alg=alg_config,
        actor=actor_config,
        critic=critic_config,
        optimizer=optimizer_config,
        wandb=wandb_config,
        rollout=rollout_config,
        noise=noise_config,
        model_save_path=None,
        n_saved_models=1,
    )


def run_logprob_test(
    bias_kind: str,
    logprob_type: str,
    autoregressive: bool,
    full_cov: bool,
    n_samples: int = 10,
):
    """Core test logic that samples actions and checks for NaN logprobs."""
    config = create_test_config(
        actor_chunksize=20,
        full_cov=full_cov,
        bias_kind=bias_kind,
        logprob_type=logprob_type,
        autoregressive=autoregressive,
    )

    # Test parameters
    batch_size = 1
    obs_dim = 4
    action_dim = 20
    rng_key = jax.random.PRNGKey(42)

    # Create actor
    if autoregressive:
        actor = ActorAutoregressive(
            action_dim=action_dim * config.alg.actor_chunksize,
            single_action_dim=config.alg.actor_chunksize,
            config=config,
        )
    else:
        actor = Actor(
            action_dim=action_dim * config.alg.actor_chunksize,
            single_action_dim=config.alg.actor_chunksize,
            config=config,
        )

    # Generate observations
    key, subkey = jax.random.split(rng_key)
    obs = jax.random.normal(subkey, (batch_size, obs_dim))

    # Initialize actor parameters
    key, params_key, batch_stats_key = jax.random.split(key, 3)
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_key = jax.random.PRNGKey(0)

    init_variables = actor.init(
        {"params": params_key, "batch_stats": batch_stats_key},
        dummy_obs,
        train=True,
        deterministic=False,
        key=dummy_key,
    )

    # Sample multiple actions and collect logprobs
    all_logprobs = []
    for _ in range(n_samples):
        key, call_key, call_params_key, call_batch_stats_key = jax.random.split(key, 4)
        _, logprobs, _, _ = actor.apply(  # type: ignore
            init_variables,
            x=obs,
            train=True,
            deterministic=False,
            key=call_key,
            rngs={"params": call_params_key, "batch_stats": call_batch_stats_key},
        )
        all_logprobs.append(logprobs)

    # Stack all logprobs and check for NaN
    all_logprobs = jnp.stack(all_logprobs)
    nan_count = jnp.sum(jnp.isnan(all_logprobs))
    total_count = all_logprobs.size
    nan_fraction = nan_count / total_count

    # Assert less than 1% are NaN
    assert (
        nan_fraction < 0.01
    ), f"NaN fraction {nan_fraction:.4f} >= 0.001 for {bias_kind=}, {logprob_type=}, {autoregressive=}, {full_cov=}"


def test_diag_marginalized_nonauto_diagcov():
    run_logprob_test("diag", "marginalized", False, False)


def test_diag_marginalized_nonauto_fullcov():
    run_logprob_test("diag", "marginalized", False, True)


def test_diag_marginalized_auto_diagcov():
    run_logprob_test("diag", "marginalized", True, False)


def test_diag_marginalized_auto_fullcov():
    run_logprob_test("diag", "marginalized", True, True)


def test_diag_joint_nonauto_diagcov():
    run_logprob_test("diag", "joint", False, False)


def test_diag_joint_nonauto_fullcov():
    run_logprob_test("diag", "joint", False, True)


def test_diag_joint_auto_diagcov():
    run_logprob_test("diag", "joint", True, False)


def test_diag_joint_auto_fullcov():
    run_logprob_test("diag", "joint", True, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
