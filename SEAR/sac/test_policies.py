import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jaxtyping import Array

from sac.actor import Actor
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

tfd = tfp.distributions


def create_test_config(
    actor_chunksize: int = 2,
    full_cov: bool = False,
    log_std_min: float = -20.0,
    log_std_max: float = 2.0,
    hs: tuple = (64, 64),
    bias: float = 0.3,
    bias_kind: str = "gp",
    logprob_type: str = "conditional",
) -> Config:
    """Create a test configuration for the Actor."""
    norm_config = NormConfig(norm="none", bn_momentum=0.9, bn_warmup=1000)

    actor_config = ActorConfig(
        norm=norm_config,
        activation="relu",
        full_cov=full_cov,
        logprob_type=logprob_type,
        hs=hs,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        bias=bias,
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
        norm_qk=True,
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


class TestActorLogProbs:
    """Test suite for Actor.__call__ method focusing on log-probability finiteness."""

    @given(
        batch_size=st.integers(min_value=1, max_value=1),
        obs_dim=st.integers(min_value=1, max_value=2),
        actor_chunksize=st.integers(min_value=1, max_value=4),
        action_dim=st.integers(min_value=1, max_value=3),
        full_cov=st.booleans(),
        non_deterministic=st.booleans(),
        train=st.booleans(),
        log_std_min=st.floats(min_value=-20.0, max_value=-5.0),
        log_std_max=st.floats(min_value=-2.0, max_value=-2.0),
        rng_key=st.integers(min_value=0, max_value=0),
        cov_bias=st.floats(min_value=1e-4, max_value=1e-3),
        logprob_type_i=st.integers(min_value=0, max_value=1),
    )
    @settings(max_examples=200, deadline=10000)  # Increase max_examples
    def test_actor_logprobs_are_finite(
        self,
        batch_size,
        obs_dim,
        actor_chunksize,
        action_dim,
        full_cov,
        non_deterministic,
        train,
        log_std_min,
        log_std_max,
        rng_key,
        cov_bias,
        logprob_type_i,
    ):
        """Test that Actor.__call__ returns finite log-probabilities for various configurations."""
        assume(not non_deterministic)  # TODO

        # Ensure log_std_max > log_std_min
        assume(log_std_max > log_std_min)

        logprob_type = ["joint", "marginalized"][logprob_type_i]
        # Create test configuration
        config = create_test_config(
            actor_chunksize=actor_chunksize,
            full_cov=full_cov,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            bias=cov_bias,
            bias_kind="diag",
            logprob_type=logprob_type,
        )

        # Create actor
        actor = Actor(
            action_dim=action_dim * actor_chunksize,
            single_action_dim=action_dim,
            config=config,
        )

        # Convert integer to JAX PRNG key
        rng_key = jax.random.PRNGKey(rng_key)

        # Generate random observations
        key, subkey = jax.random.split(rng_key)
        obs = jax.random.normal(subkey, (batch_size, obs_dim))

        # Initialize actor parameters
        key, params_key, batch_stats_key = jax.random.split(key, 3)
        dummy_obs = jnp.ones((1, obs_dim))
        dummy_key = jax.random.PRNGKey(0)

        init_variables = actor.init(
            {"params": params_key, "batch_stats": batch_stats_key},
            dummy_obs,
            train=train,
            deterministic=not non_deterministic,
            key=dummy_key,
        )

        # Call the actor
        key, call_key, call_params_key, call_batch_stats_key = jax.random.split(key, 4)
        actions, logprobs, debug_cov, _ = actor.apply(  # type: ignore
            init_variables,
            x=obs,
            train=train,
            deterministic=not non_deterministic,
            key=call_key,
            rngs={"params": call_params_key, "batch_stats": call_batch_stats_key},
        )
        actions: Array
        logprobs: Array
        debug_cov: Array

        # Main assertion: all log-probabilities must be finite
        assert jnp.all(
            jnp.isfinite(logprobs)
        ), f"Found non-finite log-probabilities: {logprobs}"

        # Additional checks for robustness
        assert not jnp.any(
            jnp.isnan(logprobs)
        ), f"Found NaN values in log-probabilities: {logprobs}"
        assert not jnp.any(
            jnp.isinf(logprobs)
        ), f"Found infinite values in log-probabilities: {logprobs}"

        # Check that actions are finite
        assert jnp.all(jnp.isfinite(actions)), f"Found non-finite actions: {actions}"

        # Check that debug_cov is finite
        assert jnp.all(
            jnp.isfinite(debug_cov)
        ), f"Found non-finite debug_cov: {debug_cov}"

        # # For non-deterministic mode, check that log-probs are reasonable (not extremely negative)
        # if non_deterministic:
        #     # Log-probs shouldn't be unreasonably negative (e.g., less than -1000)
        #     assert jnp.all(
        #         logprobs > -1000
        #     ), f"Found extremely negative log-probabilities: {logprobs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
