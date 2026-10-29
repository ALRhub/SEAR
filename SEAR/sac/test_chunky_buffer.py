from gymnasium import Env
import pytest
import numpy as np
import ogbench
from sac.chunky_buffer import ChunkyBuffer, biased_randint
from common.type_aliases import OGBenchDataset, ReplayBufferSamplesNp


def infos(truncated: bool | np.ndarray) -> list[dict]:
    """Helper function to create infos list for ChunkyBuffer.add()"""
    return [{"TimeLimit.truncated": bool(truncated)}]


class TestChunkyBuffer:
    """Test suite for ChunkyBuffer ensuring correct sampling behavior."""

    def test_only_written_data_returned_half_empty(self):
        """Test that sampling only returns data that was actually written."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=2, action_dim=1)

        # Add only 5 transitions (half buffer_size)
        for i in range(5):
            obs = np.array([[float(i), i + 0.5]], dtype=np.float32)
            action = np.array([[i * 0.1]], dtype=np.float32)
            next_obs = np.array([[float(i + 1), i + 1.5]], dtype=np.float32)
            reward = np.array([float(i)], dtype=np.float32)
            done = np.array([i == 4], dtype=bool)  # Last transition ends episode
            buffer.add(obs, next_obs, action, reward, done, infos(done))

        # Should only sample from the written portion
        samples = buffer.sample_chunk(critic_chunksize=2, batch_size=1)

        # Verify shapes
        assert samples.observations.shape == (1, 4)  # batch=1, chunksize=2 * obs_dim=2
        assert samples.actions.shape == (1, 2)  # batch=1, chunksize=2 * action_dim=1
        assert samples.rewards.shape == (1, 2)  # batch=1, chunksize=2

        # Verify values are from written data (0-4 range)
        obs_flat = samples.observations[0]
        assert all(
            0 <= val <= 5.5 for val in obs_flat
        ), f"Found values outside written range: {obs_flat}"

    def test_episode_ends(self):
        """Test that episode ends are actually samples"""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=3, action_dim=2)

        # Add a single episode with identifiable pattern
        for i in range(7):
            obs = np.array([[i, i * 10, i * 100]], dtype=np.float32)
            action = np.array([[i, i * 2]], dtype=np.float32)
            next_obs = np.array(
                [[i + 1, (i + 1) * 10, (i + 1) * 100]], dtype=np.float32
            )
            reward = np.array([float(i)], dtype=np.float32)
            done = np.array([i == 6], dtype=bool)  # End episode at last step
            buffer.add(obs, next_obs, action, reward, done, infos(False))

        # Sample chunk of size 3
        samples = buffer.sample_chunk(critic_chunksize=6, batch_size=1000)
        assert np.any(samples.dones)
        samples = buffer.sample_chunk(critic_chunksize=7, batch_size=1000)
        assert np.any(samples.dones)

    def test_full_chunks(self):
        """Test that episodes with length critic_chunksize are sampled."""
        buffer = ChunkyBuffer(buffer_size=100, obs_dim=3, action_dim=2)

        # add a few too short episodes
        for _ in range(3):
            for i in range(6):
                obs = np.array([[i, i * 10, i * 100]], dtype=np.float32)
                action = np.array([[i, i * 2]], dtype=np.float32)
                next_obs = np.array(
                    [[i + 1, (i + 1) * 10, (i + 1) * 100]], dtype=np.float32
                )
                reward = np.array([float(i)], dtype=np.float32)
                done = np.array([i == 5], dtype=bool)  # End episode at last step
                buffer.add(obs, next_obs, action, reward, done, infos(False))

        # Add a single episode with proper length
        for i in range(7):
            obs = np.array([[i, i * 10, i * 100]], dtype=np.float32)
            action = np.array([[i, i * 2]], dtype=np.float32)
            next_obs = np.array(
                [[i + 1, (i + 1) * 10, (i + 1) * 100]], dtype=np.float32
            )
            reward = np.array([float(i)], dtype=np.float32)
            done = np.array([i == 6], dtype=bool)  # End episode at last step
            buffer.add(obs, next_obs, action, reward, done, infos(False))

        # Does the buffer find the proper episode?
        samples = buffer.sample_chunk(critic_chunksize=7, batch_size=1000)
        assert np.any(samples.dones)

    def test_chunk_stitching_consecutive_data(self):
        """Test that chunks properly stitch together consecutive timesteps."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=3, action_dim=2)

        # Add a single episode with identifiable pattern
        for i in range(7):
            obs = np.array([[i, i * 10, i * 100]], dtype=np.float32)
            action = np.array([[i, i * 2]], dtype=np.float32)
            next_obs = np.array(
                [[i + 1, (i + 1) * 10, (i + 1) * 100]], dtype=np.float32
            )
            reward = np.array([float(i)], dtype=np.float32)
            done = np.array([i == 6], dtype=bool)  # End episode at last step
            buffer.add(obs, next_obs, action, reward, done, infos(done))

        # Sample chunk of size 3
        samples = buffer.sample_chunk(critic_chunksize=3, batch_size=1)

        # Reshape to verify consecutive structure
        obs_chunk = samples.observations[0].reshape(3, 3)  # 3 timesteps, 3 obs dims
        action_chunk = samples.actions[0].reshape(3, 2)  # 3 timesteps, 2 action dims
        reward_chunk = samples.rewards[0]  # 3 timesteps

        # Verify consecutive timesteps
        for t in range(3):
            # Check observation values follow pattern
            assert (
                obs_chunk[t, 1] == obs_chunk[t, 0] * 10
            ), f"Obs pattern broken at timestep {t}"
            assert (
                obs_chunk[t, 2] == obs_chunk[t, 0] * 100
            ), f"Obs pattern broken at timestep {t}"

            # Check action values follow pattern
            assert (
                action_chunk[t, 1] == action_chunk[t, 0] * 2
            ), f"Action pattern broken at timestep {t}"

            # Check reward matches observation first dimension
            assert (
                reward_chunk[t] == obs_chunk[t, 0]
            ), f"Reward mismatch at timestep {t}"

    def test_episode_boundary_handling_at_exact_capacity(self):
        """Test that chunks stay within episode boundaries when the buffer
        is filled to exactly its capacity, the edge of the incrementally
        cached fast path in _episode_boundaries()."""
        buffer = ChunkyBuffer(buffer_size=50, obs_dim=1, action_dim=1)

        for episode_id in range(10):
            for i in range(5):
                is_done = i == 4
                buffer.add(
                    obs=np.array([[float(episode_id)]], dtype=np.float32),
                    action=np.array([[float(episode_id)]], dtype=np.float32),
                    next_obs=np.array([[float(episode_id)]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([is_done], dtype=bool),
                    infos=infos(is_done),
                )

        assert buffer.n_samples == buffer.capacity

        sample = buffer.sample_chunk(critic_chunksize=5, batch_size=256)
        for obs in sample.observations:
            assert (
                len(set(obs.flatten().tolist())) == 1
            ), "Observation crossed episode boundary"

    def test_episode_starts_cache_frozen_after_wrap(self):
        """Test that the incrementally maintained episode-start cache stops
        growing once the buffer wraps, instead of tracking overwritten data."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=1, action_dim=1)

        for episode_id in range(2):
            for i in range(5):
                is_done = i == 4
                buffer.add(
                    obs=np.array([[float(episode_id)]], dtype=np.float32),
                    action=np.array([[float(episode_id)]], dtype=np.float32),
                    next_obs=np.array([[float(episode_id)]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([is_done], dtype=bool),
                    infos=infos(is_done),
                )

        assert buffer.n_samples == buffer.capacity
        cache_at_capacity = list(buffer._episode_starts)

        for i in range(20):
            is_done = i % 4 == 3
            buffer.add(
                obs=np.array([[9.0]], dtype=np.float32),
                action=np.array([[9.0]], dtype=np.float32),
                next_obs=np.array([[9.0]], dtype=np.float32),
                reward=np.array([float(i)], dtype=np.float32),
                done=np.array([is_done], dtype=bool),
                infos=infos(is_done),
            )

        assert buffer.n_samples > buffer.capacity
        assert buffer._episode_starts == cache_at_capacity

    def test_wrap_does_not_mix_episodes(self):
        """Test that sampled chunks never mix data from different episodes
        right after the buffer wraps past capacity."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=1, action_dim=1)

        # Fill exactly to capacity with two distinguishable 5-step episodes.
        for episode_id in range(2):
            for i in range(5):
                is_done = i == 4
                buffer.add(
                    obs=np.array([[float(episode_id)]], dtype=np.float32),
                    action=np.array([[float(episode_id)]], dtype=np.float32),
                    next_obs=np.array([[float(episode_id)]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([is_done], dtype=bool),
                    infos=infos(is_done),
                )

        # One more sample wraps the buffer, overwriting only episode 0's
        # first step. Episode 0's remaining 4 steps are still sitting in
        # the buffer, now stale, with no done-marker separating them from
        # the fresh overwrite.
        buffer.add(
            obs=np.array([[2.0]], dtype=np.float32),
            action=np.array([[2.0]], dtype=np.float32),
            next_obs=np.array([[2.0]], dtype=np.float32),
            reward=np.array([0.0], dtype=np.float32),
            done=np.array([False], dtype=bool),
            infos=infos(False),
        )
        assert buffer.n_samples > buffer.capacity

        sample = buffer.sample_chunk(critic_chunksize=5, batch_size=256)
        for obs in sample.observations:
            assert (
                len(set(obs.flatten().tolist())) == 1
            ), "Observation crossed episode boundary"

    def test_wrap_spanning_multiple_episodes_does_not_mix_episodes(self):
        """Test that sampled chunks don't mix stale leftover data from
        multiple old episodes with an in-progress episode that wraps
        partway through overwriting them."""
        buffer = ChunkyBuffer(buffer_size=20, obs_dim=1, action_dim=1)

        # First pass: 4 complete 5-step episodes, exactly filling capacity.
        for episode_id in range(4):
            for i in range(5):
                is_done = i == 4
                buffer.add(
                    obs=np.array([[float(episode_id)]], dtype=np.float32),
                    action=np.array([[float(episode_id)]], dtype=np.float32),
                    next_obs=np.array([[float(episode_id)]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([is_done], dtype=bool),
                    infos=infos(is_done),
                )

        assert buffer.n_samples == buffer.capacity

        # Second pass: one long, still in-progress episode that wraps and
        # overwrites episodes 0 and 1 entirely, plus the first two steps of
        # episode 2, stopping in the middle of it.
        for i in range(12):
            buffer.add(
                obs=np.array([[4.0]], dtype=np.float32),
                action=np.array([[4.0]], dtype=np.float32),
                next_obs=np.array([[4.0]], dtype=np.float32),
                reward=np.array([float(i)], dtype=np.float32),
                done=np.array([False], dtype=bool),
                infos=infos(False),
            )

        assert buffer.n_samples == 32
        assert buffer.index == 11

        sample = buffer.sample_chunk(critic_chunksize=12, batch_size=256)
        for obs in sample.observations:
            assert (
                len(set(obs.flatten().tolist())) == 1
            ), "Observation crossed episode boundary"

    def test_full_buffer_circular_behavior(self):
        """Test buffer behavior when full (circular overwriting)."""
        buffer = ChunkyBuffer(buffer_size=100, obs_dim=1, action_dim=1)
        chunk_size = 10

        # Fill buffer completely with first episode
        for _eps in range(100):
            for i in np.arange(chunk_size + np.random.randint(-1, 3)):
                buffer.add(
                    obs=np.array([[i]], dtype=np.float32),
                    action=np.array([[i * 0.1]], dtype=np.float32),
                    next_obs=np.array([[i + 1]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([False], dtype=bool),
                    infos=infos(False),
                )
            buffer.add(
                obs=np.array([[4.0]], dtype=np.float32),
                action=np.array([[0.4]], dtype=np.float32),
                next_obs=np.array([[5.0]], dtype=np.float32),
                reward=np.array([4.0], dtype=np.float32),
                done=np.array([True], dtype=bool),
                infos=infos(True),
            )

        # Add more data (should overwrite)
        for _eps in range(200):
            for i in np.arange(chunk_size + np.random.randint(-1, 3)):
                buffer.add(
                    obs=np.array([[11.0]], dtype=np.float32),
                    action=np.array([[i * 0.1]], dtype=np.float32),
                    next_obs=np.array([[i + 1]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([False], dtype=bool),
                    infos=infos(False),
                )
            buffer.add(
                obs=np.array([[11.0]], dtype=np.float32),
                action=np.array([[0.4]], dtype=np.float32),
                next_obs=np.array([[5.0]], dtype=np.float32),
                reward=np.array([4.0], dtype=np.float32),
                done=np.array([True], dtype=bool),
                infos=infos(True),
            )

        # Sample should work with overwritten data
        samples = buffer.sample_chunk(critic_chunksize=2, batch_size=1)

        # Verify we get valid samples from the newer data
        obs_vals = samples.observations
        assert np.allclose(obs_vals.flatten(), 11.0)

    def test_different_dimensions(self):
        """Test buffer with various observation and action dimensions."""
        test_cases = np.array(
            [
                (1, 1),  # minimal
                (5, 3),  # moderate
                (10, 1),  # high obs, low action
                (2, 8),  # low obs, high action
            ]
        )

        for obs_dim, action_dim in test_cases:
            buffer = ChunkyBuffer(
                buffer_size=20, obs_dim=obs_dim, action_dim=action_dim
            )

            # Add complete episode
            episode_length = 8
            for i in range(episode_length):
                obs = np.random.randn(1, obs_dim).astype(np.float32)
                action = np.random.randn(1, action_dim).astype(np.float32)
                next_obs = np.random.randn(1, obs_dim).astype(np.float32)
                reward = np.random.randn(1).astype(np.float32)
                done = np.array([i == episode_length - 1], dtype=bool)

                buffer.add(obs, next_obs, action, reward, done, infos(done))

            # Test sampling with different chunk sizes
            for chunk_size in [1, 3, 5]:
                if chunk_size <= episode_length:
                    samples = buffer.sample_chunk(
                        critic_chunksize=chunk_size, batch_size=2
                    )

                    # Verify dimensions
                    expected_obs_shape = (2, chunk_size * obs_dim)
                    expected_action_shape = (2, chunk_size * action_dim)
                    expected_reward_shape = (2, chunk_size)

                    assert samples.observations.shape == expected_obs_shape
                    assert samples.actions.shape == expected_action_shape
                    assert samples.rewards.shape == expected_reward_shape
                    assert samples.dones.shape == expected_reward_shape

    def test_episode_boundary_handling(self):
        """Test that chunks don't cross episode boundaries."""
        np.random.seed(42)
        buffer = ChunkyBuffer(buffer_size=1_000_000, obs_dim=1, action_dim=1)

        for chunk_size in range(50):
            for i in range(chunk_size):
                is_done = i == chunk_size - 1
                buffer.add(
                    obs=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    action=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    next_obs=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([is_done], dtype=bool),
                    infos=infos(is_done),
                )

        for chunk_size in range(1, 50):
            sample = buffer.sample_chunk(critic_chunksize=chunk_size, batch_size=256)
            sample: ReplayBufferSamplesNp
            for obs in sample.observations:
                assert (
                    len(set(obs.flatten().tolist())) == 1
                ), "Observation crossed episode"

    def test_episode_boundary_handling_truncation(self):
        """Test that chunks don't cross episode boundaries, even if episodes end by truncation"""
        np.random.seed(42)
        buffer = ChunkyBuffer(buffer_size=1_000_000, obs_dim=1, action_dim=1)

        for chunk_size in range(50):
            for i in range(chunk_size):
                is_done = i == chunk_size - 1
                buffer.add(
                    obs=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    action=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    next_obs=np.array([[chunk_size * 1.0]], dtype=np.float32),
                    reward=np.array([float(i)], dtype=np.float32),
                    done=np.array([False], dtype=bool),
                    infos=infos(is_done),
                )

        for chunk_size in range(1, 50):
            sample = buffer.sample_chunk(critic_chunksize=chunk_size, batch_size=256)
            sample: ReplayBufferSamplesNp
            for obs in sample.observations:
                assert (
                    len(set(obs.flatten().tolist())) == 1
                ), "Observation crossed episode"

    def test_insufficient_episode_length_error(self):
        """Test error handling when episodes are shorter than chunk size."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=1, action_dim=1)

        # Add very short episode (2 steps)
        for i in range(2):
            i *= 1.0
            is_done = i == 1
            buffer.add(
                obs=np.array([[i]], dtype=np.float32),
                action=np.array([[i]], dtype=np.float32),
                next_obs=np.array([[i + 1]], dtype=np.float32),
                reward=np.array([float(i)], dtype=np.float32),
                done=np.array([is_done], dtype=bool),
                infos=infos(is_done),
            )

        # Should raise assertion error when chunk size > episode length
        with pytest.raises(AssertionError):
            buffer.sample_chunk(critic_chunksize=3, batch_size=1)

    def test_empty_buffer_handling(self):
        """Test behavior with empty buffer."""
        buffer = ChunkyBuffer(buffer_size=10, obs_dim=2, action_dim=1)

        # Should raise error when trying to sample from empty buffer
        with pytest.raises((IndexError, ValueError, AssertionError)):
            buffer.sample_chunk(critic_chunksize=1, batch_size=1)

    def test_from_ogbench_vectorized_vs_slow(self):
        """Test that vectorized from_ogbench produces the same result as slow version using real OGBench dataset."""
        # Load real OGBench dataset
        env_name = "cube-double-play-singletask-v0"
        env, dataset, _ = ogbench.make_env_and_datasets(env_name)  # type: ignore
        assert env is not None
        assert dataset is not None
        env: Env
        dataset: OGBenchDataset

        # Create buffer kwargs
        buffer_kwargs = {
            "buffer_size": max(dataset["observations"].shape[0], 1_000_000),
            "obs_dim": env.observation_space,
            "action_dim": env.action_space,
        }

        # Load only first 1000 samples to keep test fast
        max_samples = 1000

        # Load with both methods
        buffer_fast = ChunkyBuffer.from_ogbench(
            dataset, buffer_kwargs, max_samples=max_samples
        )
        buffer_slow = ChunkyBuffer.from_ogbench_slow(
            dataset, buffer_kwargs, max_samples=max_samples
        )

        # Compare all buffer contents
        assert buffer_fast.n_samples == buffer_slow.n_samples
        assert buffer_fast.index == buffer_slow.index
        assert buffer_fast.obs_dim == buffer_slow.obs_dim
        assert buffer_fast.action_dim == buffer_slow.action_dim

        # Buffers are allocated with np.empty, so only the written prefix
        # ([:n_samples]) is guaranteed defined; the rest is uninitialized
        # memory and may legitimately differ between the two instances.
        n = buffer_fast.n_samples
        np.testing.assert_array_equal(buffer_fast.obs[:n], buffer_slow.obs[:n])
        np.testing.assert_array_equal(buffer_fast.action[:n], buffer_slow.action[:n])
        np.testing.assert_array_equal(
            buffer_fast.next_obs[:n], buffer_slow.next_obs[:n]
        )
        np.testing.assert_array_equal(buffer_fast.reward[:n], buffer_slow.reward[:n])
        np.testing.assert_array_equal(buffer_fast.done[:n], buffer_slow.done[:n])
        np.testing.assert_array_equal(
            buffer_fast.truncated[:n], buffer_slow.truncated[:n]
        )
        assert buffer_fast._episode_starts == buffer_slow._episode_starts


def test_biased_randint():
    N = 10000
    x = biased_randint(low=np.array([42] * N), high=np.array([52] * N), bias=0)
    assert x.max() == 51
    assert x.min() == 42
    x_biased = biased_randint(
        low=np.array([42] * N), high=np.array([52] * N), bias=1000
    )
    assert x.mean() < x_biased.mean()
    assert x_biased.max() == 51


def test_ogbench_loading():
    """When we load an ogbench dataset, do we ever sample a successful episode end?"""
    env_name = "cube-double-play-singletask-v0"
    env, dataset, _ = ogbench.make_env_and_datasets(env_name)  # type: ignore
    assert env is not None
    assert dataset is not None
    env: Env
    dataset: OGBenchDataset

    buffer_kwargs = {
        "buffer_size": max(dataset["observations"].shape[0], 1_000_000),
        "obs_dim": env.observation_space,
        "action_dim": env.action_space,
    }

    buffer = ChunkyBuffer.from_ogbench(dataset, buffer_kwargs)

    # (1-1e-4)**50_000 = 0.007 % chance of spurious test failure
    sample = buffer.sample_chunk(critic_chunksize=10, batch_size=50000)
    assert sample.rewards[:, -1].flatten().max() == 0

    sample = buffer.sample_chunk(critic_chunksize=10, batch_size=5000, bias=10000)
    assert sample.rewards[:, -1].flatten().max() == 0
