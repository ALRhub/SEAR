import numpy as np

from common.type_aliases import ReplayBufferSamplesNp


def test_replay_buffer_samples_combine():
    batch_size_1 = 5
    batch_size_2 = 3
    obs_dim = 4
    action_dim = 2

    sample1 = ReplayBufferSamplesNp(
        observations=np.random.rand(batch_size_1, obs_dim),
        actions=np.random.rand(batch_size_1, action_dim),
        next_observations=np.random.rand(batch_size_1, obs_dim),
        dones=np.random.rand(batch_size_1),
        truncated=np.random.rand(batch_size_1),
        rewards=np.random.rand(batch_size_1),
    )

    sample2 = ReplayBufferSamplesNp(
        observations=np.random.rand(batch_size_2, obs_dim),
        actions=np.random.rand(batch_size_2, action_dim),
        next_observations=np.random.rand(batch_size_2, obs_dim),
        dones=np.random.rand(batch_size_2),
        truncated=np.random.rand(batch_size_2),
        rewards=np.random.rand(batch_size_2),
    )

    combined = ReplayBufferSamplesNp.combine(sample1, sample2)

    assert combined.observations.shape == (batch_size_1 + batch_size_2, obs_dim)
    assert combined.actions.shape == (batch_size_1 + batch_size_2, action_dim)
    assert combined.next_observations.shape == (batch_size_1 + batch_size_2, obs_dim)
    assert combined.dones.shape == (batch_size_1 + batch_size_2,)
    assert combined.truncated.shape == (batch_size_1 + batch_size_2,)
    assert combined.rewards.shape == (batch_size_1 + batch_size_2,)

    np.testing.assert_array_equal(
        combined.observations[:batch_size_1], sample1.observations
    )
    np.testing.assert_array_equal(
        combined.observations[batch_size_1:], sample2.observations
    )
    np.testing.assert_array_equal(combined.actions[:batch_size_1], sample1.actions)
    np.testing.assert_array_equal(combined.actions[batch_size_1:], sample2.actions)
    np.testing.assert_array_equal(
        combined.next_observations[:batch_size_1], sample1.next_observations
    )
    np.testing.assert_array_equal(
        combined.next_observations[batch_size_1:], sample2.next_observations
    )
    np.testing.assert_array_equal(combined.dones[:batch_size_1], sample1.dones)
    np.testing.assert_array_equal(combined.dones[batch_size_1:], sample2.dones)
    np.testing.assert_array_equal(combined.truncated[:batch_size_1], sample1.truncated)
    np.testing.assert_array_equal(combined.truncated[batch_size_1:], sample2.truncated)
    np.testing.assert_array_equal(combined.rewards[:batch_size_1], sample1.rewards)
    np.testing.assert_array_equal(combined.rewards[batch_size_1:], sample2.rewards)


def test_replay_buffer_samples_combine_empty():
    batch_size = 3
    obs_dim = 4
    action_dim = 2

    sample1 = ReplayBufferSamplesNp(
        observations=np.random.rand(batch_size, obs_dim),
        actions=np.random.rand(batch_size, action_dim),
        next_observations=np.random.rand(batch_size, obs_dim),
        dones=np.random.rand(batch_size),
        truncated=np.random.rand(batch_size),
        rewards=np.random.rand(batch_size),
    )

    sample2 = ReplayBufferSamplesNp(
        observations=np.empty((0, obs_dim)),
        actions=np.empty((0, action_dim)),
        next_observations=np.empty((0, obs_dim)),
        dones=np.empty((0,)),
        truncated=np.empty((0,)),
        rewards=np.empty((0,)),
    )

    combined = ReplayBufferSamplesNp.combine(sample1, sample2)

    assert combined.observations.shape == (batch_size, obs_dim)
    assert combined.actions.shape == (batch_size, action_dim)
    assert combined.next_observations.shape == (batch_size, obs_dim)
    assert combined.dones.shape == (batch_size,)
    assert combined.truncated.shape == (batch_size,)
    assert combined.rewards.shape == (batch_size,)

    np.testing.assert_array_equal(combined.observations, sample1.observations)
    np.testing.assert_array_equal(combined.actions, sample1.actions)
    np.testing.assert_array_equal(combined.next_observations, sample1.next_observations)
    np.testing.assert_array_equal(combined.dones, sample1.dones)
    np.testing.assert_array_equal(combined.truncated, sample1.truncated)
    np.testing.assert_array_equal(combined.rewards, sample1.rewards)
