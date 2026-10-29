from gymnasium.spaces.box import Box
from jaxtyping import Bool, Float, Int, jaxtyped
import numpy as np
from typeguard import typechecked
import numba

from .utils import Any
from common.type_aliases import (
    OGBenchDataset,
    ReplayBufferSamplesNp,
)


class ChunkyBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int | Box,
        action_dim: int | Box,
        device: str = "",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        assert n_envs == 1
        assert not optimize_memory_usage

        if isinstance(obs_dim, Box):
            obs_dim = sum(obs_dim.shape)
        if isinstance(action_dim, Box):
            action_dim = sum(action_dim.shape)

        _ = device
        self.index = -1
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = buffer_size
        self.n_samples = 0

        # Incrementally maintained episode-start indices (index 0 always
        # starts an "episode", even before any data has been added). Kept
        # up to date in add() while the buffer hasn't wrapped yet, so
        # sample_chunk() can avoid rescanning done/truncated from scratch.
        # Once the buffer wraps (n_samples > capacity), old entries may
        # point at overwritten data, so we stop maintaining it and fall
        # back to a full rescan in sample_chunk() instead.
        self._episode_starts: list[int] = [0]

        self.obs = np.empty((buffer_size, obs_dim), dtype=np.float32)
        self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
        self.next_obs = np.empty((buffer_size, obs_dim), dtype=np.float32)
        self.done = np.empty((buffer_size,), dtype=bool)
        self.truncated = np.empty((buffer_size,), dtype=bool)
        self.reward = np.empty((buffer_size,), dtype=np.float32)

    @jaxtyped(typechecker=typechecked)
    def add(
        self,
        obs: Float[Any, "1 obs"],
        next_obs: Float[Any, "1 obs"],
        action: Float[Any, "1 action"],
        reward: Float[Any, "1"],
        done: Bool[Any, "1"],
        infos: list[dict[str, Any]],
    ):
        assert len(infos) == 1
        truncated = infos[0]["TimeLimit.truncated"]
        assert isinstance(truncated, bool)

        obs = obs.flatten()
        next_obs = next_obs.flatten()
        action = action.flatten()
        reward = reward.flatten()
        assert obs.shape == (self.obs_dim,)
        assert action.shape == (self.action_dim,)
        assert next_obs.shape == (self.obs_dim,)

        self.index = (self.index + 1) % self.capacity
        self.n_samples += 1

        self.obs[self.index] = obs
        self.action[self.index] = action
        self.next_obs[self.index] = next_obs
        self.reward[self.index] = reward[0]
        self.done[self.index] = bool(done)
        self.truncated[self.index] = truncated

        if self.n_samples <= self.capacity and (bool(done) or truncated):
            # self.index + 1 == self.n_samples here (buffer hasn't wrapped)
            self._episode_starts.append(self.n_samples)

    def sample_chunk(
        self,
        critic_chunksize: int,
        batch_size: int,
        env: Any = None,
        bias: float = 0.0,
    ) -> ReplayBufferSamplesNp:
        _ = env
        assert self.n_samples, "You can't sample samples without adding samples."
        episode_start, episode_end = self._episode_boundaries()

        (
            observations,
            actions,
            next_observations,
            dones,
            truncated,
            rewards,
        ) = _sample_chunk(
            critic_chunksize=critic_chunksize,
            batch_size=batch_size,
            bias=bias,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.capacity,
            episode_start=episode_start,
            episode_end=episode_end,
            obs=self.obs,
            action=self.action,
            next_obs=self.next_obs,
            done=self.done,
            reward=self.reward,
            truncated=self.truncated,
        )

        result = ReplayBufferSamplesNp(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            truncated=truncated,
            rewards=rewards,
        )
        return result

    def _scan_episode_starts(self, up_to: int) -> list[int]:
        """Full rescan of done/truncated[:up_to] for episode-start indices.
        Only valid while the buffer hasn't wrapped, so physical position
        equals absolute sample index."""
        boundary = np.flatnonzero(np.logical_or(self.done, self.truncated)[:up_to]) + 1
        return [0] + boundary.tolist()

    def _scan_episode_starts_wrapped(self) -> Int[Any, " n"]:
        """Full rescan of done/truncated for episode-start indices once the
        buffer has wrapped, expressed in absolute (monotonically
        increasing, never-wrapping) sample-index space.

        Physical position no longer equals absolute sample index once the
        buffer has overwritten data, so we can't scan done/truncated in
        physical order starting at 0: whatever sits right after
        self.index is stale leftover data from an older episode, not a
        continuation of the one currently being written. We instead
        rotate the view to start at the oldest surviving sample (physical
        position n_samples % capacity) and walk forward in true temporal
        order.
        """
        capacity = self.capacity
        oldest_live_abs = self.n_samples - capacity
        oldest_live_phys = self.n_samples % capacity
        rotated_done = np.roll(self.done, -oldest_live_phys)
        rotated_truncated = np.roll(self.truncated, -oldest_live_phys)
        boundary = np.flatnonzero(np.logical_or(rotated_done, rotated_truncated)) + 1

        episode_start = np.empty(boundary.size + 1, dtype=np.int64)
        episode_start[0] = oldest_live_abs
        episode_start[1:] = boundary + oldest_live_abs
        return episode_start

    def _episode_boundaries(self) -> tuple[Int[Any, " n"], Int[Any, " n"]]:
        if self.n_samples <= self.capacity:
            # Fast path: buffer hasn't wrapped, so the incrementally
            # maintained cache from add() is complete and valid, and
            # physical position equals absolute sample index.
            episode_start = np.array(self._episode_starts, dtype=np.int64)
        else:
            # Slow path: the buffer has wrapped at least once, so cached
            # episode-start indices may point at data that has since been
            # overwritten. We fall back to a full O(capacity) rescan here,
            # in absolute sample-index space, correct but not optimized:
            # buffers are normally sized to outlive the whole run, so this
            # path is not expected to be hit in practice. _sample_chunk()
            # already reads chunks via `% capacity`, so absolute indices
            # beyond capacity work unchanged there.
            episode_start = self._scan_episode_starts_wrapped()

        episode_end = np.empty_like(episode_start)
        episode_end[:-1] = episode_start[1:]
        episode_end[-1] = self.n_samples
        return episode_start, episode_end

    @staticmethod
    def from_ogbench(
        ds: OGBenchDataset, buffer_kwargs: dict, max_samples: int | None = None
    ) -> "ChunkyBuffer":
        """
        Create ChunkyBuffer from OGBench dataset using vectorized operations.

        Args:
            ds: OGBench dataset dictionary
            buffer_kwargs: Arguments to pass to ChunkyBuffer constructor
            max_samples: If provided, only load first N samples from dataset
        """
        buffer_kwargs = buffer_kwargs or {}

        # Slice arrays if max_samples is specified
        if max_samples is not None:
            observations = ds["observations"][:max_samples]
            actions = ds["actions"][:max_samples]
            terminals = ds["terminals"][:max_samples]
            next_observations = ds["next_observations"][:max_samples]
            rewards = ds["rewards"][:max_samples]
            masks = ds["masks"][:max_samples]
        else:
            observations = ds["observations"]
            actions = ds["actions"]
            terminals = ds["terminals"]
            next_observations = ds["next_observations"]
            rewards = ds["rewards"]
            masks = ds["masks"]

        n_samples = observations.shape[0]
        res = ChunkyBuffer(**buffer_kwargs)

        # Vectorized translation of ogbench's masks/terminals into SB3's done/truncated
        # From the ogbench README:
        #   * `masks` denotes whether the agent should get a Bellman backup from the next observation.
        #   > It is 0 only when the task is complete (and 1 otherwise).
        #   > In this case, the agent should set the target Q-value to 0,
        #   > instead of using the next observation's target Q-value.
        #   > * `terminals` simply denotes whether the dataset trajectory is over,
        #   > regardless of task completion.

        # Truth table:
        #        terminal | trunc |  mask
        # ----------------|-------|--------
        #               0 |     0 |       1
        #               0 |     1 | invalid
        #               1 |     0 |       0
        #               1 |     1 |       1

        # Vectorized computation
        truncated = np.logical_and(terminals, masks)
        done = np.logical_not(masks)

        # Flatten observations and actions to match expected dimensions
        obs_flat = observations.reshape(n_samples, -1)
        actions_flat = actions.reshape(n_samples, -1)
        next_obs_flat = next_observations.reshape(n_samples, -1)

        # Directly populate buffer arrays
        res.obs[:n_samples] = obs_flat
        res.action[:n_samples] = actions_flat
        res.next_obs[:n_samples] = next_obs_flat
        res.reward[:n_samples] = rewards.flatten()
        res.done[:n_samples] = done
        res.truncated[:n_samples] = truncated

        # Update buffer state
        res.index = n_samples - 1
        res.n_samples = n_samples

        # from_ogbench bypasses add(), so the episode-start cache is still
        # just its initial [0]. Seed it with a one-time rescan (same logic
        # _episode_boundaries() uses for its slow path) so later
        # sample_chunk() calls can use the fast path.
        if n_samples <= res.capacity:
            res._episode_starts = res._scan_episode_starts(n_samples)

        return res

    @staticmethod
    def from_ogbench_slow(
        ds: OGBenchDataset, buffer_kwargs: dict, max_samples: int | None = None
    ) -> "ChunkyBuffer":
        """
        Create ChunkyBuffer from OGBench dataset using iterative approach (slow).
        Kept for comparison and testing purposes.

        Args:
            ds: OGBench dataset dictionary
            buffer_kwargs: Arguments to pass to ChunkyBuffer constructor
            max_samples: If provided, only load first N samples from dataset
        """
        buffer_kwargs = buffer_kwargs or {}
        res = ChunkyBuffer(**buffer_kwargs)

        # Prepare data slices
        if max_samples is not None:
            observations = ds["observations"][:max_samples]
            actions = ds["actions"][:max_samples]
            terminals = ds["terminals"][:max_samples]
            next_observations = ds["next_observations"][:max_samples]
            rewards = ds["rewards"][:max_samples]
            masks = ds["masks"][:max_samples]
        else:
            observations = ds["observations"]
            actions = ds["actions"]
            terminals = ds["terminals"]
            next_observations = ds["next_observations"]
            rewards = ds["rewards"]
            masks = ds["masks"]

        for obs, action, terminal, next_obs, reward, mask in zip(
            observations,
            actions,
            terminals,
            next_observations,
            rewards,
            masks,
        ):
            # we need to translate ogbench's masks/terminals into SB3's done/trunkated
            # From the ogbench README:

            #   * `masks` denotes whether the agent should get a Bellman backup from the next observation.
            #   > It is 0 only when the task is complete (and 1 otherwise).
            #   > In this case, the agent should set the target Q-value to 0,
            #   > instead of using the next observation's target Q-value.
            #   > * `terminals` simply denotes whether the dataset trajectory is over,
            #   > regardless of task completion.

            # This leaves us with the following truth table

            #        terminal | trunc |  mask
            # ----------------|-------|--------
            #               0 |     0 |       1
            #               0 |     1 | invalid
            #               1 |     0 |       0
            #               1 |     1 |       1

            # therefore:

            truncated = terminal and mask

            res.add(
                obs=obs.reshape(1, -1),
                action=action.reshape(1, -1),
                next_obs=next_obs.reshape(1, -1),
                done=np.array([not mask], dtype=bool),
                reward=np.array([reward]),
                infos=[{"TimeLimit.truncated": bool(truncated)}],
            )
        return res


# @jaxtyped(typechecker=typechecked)
@numba.njit
def biased_randint(
    low: Int[Any, " n"],  # 1D int array
    high: Int[Any, " n"],  # 1D int array
    bias: float,  # in [0, inf)
) -> Int[Any, " n"]:
    res = np.zeros_like(low, dtype=np.int64)
    for i in range(len(low)):
        indices = np.arange(high[i] - low[i])
        weights = indices + 1.0
        weights /= np.sum(weights)
        weights = np.exp(weights * bias)
        probs = weights / np.sum(weights)
        x = np.random.uniform(0, 1)
        chosen_index = np.searchsorted(np.cumsum(probs), x)
        res[i] = low[i] + chosen_index
    return res


@numba.njit
def unbiased_randint(
    low: Int[Any, " n"],  # 1D int array
    high: Int[Any, " n"],  # 1D int array
) -> Int[Any, " n"]:
    res = np.zeros_like(low, dtype=np.int64)
    for i in range(len(low)):
        res[i] = np.random.randint(low[i], high[i])
    return res


@numba.njit(parallel=True)
def _sample_chunk(
    critic_chunksize: int,
    batch_size: int,
    bias: float,
    obs_dim: int,
    action_dim: int,
    capacity: int,
    episode_start: Int[Any, " n_episodes"],
    episode_end: Int[Any, " n_episodes"],
    obs,
    action,
    next_obs,
    done,
    reward,
    truncated,
):
    valid_episodes_starts = np.flatnonzero(
        episode_end - episode_start >= critic_chunksize
    )
    assert valid_episodes_starts.size, "There are no episodes with the given chunksize"
    episode_indices = np.random.choice(
        valid_episodes_starts, replace=True, size=batch_size
    )

    # how far into the episode can our chunk start?
    max_offset = episode_end - episode_start - critic_chunksize + 1
    assert np.all(max_offset[episode_indices] >= 0)

    if bias == 0.0:
        chunk_start_indices = unbiased_randint(
            episode_start[episode_indices],  # low
            episode_start[episode_indices] + max_offset[episode_indices],  # high
        )
    else:
        chunk_start_indices = biased_randint(
            low=episode_start[episode_indices],
            high=episode_start[episode_indices] + max_offset[episode_indices],
            bias=bias,
        )

    result_observations = np.empty(
        (batch_size, critic_chunksize * obs_dim), dtype=np.float32
    )
    result_actions = np.empty(
        (batch_size, critic_chunksize * action_dim), dtype=np.float32
    )
    result_next_observations = np.empty(
        (batch_size, critic_chunksize * obs_dim), dtype=np.float32
    )
    result_dones = np.empty((batch_size, critic_chunksize), dtype=np.float32)
    result_truncated = np.empty((batch_size, critic_chunksize), dtype=np.float32)
    result_rewards = np.empty((batch_size, critic_chunksize), dtype=np.float32)

    for i in numba.prange(batch_size):
        start = chunk_start_indices[i]
        # chunks may wrap around the buffer; iterate chunk_i in-order so writes
        # to each output row stay sequential (row-major => cache-friendly)
        for chunk_i in range(critic_chunksize):
            buffer_i = (start + chunk_i) % capacity
            result_observations[i, chunk_i * obs_dim : (chunk_i + 1) * obs_dim] = obs[
                buffer_i
            ]
            result_actions[i, chunk_i * action_dim : (chunk_i + 1) * action_dim] = (
                action[buffer_i]
            )
            result_next_observations[i, chunk_i * obs_dim : (chunk_i + 1) * obs_dim] = (
                next_obs[buffer_i]
            )
            result_dones[i, chunk_i] = done[buffer_i] * (1 - truncated[buffer_i])
            result_truncated[i, chunk_i] = truncated[buffer_i]
            result_rewards[i, chunk_i] = reward[buffer_i]
    return (
        result_observations,
        result_actions,
        result_next_observations,
        result_dones,
        result_truncated,
        result_rewards,
    )
