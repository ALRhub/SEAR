from typing import NamedTuple, Any, Required, TypedDict, Union
import torch as th
from gymnasium import spaces

import flax
import flax.core
import numpy as np
from flax.training.train_state import TrainState
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from scale_rl.agents.simbaV2.simbaV2_update import (
    l2normalize_layer,
    tree_map_until_match,
)


def _l2normalize_params(
    params: flax.core.FrozenDict[str, Any],
    regex: str = "hyper_dense",
) -> flax.core.FrozenDict[str, Any]:
    """scale_rl.agents.simbaV2.simbaV2_update.l2normalize_network adapted
    for use without the Network wrapper"""
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x), tree=params, target_re=regex, keep_values=True
    )
    return new_params  # type: ignore


class ActorTrainState(TrainState):
    batch_stats: flax.core.FrozenDict
    old_params: flax.core.FrozenDict


class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict

    def l2normalize(self) -> "RLTrainState":
        return self.replace(params=_l2normalize_params(self.params))


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    rewards: np.ndarray

    @staticmethod
    def combine(
        sample1: "ReplayBufferSamplesNp", sample2: "ReplayBufferSamplesNp"
    ) -> "ReplayBufferSamplesNp":
        return ReplayBufferSamplesNp(
            observations=np.concatenate(
                [sample1.observations, sample2.observations], axis=0
            ),
            actions=np.concatenate([sample1.actions, sample2.actions], axis=0),
            next_observations=np.concatenate(
                [sample1.next_observations, sample2.next_observations], axis=0
            ),
            dones=np.concatenate([sample1.dones, sample2.dones], axis=0),
            truncated=np.concatenate([sample1.truncated, sample2.truncated], axis=0),
            rewards=np.concatenate([sample1.rewards, sample2.rewards], axis=0),
        )


class OGBenchDataset(TypedDict):
    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
