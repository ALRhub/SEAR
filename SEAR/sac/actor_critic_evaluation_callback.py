import os
import warnings
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import jax
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)

from sac.chunky_eval import evaluate_policy


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        jax_random_key_for_seeds: int,
        actor_chunksize: int,
        rollout_len: int,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.actor_chunksize = actor_chunksize
        self.rollout_len = rollout_len

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self._per_time_is_success_buffer = []
        self.evaluations_successes = []

        # generate a list of 1M random integers, using a jax random key supplied in the args
        seed_list = jax.random.randint(
            jax.random.key(jax_random_key_for_seeds), (10000000,), 0, 2**30 - 1
        )
        # cast to numpy
        self.seed_list = np.array(seed_list)

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        _ = globals_
        info = locals_["info"]
        maybe_is_success = info.get("is_success") or info.get(
            "solved"
        )  # solved for myo suite
        local_per_time_step_buffer = []
        if maybe_is_success is not None:
            local_per_time_step_buffer.append(maybe_is_success)
        if locals_["done"]:
            maybe_is_success = info.get("is_success") or info.get(
                "solved"
            )  # solved for myo suite
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            if len(local_per_time_step_buffer) > 0:
                local_per_time_step_buffer = np.array(local_per_time_step_buffer)
                local_per_time_step_buffer = np.mean(local_per_time_step_buffer, axis=0)
                self._per_time_is_success_buffer.append(
                    1 if local_per_time_step_buffer > 0 else 0
                )

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and (
            self.n_calls % self.eval_freq == 0 or self.n_calls == 1
        ):
            # reset the env with a new seed at the current timestep (reproducibilty)
            self.eval_env.seed(int(self.seed_list[self.n_calls]))

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self._per_time_is_success_buffer = []

            eval_result = evaluate_policy(
                self.model,
                self.eval_env,
                self.actor_chunksize,
                self.rollout_len,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            # When return_episode_rewards=True, we get 3 values
            episode_rewards, episode_lengths, episode_successes = eval_result

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    # kwargs = dict(successes=self.evaluations_successes)

                # np.savez(
                #     self.log_path,
                #     timesteps=self.evaluations_timesteps,
                #     results=self.evaluations_results,
                #     ep_lengths=self.evaluations_length,
                #     **kwargs,
                # )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval-{self.rollout_len} num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            if self.rollout_len > 1:
                suffix = f"_{self.rollout_len}"
            else:
                suffix = ""
            self.logger.record(f"eval/mean_reward{suffix}", float(mean_reward))
            self.logger.record(f"eval/mean_ep_length{suffix}", mean_ep_length)

            # Log success rate from direct episode success tracking
            if len(episode_successes) > 0:
                episode_success_rate = np.mean(episode_successes)
                if self.verbose >= 1:
                    print(f"Episode success rate: {100 * episode_success_rate:.2f}%")
                self.logger.record(
                    f"eval/episode_success_rate{suffix}", episode_success_rate
                )

            # Log success rate from callback-based tracking (legacy)
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                per_time_success_rate = np.mean(self._per_time_is_success_buffer)
                if self.verbose >= 1:
                    print(f"Step-based success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"eval/step_success_rate{suffix}", success_rate)
                self.logger.record(
                    f"eval/per_time_success_rate{suffix}", per_time_success_rate
                )

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
