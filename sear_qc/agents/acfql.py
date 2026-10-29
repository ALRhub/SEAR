import copy
from typing import Any
from jaxtyping import Array, Float

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, TransformerValue
from functools import partial

from agents.distributional import DistConfig, project


class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _dist_config(self):
        """Construct DistConfig from agent config."""
        return DistConfig(
            n_atoms=self.config["n_atoms"],
            v_min=self.config["v_min"],
            v_max=self.config["v_max"],
        )

    def _q_values(self, q_output):
        """Convert critic output to scalar Q-values (expected value for distributional)."""
        if self.config["distributional"]:
            dist = self._dist_config()
            q_probs = jax.nn.softmax(q_output, axis=-1)
            leading_shape = q_probs.shape[:-1]
            flat = q_probs.reshape(-1, dist.n_atoms)
            flat_values = jax.vmap(dist.expected_value)(flat)
            return flat_values.reshape(leading_shape)
        return q_output

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        assert self.config["action_chunking"]
        BATCH, CHUNK, ACTION = batch["actions"].shape
        BATCH, CHUNK, OBS = batch["full_observations"].shape
        assert CHUNK == self.config["horizon_length"]

        assert batch["observations"].shape == (BATCH, OBS)
        assert batch["full_observations"].shape == (BATCH, CHUNK, OBS)
        assert batch["actions"].shape == (BATCH, CHUNK, ACTION)
        assert batch["next_actions"].shape == (BATCH, CHUNK, ACTION)
        assert batch["next_observations"].shape == (BATCH, CHUNK, OBS)
        assert batch["rewards"].shape == (BATCH, CHUNK)  # accumulated
        assert batch["masks"].shape == (BATCH, CHUNK)
        assert batch["terminals"].shape == (BATCH, CHUNK)
        assert batch["valid"].shape == (BATCH, CHUNK)

        batch_actions = jnp.reshape(batch["actions"], (BATCH, CHUNK * ACTION))

        q_rng = jax.random.split(rng, num=CHUNK)
        discount = self.config["discount"] ** jnp.ones((BATCH, CHUNK)).cumsum(axis=1)

        if self.config["distributional"]:
            assert self.config["q_agg"] == "mean"
            dist = self._dist_config()

            @partial(jax.vmap, in_axes=(1, 1, 1, 1, 0), out_axes=1)
            def q_target_dist(next_obs, reward, mask, discount, rng):
                assert next_obs.shape == (BATCH, OBS)
                assert reward.shape == (BATCH,)
                assert mask.shape == (BATCH,)
                assert discount.shape == (BATCH,)
                rng, sample_rng = jax.random.split(rng)
                next_actions = self.sample_actions(next_obs, rng=sample_rng)
                assert next_actions.shape == (BATCH, CHUNK * ACTION)

                next_q_logits = self.network.select("target_critic")(
                    next_obs, actions=next_actions
                )
                # next_q_logits: (num_qs, BATCH, CHUNK, n_atoms)
                next_probs = jax.nn.softmax(next_q_logits, axis=-1)
                # Mean over ensemble
                mean_probs = next_probs.mean(axis=0)
                # Take last chunk position
                next_dist = mean_probs[:, -1, :]  # (BATCH, n_atoms)
                # Project with Bellman operator: r + gamma * z
                target = jax.vmap(
                    lambda p, r, m, d: project(p, add=r, scale=d * m, dist=dist)
                )(next_dist, reward, mask, discount)
                return target  # (BATCH, n_atoms)

            target_dist = q_target_dist(
                batch["next_observations"],
                batch["rewards"],
                batch["masks"],
                discount,
                q_rng,
            )
            # target_dist: (BATCH, CHUNK, n_atoms)

            q_logits = self.network.select("critic")(
                batch["observations"], actions=batch_actions, params=grad_params
            )
            # q_logits: (num_qs, BATCH, CHUNK, n_atoms)

            # Mean log-probs over ensemble, then cross-entropy
            log_probs = jax.nn.log_softmax(q_logits, axis=-1)
            mean_log_probs = log_probs.mean(axis=0)  # (BATCH, CHUNK, n_atoms)
            diff = -(target_dist * mean_log_probs).sum(axis=-1)  # (BATCH, CHUNK)

            q = self._q_values(q_logits)
            diff = diff.mean(axis=-1)  # over chunk
        else:

            @partial(jax.vmap, in_axes=(1, 1, 1, 1, 0), out_axes=1)
            def q_target(next_obs, reward, mask, discount, rng):
                assert next_obs.shape == (BATCH, OBS)
                assert reward.shape == (BATCH,)
                assert mask.shape == (BATCH,)
                assert discount.shape == (BATCH,)
                rng, sample_rng = jax.random.split(rng)
                next_actions = self.sample_actions(next_obs, rng=sample_rng)
                assert next_actions.shape == (BATCH, CHUNK * ACTION)

                next_qs = self.network.select("target_critic")(
                    next_obs, actions=next_actions
                )
                next_q_values = self._q_values(next_qs)
                next_q = next_q_values.mean(axis=0)

                target_q = reward + discount * mask * next_q[:, -1]
                return target_q

            target_q = q_target(
                batch["next_observations"],
                batch["rewards"],
                batch["masks"],
                discount,
                q_rng,
            )
            assert target_q.shape == (BATCH, CHUNK)

            q = self.network.select("critic")(
                batch["observations"], actions=batch_actions, params=grad_params
            )
            assert q.shape == (2, BATCH, CHUNK)
            diff = jnp.square(q - target_q)
            diff = diff.mean(axis=0).mean(axis=-1)  # over critics, over chunk

        diff *= batch["valid"][..., -1]
        assert diff.shape == (BATCH,)
        critic_loss = diff.mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, t, params=grad_params
        )

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (
                        batch_size,
                        self.config["horizon_length"],
                        self.config["action_dim"],
                    ),
                )
                * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(
                batch["observations"], noises=noises
            )
            actor_actions = self.network.select("actor_onestep_flow")(
                batch["observations"], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select("critic")(
                batch["observations"], actions=actor_actions
            )
            q = jnp.mean(self._q_values(qs), axis=0)
            assert q.shape == (batch_size, self.config["horizon_length"])
            q_loss = -q[:, -1].mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],  # batch_size
                    self.config["action_dim"]
                    * (
                        self.config["horizon_length"]
                        if self.config["action_chunking"]
                        else 1
                    ),
                ),
            )
            actions = self.network.select("actor_onestep_flow")(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config["action_dim"] * (
                self.config["horizon_length"] if self.config["action_chunking"] else 1
            )
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],  # batch_size
                    self.config["actor_num_samples"],
                    action_dim,
                ),
            )
            observations = jnp.repeat(
                observations[..., None, :], self.config["actor_num_samples"], axis=-2
            )
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            raw_q = self.network.select("critic")(observations, actions)
            q_vals = self._q_values(raw_q)
            if self.config["q_agg"] == "mean":
                q = q_vals.mean(axis=0)
            else:
                q = q_vals.min(axis=0)
            # this assert is only valid during training. During evalutation the batch axis is missing.
            # batch_size = observations.shape[0]
            # assert q.shape == (
            #     batch_size,
            #     self.config["actor_num_samples"],
            #     self.config["horizon_length"],
            # )
            q = q[..., -1]
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(
                actions, (-1, self.config["actor_num_samples"], action_dim)
            )[jnp.arange(bsize), indices, :].reshape(bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_bc_flow_encoder")(observations)
        actions = noises
        # Euler method.
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()
            encoders["actor_onestep_flow"] = encoder_module()

        # Define networks.
        if config["critic_backbone"] == "transformer":
            if config["distributional"]:
                critic_output_dim = config["n_atoms"]
            else:
                critic_output_dim = 1
            critic_def = TransformerValue(
                output_dim=critic_output_dim,
                horizon_length=config["horizon_length"],
                action_dim=action_dim,
                n_embed=config["transformer_n_embed"],
                n_heads=config["transformer_n_heads"],
                num_ensembles=config["num_qs"],
                encoder=encoders.get("critic"),
            )
        else:
            assert not config["distributional"], (
                "Distributional critic only supported with transformer backbone"
            )
            critic_def = Value(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=config["num_qs"],
                encoder=encoders.get("critic"),
                output_dim=config["horizon_length"],
            )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_bc_flow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_onestep_flow"),
        )

        network_info = dict(
            actor_bc_flow=(
                actor_bc_flow_def,
                (ex_observations, full_actions, ex_times),
            ),
            actor_onestep_flow=(
                actor_onestep_flow_def,
                (ex_observations, full_actions),
            ),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get("actor_bc_flow") is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info["actor_bc_flow_encoder"] = (
                encoders.get("actor_bc_flow"),
                (ex_observations,),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.0:
            network_tx = optax.chain(
                optax.adamw(
                    learning_rate=config["lr"], weight_decay=config["weight_decay"]
                ),
                optax.clip_by_global_norm(max_norm=config["max_grad_norm"]),
            )
        else:
            network_tx = optax.chain(
                optax.adam(learning_rate=config["lr"]),
                optax.clip_by_global_norm(max_norm=config["max_grad_norm"]),
            )
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="acfql",  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(
                list
            ),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg="mean",  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2,  # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,
            max_grad_norm=300.0,
            critic_backbone="mlp",  # "mlp" or "transformer"
            transformer_n_embed=512,  # Transformer embedding dimension.
            transformer_n_heads=16,  # Number of attention heads.
            distributional=False,  # Use C51-style distributional critic.
            n_atoms=101,  # Number of atoms for distributional critic.
            v_min=-300.0,  # Min value support for distributional critic.
            v_max=0.0,  # Max value support for distributional critic.
        )
    )
    return config
