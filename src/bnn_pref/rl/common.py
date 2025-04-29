from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jaxtyping import Array, Bool, Float


# * Containers for agent training state and transition data
class AgentState(NamedTuple):
    actor: TrainState
    dual_q: TrainState
    dual_q_target: TrainState
    value: TrainState


class Transition(NamedTuple):
    obs: Float[Array, "N O"]  # N = number of transitions
    action: Float[Array, "N A"]
    reward: Float[Array, "N"]
    next_obs: Float[Array, "N O"]
    done: Bool[Array, "N"]


# * IQL networks
class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=2,  # Two Q networks
        )
        q_values = vmap_critic(self.obs_mean, self.obs_std)(obs, action)
        return q_values


class StateValueFunction(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)


class TanhGaussianActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    log_std_max: float = 2.0
    log_std_min: float = -20.0

    @nn.compact
    def __call__(self, x, eval=False):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.tanh(x)
        if eval:
            return distrax.Deterministic(x)
        logstd = self.param(
            "logstd",
            init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
        )
        std = jnp.exp(jnp.clip(logstd, self.log_std_min, self.log_std_max))
        return distrax.Normal(x, std)
