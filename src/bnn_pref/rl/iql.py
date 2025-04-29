import os
import warnings
from functools import partial
from typing import Callable, List, Tuple

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import d4rl
import flax.linen as nn
import gym
import hydra
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray

from bnn_pref.rl.common import (
    AgentState,
    DualQNetwork,
    StateValueFunction,
    TanhGaussianActor,
    Transition,
)
from bnn_pref.utils.utils import get_random_seed

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


def create_ts(
    rl_cfg,
    rng: PRNGKeyArray,
    network: nn.Module,
    dummy_input: List[jnp.ndarray],
) -> TrainState:
    lr_schedule = optax.cosine_decay_schedule(rl_cfg.lr, rl_cfg.n_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr_schedule, eps=1e-5),
    )


def eval_agent(
    rl_cfg,
    rng: PRNGKeyArray,
    env: gym.Env,
    agent_state: AgentState,
) -> Float[Array, "n_workers"]:
    # --- Reset environment ---
    step = 0
    returned = np.zeros(rl_cfg.n_eval_workers).astype(bool)
    cum_reward = np.zeros(rl_cfg.n_eval_workers)
    rng, rng_reset = jr.split(rng)
    rng_reset = jr.split(rng_reset, rl_cfg.n_eval_workers)
    obs = env.reset()

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jr.split(rng)
        rng_step = jr.split(rng_step, rl_cfg.n_eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(np.array(action))

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


def make_train_step(
    rl_cfg,
    actor_apply_fn: Callable,
    q_apply_fn: Callable,
    value_apply_fn: Callable,
    dataset: Transition,
) -> Callable:
    """Make JIT-compatible agent train step."""

    def _train_step(carry: Tuple[PRNGKeyArray, AgentState], _):
        rng, agent_state = carry

        # --- Sample batch ---
        rng, rng_batch = jr.split(rng)
        batch_indices = jr.randint(rng_batch, (rl_cfg.batch_size,), 0, len(dataset.obs))
        batch = jax.tree.map(lambda x: x[batch_indices], dataset)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.dual_q.params,
            agent_state.dual_q_target.params,
            rl_cfg.polyak_step_size,
        )
        updated_q_target = agent_state.dual_q_target.replace(
            step=agent_state.dual_q_target.step + 1,
            params=updated_q_target_params,
        )
        agent_state = agent_state._replace(dual_q_target=updated_q_target)

        # --- Compute targets ---
        v_target = q_apply_fn(agent_state.dual_q_target.params, batch.obs, batch.action)
        v_target = v_target.min(-1)
        next_v_target = value_apply_fn(agent_state.value.params, batch.next_obs)
        q_targets = batch.reward + rl_cfg.gamma * (1 - batch.done) * next_v_target

        # --- Update Q and value functions ---
        @jax.value_and_grad
        def _q_loss_fn(params):
            # Compute loss for both critics
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            q_loss = jnp.square(q_pred - jnp.expand_dims(q_targets, axis=-1)).mean()
            return q_loss

        @partial(jax.value_and_grad, has_aux=True)
        def _value_loss_fn(params):
            adv = v_target - value_apply_fn(params, batch.obs)
            # Asymmetric L2 loss
            value_loss = jnp.abs(rl_cfg.iql_tau - (adv < 0.0).astype(float)) * (adv**2)
            return jnp.mean(value_loss), adv

        q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
        (v_loss, adv), v_grad = _value_loss_fn(agent_state.value.params)
        agent_state = agent_state._replace(
            dual_q=agent_state.dual_q.apply_gradients(grads=q_grad),
            value=agent_state.value.apply_gradients(grads=v_grad),
        )

        # --- Update actor ---
        exp_adv = jnp.exp(adv * rl_cfg.beta).clip(max=rl_cfg.exp_adv_clip)

        @jax.value_and_grad
        def _actor_loss_function(params):
            def _compute_loss(transition, exp_adv):
                pi = actor_apply_fn(params, transition.obs)
                bc_loss = -pi.log_prob(transition.action)
                return exp_adv * bc_loss.sum()

            actor_loss = jax.vmap(_compute_loss)(batch, exp_adv)
            return actor_loss.mean()

        actor_loss, actor_grad = _actor_loss_function(agent_state.actor.params)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        loss = {
            "value_loss": v_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
        return (rng, agent_state), loss

    return _train_step


@hydra.main(config_name="config", config_path="../../cfg")
def main(cfg):
    # --- Parse arguments ---
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    rng = jr.key(seed)
    rl_cfg = cfg["rl"]
    task_cfg = cfg["task"]
    # --- Initialize logger ---
    if rl_cfg.use_wandb:
        wandb.init(
            config=rl_cfg,
            project=rl_cfg.wandb_project,
            entity=rl_cfg.wandb_team,
            group=rl_cfg.wandb_group,
            job_type="train_agent",
        )

    # --- Initialize environment and dataset ---
    env = gym.vector.make(task_cfg.name, num_envs=rl_cfg.n_eval_workers)
    dataset = d4rl.qlearning_dataset(gym.make(task_cfg.name))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    # --- Initialize agent and value networks ---
    num_actions = env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_net = TanhGaussianActor(num_actions, obs_mean, obs_std)
    q_net = DualQNetwork(obs_mean, obs_std)
    value_net = StateValueFunction(obs_mean, obs_std)

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q, rng_value = jr.split(rng, 4)
    agent_state = AgentState(
        actor=create_ts(rl_cfg, rng_actor, actor_net, [dummy_obs]),
        dual_q=create_ts(rl_cfg, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=create_ts(rl_cfg, rng_q, q_net, [dummy_obs, dummy_action]),
        value=create_ts(rl_cfg, rng_value, value_net, [dummy_obs]),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        rl_cfg, actor_net.apply, q_net.apply, value_net.apply, dataset
    )

    num_evals = rl_cfg.n_updates // rl_cfg.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            init=(rng, agent_state),
            length=rl_cfg.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jr.split(rng)
        returns = eval_agent(rl_cfg, rng_eval, env, agent_state)
        scores = d4rl.get_normalized_score(task_cfg.name, returns) * 100.0

        # --- Log metrics ---
        step = (eval_idx + 1) * rl_cfg.eval_interval
        print(
            f"Step: {step} / {rl_cfg.n_updates} ({eval_idx + 1:02d}/{num_evals}) | "
            f"Score: {scores.mean():.2f} ± {scores.std():.2f}"
        )
        if rl_cfg.use_wandb:
            log_dict = {
                "return": returns.mean(),
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    if rl_cfg.n_eval_episodes > 0:
        final_iters = int(np.ceil(rl_cfg.n_eval_episodes / rl_cfg.n_eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jr.split(rng, final_iters)
        rets = np.concatenate(
            [eval_agent(rl_cfg, _rng, env, agent_state) for _rng in _rng]
        )  # (n_eval_workers * final_iters)
        scores = d4rl.get_normalized_score(task_cfg.name, rets) * 100.0
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")
        print(
            f"{task_cfg.name}\n"
            f"  {rl_cfg.n_eval_episodes} episodes {rl_cfg.n_eval_workers} workers\n"
            f"  final return: {rets.mean():.2f} ± {rets.std():.2f}\n"
            f"  final normalized score: {scores.mean():.2f} ± {scores.std():.2f}\n"
        )

        # --- Write final returns to file ---
        filename = f"{rl_cfg.name}_{task_cfg.name}.npz"
        with open(f"{cfg.paths.output_dir}/{filename}", "wb") as f:
            np.savez_compressed(f, **info, args=rl_cfg)

        if rl_cfg.use_wandb:
            wandb.save(f"{cfg.paths.output_dir}/{filename}")

    if rl_cfg.use_wandb:
        wandb.finish()


def load_reward_model(
    key: PRNGKeyArray,
    ckpts_dir: str,
    cfg,
    task_name: str = "halfcheetah-random-v2",
    alg: str = "ekf",
    is_al: bool = False,
) -> Callable[
    [Float[Array, "T D"]],
    Float[Array, "T"],
]:
    """
    model shape (n_seeds, ensemble_size, ...)
    key will be used to sample models for ekf
    """
    import distrax
    import orbax
    import orbax.checkpoint

    from bnn_pref.alg.agent_utils import subspace2full_params
    from bnn_pref.alg.ekf_subspace import EKFBeliefState
    from bnn_pref.alg.ensemble import init_model
    from bnn_pref.utils.network import RewardNet

    ckpt_fp = f"{ckpts_dir}/{task_name}_{alg}_al={is_al}"

    cktper = orbax.checkpoint.PyTreeCheckpointer()
    empty_stats = cktper.restore(ckpt_fp)

    obs_shape = gym.make(task_name).observation_space.shape
    if alg == "sgd":
        key, *keys = jr.split(key, 1 + cfg["sgd"]["M"])
        keys = jnp.array(keys)
        model = RewardNet(cfg["sgd"]["hidden_sizes"])
        item = jax.vmap(init_model, in_axes=(0, None, None, None))(
            keys, model, (2, 50, *obs_shape), optax.adam(cfg["sgd"]["learning_rate"])
        )
        ts = cktper.restore(ckpt_fp, item=item)
        params = {"params": ts.params}

    elif alg == "ekf":
        model = RewardNet(cfg["ekf"]["hidden_sizes"])
        item = EKFBeliefState(
            mean=empty_stats["mean"],
            cov=empty_stats["cov"],
            t=empty_stats["t"],
            proj_matrix=empty_stats["proj_matrix"],
            offset_ts=init_model(
                key, model, (2, 50, *obs_shape), optax.adam(cfg["ekf"]["learning_rate"])
            ),
        )
        bel = cktper.restore(ckpt_fp, item=item)
        ts = bel.offset_ts

        params_offset, unravel_fn = ravel_pytree(ts.params)

        distr = distrax.MultivariateNormalFullCovariance(bel.mean, bel.cov)
        ss_params = distr.sample(seed=key, sample_shape=(cfg["ekf"]["M"],))
        full_params_flattened = jax.vmap(subspace2full_params, in_axes=(0, None, None))(
            ss_params, bel.proj_matrix, params_offset
        )
        params = jax.vmap(unravel_fn)(full_params_flattened)
        params = {"params": params}

    else:
        raise ValueError(f"Algorithm {alg} not supported")

    def predict_reward(obs: Float[Array, "T D"]) -> Float[Array, "T"]:
        """M = ensemble size"""
        apply_fn = partial(ts.apply_fn, method=model.predict_traj_rewards)
        out_MT = jax.vmap(apply_fn, in_axes=(0, None))(params, obs)
        return out_MT.mean(axis=0)

    return predict_reward


if __name__ == "__main__":
    # main()

    from omegaconf import OmegaConf

    output_dir = "/Users/yutai/dev/projects/bnn_pref/results/20250428_171621"
    ckpts_dir = f"{output_dir}/ckpts"
    cfg = OmegaConf.load(f"{output_dir}/.hydra/config.yaml")

    task_name = "halfcheetah-random-v2"
    alg = "sgd"
    is_al = False

    key = jr.key(0)
    predict_reward = load_reward_model(
        key=key,
        ckpts_dir=ckpts_dir,
        cfg=cfg,
        task_name=task_name,
        alg=alg,
        is_al=is_al,
    )  # (T,D) -> (T,)

    obs = jnp.zeros((50, 17))
    reward = predict_reward(obs)
