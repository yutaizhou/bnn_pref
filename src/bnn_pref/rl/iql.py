import os
import warnings
from functools import partial
from typing import Callable, List, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import d4rl
import flax.linen as nn
import gym
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import OmegaConf

from bnn_pref.data.traj_utils import normalize, process_rewards
from bnn_pref.rl.common import (
    AgentState,
    DualQNetwork,
    StateValueFunction,
    TanhGaussianActor,
    Transition,
)
from bnn_pref.rl.rm_util import load_reward_model, relabel_rewards
from bnn_pref.utils.utils import get_random_seed, slurm_auto_scancel

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"  # from unifloral


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


def run_iql(rng, cfg):
    # --- Parse arguments ---
    rl_cfg = cfg["rl"]
    task_cfg = cfg["task"]
    assert rl_cfg["reward"] in ["gt", "pref", "zero"]

    # --- Initialize logger ---
    alg_str = (
        f"{rl_cfg.pref_alg}_al={rl_cfg.pref_is_al}"
        if rl_cfg.reward == "pref"
        else rl_cfg.reward
    )
    task_alg_str = f"{task_cfg.name}_{alg_str}"
    if rl_cfg.use_wandb:
        wandb.init(
            name=task_alg_str,
            config=OmegaConf.to_container(rl_cfg, resolve=True),
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            job_type="offlineRL",
            tags=cfg.wandb.tags,
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

    reward_src = None
    if rl_cfg["reward"] == "gt":
        reward_src = "gt"
    elif rl_cfg["reward"] == "pref":
        rng, rng_reward = jr.split(rng)
        reward_fn, ckpt_fp = load_reward_model(
            key=rng_reward,
            run_dir=rl_cfg["run_dir"],
            task_name=rl_cfg["task_name"],
            alg=rl_cfg["pref_alg"],
            is_al=rl_cfg["pref_is_al"],
        )
        reward_src = ckpt_fp
        obs = normalize(dataset.obs, axis=(0,))  # (N, obsDim)
        rhat = relabel_rewards(reward_fn, obs)  # (N,)
        rhat = process_rewards(rhat, rl_cfg["normalize_reward"], rl_cfg["clip_reward"])
        dataset = dataset._replace(reward=rhat)
    elif rl_cfg["reward"] == "zero":
        reward_src = "zeros"
        rhat = jnp.zeros_like(dataset.reward)
        dataset = dataset._replace(reward=rhat)

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

    if rl_cfg.log:
        print(
            f"{task_cfg.name}: Training {alg_str} policy for {num_evals} iterations..."
        )
        print(f"  Reward source: {reward_src}")
    returns_list = []
    scores_list = []
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            init=(rng, agent_state),
            length=rl_cfg.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jr.split(rng)
        returns = eval_agent(rl_cfg, rng_eval, env, agent_state)  # (n_eval_workers,)
        scores = d4rl.get_normalized_score(task_cfg.name, returns) * 100.0
        returns_list.append(returns)
        scores_list.append(scores)
        # --- Log metrics ---
        step = (eval_idx + 1) * rl_cfg.eval_interval
        if rl_cfg.log:
            print(
                f"Step: {step} / {rl_cfg.n_updates} ({eval_idx + 1:02d}/{num_evals}) | "
                f"Score: {scores.mean():.2f} ± {scores.std():.2f}"
            )
        if rl_cfg.use_wandb:
            log_dict = {
                "eval/return": returns.mean(),
                "eval/return_std": returns.std(),
                "eval/score": scores.mean(),
                "eval/score_std": scores.std(),
                "num_updates": step,
                **{f"train/{k}": loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    info = {}
    if rl_cfg.n_final_eval_episodes > 0:
        final_iters = int(np.ceil(rl_cfg.n_final_eval_episodes / rl_cfg.n_eval_workers))
        if rl_cfg.log:
            print(f"  Evaluating final agent for {final_iters} iterations...")
        _rng = jr.split(rng, final_iters)
        rets = np.concatenate(
            [eval_agent(rl_cfg, _rng, env, agent_state) for _rng in _rng]
        )  # (n_eval_workers * final_iters)
        scores = d4rl.get_normalized_score(task_cfg.name, rets) * 100.0
        returns_list.append(rets)
        scores_list.append(scores)
        agg_fn = lambda x, k: {
            k: x,
            f"{k}_mean": x.mean(),
            f"{k}_std": x.std(),
        }
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")
        if rl_cfg.log:
            print(
                f"{task_cfg.name}: {alg_str}\n"
                f"  {rl_cfg.n_final_eval_episodes} episodes {rl_cfg.n_eval_workers} workers\n"
                f"  final return: {rets.mean():.2f} ± {rets.std():.2f}\n"
                f"  final normalized score: {scores.mean():.2f} ± {scores.std():.2f}\n"
            )

    results = {
        "returns": np.array(returns_list),  # (n_evals_steps, n_eval_workers)
        "scores": np.array(scores_list),  # (n_evals_steps, n_eval_workers)
        "reward_src": reward_src,  # str
        **(info if info else {}),
    }

    # --- Write final returns to file ---
    metric_fp = f"{cfg.paths.output_dir}/stats.npz"
    with open(metric_fp, "wb") as f:
        np.savez_compressed(f, **results)

    if rl_cfg.use_wandb:
        wandb.save(metric_fp)
        wandb.finish()

    return results


@hydra.main(config_name="configOfflineRL", config_path="../../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    run_iql(key, cfg)
    slurm_auto_scancel()  # prevent completed jobs from hanging on slurm


if __name__ == "__main__":
    main()
