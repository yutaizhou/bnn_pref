import warnings
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int
from tqdm import tqdm

from bnn_pref.alg import AgentState, alg_classes
from bnn_pref.alg.agent_utils import Agent
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_coverage_rate,
    compute_ece,
    compute_logpdf,
    compute_sharpness,
)
from bnn_pref.utils.network import RewardNet, isfinite_param, isfinite_param_pytree
from bnn_pref.utils.utils import Timer

warnings.filterwarnings("ignore")


def run_alg(key, alg_name: str, cfg, data_dict, env):
    assert alg_name in alg_classes.keys()
    alg_cfg = cfg[alg_name]
    alg_cls = alg_classes[alg_name]
    data_cfg = cfg["data"]
    test_items_NTD = data_dict["test_trajs"]["observations"]
    test_queries_Q2 = data_dict["test_prefs"].queries_Q2
    test_labels_Q1 = data_dict["test_prefs"].responses_Q1
    test_data = (test_items_NTD, test_queries_Q2, test_labels_Q1)

    # * build + train
    key, key_train = jr.split(key, 2)
    eval_results, bel_final, durations, alg_info = alg_pipeline(
        key=key_train,
        alg_name=alg_name,
        alg_cls=alg_cls,
        env=env,
        alg_cfg=alg_cfg,
        data_cfg=data_cfg,
        test_data=test_data,
        coverage_alpha=cfg["coverage_alpha"],
        verbose=cfg["verbose"],
    )

    results = {
        "final_belief": bel_final,
        **eval_results,  # (n_seeds, 1 + nq_update)
        **alg_info,
        **durations,
    }

    return results


def alg_pipeline(
    key,
    alg_name: str,
    alg_cls: Agent,
    env: PreferenceEnv,
    alg_cfg: Dict,
    data_cfg: Dict,
    test_data: Tuple[Float[Array, "N T D"], Int[Array, "Q 2"], Int[Array, "Q"]],
    coverage_alpha: float,
    verbose: bool = False,
) -> Tuple[AgentState, Agent]:
    # * build pool for active learning
    nq_init, n_steps = data_cfg["nq_init"], data_cfg["nsteps"]
    if nq_init < alg_cfg["bs"]:
        alg_cfg["bs"] = 1
        # print(f"WARNING: {nq_init=} < {bs=}, setting alg_cfg.bs = 1")

    pool_idxs = jnp.arange(nq_init, len(env))

    # * build reward model, agent, env
    traj_shape = env.get_traj_shape()  # (T, D) or (T, H, W, C)
    model = RewardNet(
        hidden_sizes=alg_cfg["hidden_sizes"],
        n_splits=alg_cfg["n_splits"],
        dropout_prob=alg_cfg["dropout_prob"],
        encoder_type=alg_cfg["encoder"],
    )
    bandit = alg_cls(
        model,
        traj_shape=traj_shape,
        **alg_cls.get_hydra_config(alg_cfg),
        verbose=verbose,
    )

    def eval_bel(key_eval, bel: AgentState):
        test_items_NTD, test_queries_Q2, test_labels_Q1 = test_data
        # compute posterior predictive
        prob_Q2 = bandit.compute_postpred(
            key_eval, bel, test_items_NTD, test_queries_Q2
        )
        prob_Q2 = jnp.clip(prob_Q2, min=1e-4, max=1 - 1e-4)  # stability

        # compute metrics
        test_acc = compute_accuracy(prob_Q2, test_labels_Q1)
        test_logpdf = compute_logpdf(prob_Q2, test_labels_Q1)
        test_ece = compute_ece(prob_Q2, test_labels_Q1)
        test_brier_score = compute_brier_score(prob_Q2, test_labels_Q1)
        test_coverage_rate = compute_coverage_rate(
            prob_Q2, test_labels_Q1, alpha=coverage_alpha
        )
        test_sharpness = compute_sharpness(prob_Q2)

        # all scalar arrays: concatenated by scan to form array of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
            "test_ece": test_ece,
            "test_brier": test_brier_score,
            "test_coverage": test_coverage_rate,
            "test_sharpness": test_sharpness,
        }
        return result

    def get_query_idx(key_query, bandit, bel, env, pool_idxs, is_active):
        if not is_active:
            return jr.choice(key_query, pool_idxs)
        else:
            return bandit.compute_next_query(key_query, bel, env, pool_idxs)

    timer = Timer()
    # * belief init
    key, key_warm_data, key_bel_init, key_bel_init_eval = jr.split(key, 4)
    warmup_data = env.warmup(key_warm_data, nq_init)
    with timer.context("train"):
        bel = jax.block_until_ready(bandit.init_bel(key_bel_init, warmup_data))
    with timer.context("eval"):
        eval_results = [eval_bel(key_bel_init_eval, bel)]

    # * belief updates
    # for t in tqdm(range(n_steps), desc="Updating belief"):
    active_str = "A" if alg_cfg["active"] else "R"
    pbar = tqdm(
        range(n_steps),
        desc=f"Belief updates {alg_name} ({active_str}) ",
        disable=not verbose,
    )
    for t in pbar:
        key = jr.fold_in(key, t)
        key_idx, key_update, key_eval = jr.split(key, 3)

        # retrieve query, update belief
        with timer.context("train"):
            idx = get_query_idx(key_idx, bandit, bel, env, pool_idxs, alg_cfg["active"])
            query_data = env.get_batched_query(nq_init + idx)  # (1, 2, T, D)
            bel = jax.block_until_ready(bandit.update_bel(key_update, bel, query_data))

        # eval belief
        with timer.context("eval"):
            eval_result = eval_bel(key_eval, bel)
            eval_results.append(eval_result)

        postfix = {"test_logpdf": eval_result["test_logpdf"]}
        pbar.set_postfix(postfix)

    # * aggregate eval results and final belief
    eval_results = jax.tree.map(lambda *xs: jnp.stack(xs), *eval_results)
    bel_final = bel if alg_name in ["ekf", "llmcmc", "laplace"] else bel.ts
    if alg_name == "do":
        bel_final = bel_final.replace(dropout_key=jr.key_data(bel_final.dropout_key))

    times = timer.get_total_times()
    durations = {
        "train_duration": times["train"],
        "eval_duration": times["eval"],
    }
    alg_info = bandit.get_alg_info()

    return eval_results, bel_final, durations, alg_info
