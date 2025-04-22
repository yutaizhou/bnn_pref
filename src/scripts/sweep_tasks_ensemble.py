import os
from collections import defaultdict

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from datetime import datetime

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.ensemble import DeepEnsemble
from bnn_pref.alg.trainer import alg_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


def run_ensemble(key, cfg, data_dict, env):
    data_cfg = cfg["data"]
    alg_cfg = cfg["sgd"]
    test_trajs_obs = data_dict["test_trajs"]["observations"]
    test_prefs = data_dict["test_prefs"]

    # * build + run ensemble alg
    key, key_pipe = jr.split(key, 2)
    ts_trace, bandit = alg_pipeline(key_pipe, DeepEnsemble, env, alg_cfg, data_cfg)

    # * compute metrics
    def eval_bel(_, ts: TrainState):
        prob_Q2 = bandit.compute_predictive(ts, test_trajs_obs, test_prefs.queries_Q2)
        pred_Q = prob_Q2.argmax(axis=1)

        test_acc = jnp.mean(pred_Q == test_prefs.responses_Q1.squeeze())
        prob_Q1 = jnp.take_along_axis(prob_Q2, test_prefs.responses_Q1, axis=1)
        test_logpdf = jnp.log(prob_Q1).mean()

        # all arrays of (1 + nq_updates, )
        result = {
            "test_logpdf": test_logpdf,
            "test_acc": test_acc,
        }
        return (), result

    *_, al_results = jax.lax.scan(eval_bel, init=(), xs=ts_trace)

    results = {
        **al_results,  # (n_seeds, nq_update)
        "param_count": bandit.param_count,
        "ensemble_param_count": bandit.ensemble_param_count,
    }
    return results


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)

    tasks = [
        "reacher",
        "lunar",
        "cheetah",
        "acrobot",
        "ball",
        "cartpoleSwing",
        "cheetahDMC",
        "hopperHop",
        "pendulum",
        "reacherEasy",
        "reacherHard",
        "walkerWalk",
        "ogbench",
    ]
    stats = defaultdict(lambda: defaultdict(dict))

    data_cfg = cfg["data"]
    alg_cfg = cfg["sgd"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    batch_size = alg_cfg["bs"]
    niters = alg_cfg["niters"]
    print(
        f"Seed: {seed} x {cfg['seeds']}\n"
        f"Data:\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"Ensemble:\n"
        f"  n_models={alg_cfg['M']}\n"
        f"  init: bs={batch_size}, niters={niters}\n"
    )

    for task in tasks:
        print(f"{task}: ")
        key, key_data, *key_seeds = jr.split(key, 2 + cfg["seeds"])
        data_dict = dataset_creators[cfg["task"]["ds_type"]](key_data, cfg)
        train_prefs = data_dict["train_prefs"]
        env = PreferenceEnv(
            X=train_prefs.queries_Q2TD,
            Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
        )
        for is_al in [False, True]:
            # * update cfg
            new_cfg = hydra.compose("config", overrides=[f"task={task}"])
            cfg["task"].update(new_cfg["task"])
            cfg["sgd"]["active"] = is_al

            # * run
            seeds = jnp.array(key_seeds)
            run_vmap = jax.vmap(run_ensemble, in_axes=(0, None, None, None))
            start_time = datetime.now()
            res_m, metadata_m = run_vmap(seeds, cfg, data_dict, env)
            duration = (datetime.now() - start_time).total_seconds()

            res = {
                "task": task,
                "active": is_al,
                # * logpdf
                "test_logpdf_all": res_m["test_logpdf"],
                "test_logpdf": MeanStd(res_m["test_logpdf"][:, -1]),
                # * acc
                "test_acc_all": res_m["test_acc"],
                "test_acc": MeanStd(res_m["test_acc"][:, -1]),
            }

            stats[task][is_al] = res

            print(
                f"  active={str(is_al):5}, "
                f"acc: {res['test_acc'].mean:.2%} ± {res['test_acc'].std:.2%}, "
                f"logpdf: {res['test_logpdf'].mean:.2f} ± {res['test_logpdf'].std:.2f}, "
                # f"({metadata_m['full_param_count'][0]:,d} -> {metadata_m['subspace_param_count'][0]:,d}) "
                f"({duration:.1f}s)"
            )

    # * plot logpdf learning curve
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()

    for i, task in enumerate(tasks):
        ax = axs[i]
        y_min = min(stat["test_logpdf_all"].min() for stat in stats[task].values())
        ax.set_ylim(y_min, 0)
        for is_al in [False, True]:
            values = stats[task][is_al]["test_logpdf_all"]  # (n_seeds, nq_update)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label="Active" if is_al else "Random")
            ax.fill_between(
                jnp.arange(values.shape[1]), mean - std, mean + std, alpha=0.2
            )
        ax.set_title(f"{task}")

    dummy_lines = [plt.plot([], [], label=label)[0] for label in ["Random", "Active"]]
    fig.legend(dummy_lines, ["Random", "Active"], loc="center right")
    fig.suptitle("log PDF vs. num queries ")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    plt.show()

    # * plot acc learning curve
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  # 13 tasks total
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0, 1)
        for is_al in [False, True]:
            values = stats[task][is_al]["test_acc_all"]  # (n_seeds, nq_update)
            mean, std = values.mean(0), values.std(0)
            ax.plot(mean, label="Active" if is_al else "Random")
            ax.fill_between(
                jnp.arange(values.shape[1]), mean - std, mean + std, alpha=0.2
            )
        ax.set_title(f"{task}")

    dummy_lines = [plt.plot([], [], label=label)[0] for label in ["Random", "Active"]]
    fig.legend(dummy_lines, ["Random", "Active"], loc="center right")
    fig.suptitle("Accuracy vs. num queries")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    plt.show()


if __name__ == "__main__":
    main()
