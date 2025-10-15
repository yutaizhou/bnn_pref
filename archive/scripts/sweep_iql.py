import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pybullet_envs")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import itertools as it
import logging
import os
import warnings

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
logging.getLogger("absl").setLevel(logging.ERROR)


import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from bnn_pref.rl.iql import run_iql
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import MeanStd
from bnn_pref.utils.print_utils import get_param_count_msg
from bnn_pref.utils.utils import get_random_seed, nested_defaultdict

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configOfflineRL", config_path="../cfg")
def main(cfg):
    rl_cfg = cfg["rl"]
    loaded_cfg = OmegaConf.load(f"{rl_cfg['run_dir']}/.hydra/config.yaml")
    tasks = [
        # # * D4RL
        "cheetahRandom",
        "cheetahMediumReplay",
        "cheetahMediumExpert",
        "hopperRandom",
        "hopperMediumReplay",
        "hopperMediumExpert",
        "walkerRandom",
        "walkerMediumReplay",
        "walkerMediumExpert",
        "penHuman",
        "penExpert",
        "penCloned",
        "kitchenComplete",
        "kitchenPartial",
        "kitchenMixed",
        "mazeUDense",
        "mazeMediumDense",
        "mazeLargeDense",
    ]
    algs = ["ekf", "sgd"]
    is_als = [False, True]

    # algs = ["ekf"]
    # algs = ["sgd"]
    # is_als = [False]

    stats = nested_defaultdict()

    loaded_data_cfg = loaded_cfg["data"]
    loaded_ekf_cfg = loaded_cfg["ekf"]
    loaded_sgd_cfg = loaded_cfg["sgd"]
    nq_train, nq_test = loaded_data_cfg["nq_train"], loaded_data_cfg["nq_test"]
    nq_init, nsteps = loaded_data_cfg["nq_init"], loaded_data_cfg["nsteps"]
    n_eff_iterates = (
        loaded_ekf_cfg["niters"] - loaded_ekf_cfg["warm_burns"]
    ) // loaded_ekf_cfg["thinning"]
    n_evals = rl_cfg["n_updates"] // rl_cfg["eval_interval"]

    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    print(
        f"Run:\n"
        f"  Seed: {seed} x {loaded_cfg['seeds']} (seed_vmap={loaded_cfg['seed_vmap']})\n"
        f"  Sanity: {loaded_cfg['sanity']} ({loaded_cfg['sanity_frac']} real frac)\n"
        f"  Network: {loaded_cfg['network']['hidden_sizes']}\n"
        f"Data:\n"
        f"  prune: {loaded_data_cfg['n_bins']} bins, {loaded_data_cfg['max_count_per_bin']} max_count_per_bin, {loaded_data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {loaded_data_cfg['noisy_label']} (beta={loaded_data_cfg['bt_beta']})\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  M={loaded_ekf_cfg['M']}, use_vmap={loaded_ekf_cfg['use_vmap']}\n"
        f"  prior / dynamics / obs noise: {loaded_ekf_cfg['prior_noise']} / {loaded_ekf_cfg['dynamics_noise']} / {loaded_ekf_cfg['obs_noise']}\n"
        f"  init: bs={loaded_ekf_cfg['bs']}, niters={loaded_ekf_cfg['niters']}[{loaded_ekf_cfg['warm_burns']}::{loaded_ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={loaded_ekf_cfg['sub_dim']}, rnd_proj={loaded_ekf_cfg['rnd_proj']}\n"
        f"Ensemble:\n"
        f"  M={loaded_sgd_cfg['M']}, use_vmap={loaded_sgd_cfg['use_vmap']}\n"
        f"  init: bs={loaded_sgd_cfg['bs']}, niters={loaded_sgd_cfg['niters']}\n"
        f"IQL:\n"
        f"  n_updates={rl_cfg['n_updates']}, eval_interval={rl_cfg['eval_interval']} ({n_evals} evals)\n"
        f"  n_eval_workers={rl_cfg['n_eval_workers']}, n_final_eval_episodes={rl_cfg['n_final_eval_episodes']}\n"
    )

    for task in tasks:
        # * update cfg
        new_cfg = hydra.compose("configOfflineRL", overrides=[f"task={task}"])
        cfg["task"].update(new_cfg["task"])
        key, key_run = jr.split(key, 2)
        for alg, is_al in it.product(algs, is_als):
            cfg["rl"]["reward"] = "pref"
            cfg["rl"]["pref_is_al"] = is_al
            cfg["rl"]["pref_alg"] = alg
            cfg["rl"]["log"] = False

            results = run_iql(key_run, cfg)
            scores = results["scores"]  # (n_evals_steps, n_eval_workers)
            reward_src = results["reward_src"]
            final_score = scores[-1]  # (n_eval_workers,)
            print(
                f"{cfg['task']['name']}: {alg}_al={is_al}\n"
                f"  reward: {reward_src}\n"
                f"  final score: {final_score.mean():.2f} ± {final_score.std():.2f}"
            )

            stats[task][alg][is_al] = results
    jnp.savez(f"{cfg.paths.output_dir}/stats.npz", **stats)

    # fake test curves for plotting
    # for task in tasks:
    #     for alg, is_al in it.product(algs, is_als):
    #         key, key_syn = jr.split(key, 2)
    #         scores = jnp.array(
    #             [
    #                 jr.randint(key_syn, (100,), 0, 100) * scale
    #                 for scale in [0.9, 1.0, 1.1]
    #             ]
    #         ).T
    #         stats[task][alg][is_al] = {
    #             "scores": scores
    #         }  # (n_eval_steps, n_eval_workers)

    # * plot policy evaluation curves
    fig, axs = plt.subplots(3, 4, figsize=(12, 10))
    axs = axs.flatten()

    def get_label(alg: str, is_al: bool) -> str:
        if alg == "ekf":
            return "EKF (Active)" if is_al else "EKF (Random)"
        else:
            return "Ensemble (Active)" if is_al else "Ensemble (Random)"

    def get_style(alg: str, is_al: bool) -> dict:
        color = "blue" if alg == "ekf" else "orange"
        linestyle = "-" if is_al else "--"
        return {"color": color, "linestyle": linestyle, "linewidth": 1}

    for i, task in enumerate(tasks):
        ax = axs[i]
        ax.set_ylim(0, 100)
        # todo: add zero + gt baselines
        for alg, is_al in it.product(algs, is_als):
            stat = stats[task][alg][is_al]
            scores = stat["scores"]  # (n_eval_steps, n_eval_workers)
            mean, std = scores.mean(1), scores.std(1)
            label = get_label(alg, is_al)
            style = get_style(alg, is_al)
            ax.plot(mean, label=label, **style)
            ax.fill_between(
                jnp.arange(len(mean)), mean - std, mean + std, alpha=0.2, **style
            )
        ax.set_title(f"{task}")
    dummy_lines = [
        plt.plot([], [], color="blue", linestyle="--", label="EKF (Random)")[0],
        plt.plot([], [], color="blue", linestyle="-", label="EKF (Active)")[0],
        plt.plot([], [], color="orange", linestyle="--", label="Ensemble (Random)")[0],
        plt.plot([], [], color="orange", linestyle="-", label="Ensemble (Active)")[0],
    ]
    fig.legend(
        dummy_lines,
        ["EKF (Random)", "EKF (Active)", "Ensemble (Random)", "Ensemble (Active)"],
        loc="center right",
    )
    fig.suptitle("IQL Policy Evaluation")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]
    plt.savefig(f"{cfg.paths.output_dir}/policy_eval_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
