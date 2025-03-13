import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["DISABLE_CODESIGN_WARNING"] = "1"
import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.ekf_subspace import SubspaceNeuralBandit
from bnn_pref.alg.ekf_trainer import bandit_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.ekf_env import EKFEnvironment
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_logpdf_nn
from bnn_pref.utils.plotting import plot_reward_heatmap
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    task_kw = cfg["task"]

    # * generate true params + preference data
    output = dataset_creators[task_kw["ds_type"]](key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    Q, _, T, D = train_data.queries_Q2TD.shape

    print(
        f"Seed: {seed}\n"
        f"N={data_kw['n_demos']}, Q={Q} (warm_obs={ekf_kw['warm_obs']}), T={T}, D={D}\n"
        f"EKF: rnd_proj={ekf_kw['cls']['rnd_proj']}, n_iterates={ekf_kw['cls']['n_iterates']}, warm_burns={ekf_kw['cls']['warm_burns']}"
    )
    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = EKFEnvironment(
        key1,
        X=train_data.queries_Q2TD,
        Y=jax.nn.one_hot(train_data.responses_Q1.squeeze(), num_classes=2),
    )

    rewards_info, bel_trace, bandit = bandit_pipeline(
        key2,
        SubspaceNeuralBandit,
        env,
        warmup_obs=ekf_kw["warm_obs"],
        bandit_kw=ekf_kw,
    )
    bel = jax.tree_util.tree_map(lambda x: x[-1], bel_trace)  # final belief
    warmup_rewards, rewards_trace, _ = rewards_info

    pref_predictor = jax.vmap(partial(bandit.apply_model, bel.mean))
    reward_predictor = jax.vmap(partial(bandit.predict_reward, bel.mean))
    train_acc = compute_accuracy_nn(pref_predictor, train_data)
    test_acc = compute_accuracy_nn(pref_predictor, test_data)
    test_logpdf = compute_logpdf_nn(pref_predictor, test_data)
    # pref_acc = compute_pref_ranking_acc(reward_predictor, test_data)
    print(f"Param Count: {bandit.full_params_count} -> {bandit.subspace_params_count}")
    print(f"Train acc: {train_acc:.2%}")
    print(f"Test acc:  {test_acc:.2%}")
    print(f"Test avg_ll: {test_logpdf:.2f}")
    # print(f"{pref_acc=:.2%}")

    if D == 2:
        train_trajs = output["train_trajs"]
        train_traj_obs = train_trajs["observations"]
        mins, maxs = (train_traj_obs.min(axis=(0, 1)), train_traj_obs.max(axis=(0, 1)))
        feature_bounds = (
            (mins[0].item(), maxs[0].item()),
            (mins[1].item(), maxs[1].item()),
        )
        print(f"Feature bounds: {feature_bounds}")

        nrows, ncols = 1, 2
        fig = plt.figure(figsize=(12, 4))

        ax = fig.add_subplot(nrows, ncols, 1)
        all_obs = train_traj_obs.reshape(-1, 2)
        all_starts = train_traj_obs[:, 0, :]
        all_ends = train_traj_obs[:, -1, :]
        xx, yy = jnp.where(train_trajs["rewards"] == 0)
        all_goals = train_traj_obs[xx, yy, :]

        ax.scatter(all_obs[:, 0], all_obs[:, 1], s=1)
        ax.scatter(all_starts[:, 0], all_starts[:, 1], c="yellow", s=3, label="start")
        ax.scatter(all_ends[:, 0], all_ends[:, 1], c="orange", s=3, label="end")
        ax.scatter(all_goals[:, 0], all_goals[:, 1], c="red", s=3, label="goal")
        ax.set_title("Train Demos")

        learn_reward_plotkw = {
            "reward_fn": jax.vmap(reward_predictor),
            "bounds": feature_bounds,
        }
        title = "Learned Reward"
        ax = fig.add_subplot(nrows, ncols, 2)
        plot_reward_heatmap(ax, **learn_reward_plotkw, title=title)

        plt.show()


if __name__ == "__main__":
    main()
