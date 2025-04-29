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

from bnn_pref.alg.ekf_subspace import SubspaceEKF
from bnn_pref.alg.trainer import alg_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import compute_acc_nn, compute_acc_nn_bma, compute_logpdf_nn
from bnn_pref.utils.plotting import plot_reward_heatmap
from bnn_pref.utils.print_utils import print_ekf_cfg
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    task_cfg = cfg["task"]

    # * generate true params + preference data
    output = dataset_creators[task_cfg["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    Q, _, T, n_feats = train_prefs.queries_Q2TD.shape
    print_ekf_cfg(seed, cfg, n_feats=n_feats, length=T)

    # * build + run bandit alg
    key, key_pipe, key_bma = jr.split(key, 3)
    env = PreferenceEnv(
        X=train_prefs.queries_Q2TD,
        Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
    )

    bel_trace, bandit = alg_pipeline(key_pipe, SubspaceEKF, env, ekf_cfg, data_cfg)

    # * compute metrics
    sub2full_logits_fn = bandit.sub2full_predict_logits  # (params, N2TD) -> (N2,)

    def eval_bel(_, bel):
        mean, cov, t = bel
        key = jr.fold_in(key_bma, t)
        fn = jax.vmap(partial(sub2full_logits_fn, mean))
        train_logpdf = compute_logpdf_nn(fn, train_prefs)
        test_logpdf = compute_logpdf_nn(fn, test_prefs)
        train_acc = compute_acc_nn(fn, train_prefs)
        test_acc = compute_acc_nn(fn, test_prefs)
        train_acc_bma = compute_acc_nn_bma(key, sub2full_logits_fn, bel, train_prefs)
        test_acc_bma = compute_acc_nn_bma(key, sub2full_logits_fn, bel, test_prefs)

        result = {
            # * logpdf
            "train_logpdf": train_logpdf,
            "test_logpdf": test_logpdf,
            # * acc
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_acc_bma": train_acc_bma,
            "test_acc_bma": test_acc_bma,
        }
        return (), result

    *_, res = jax.lax.scan(eval_bel, init=(), xs=bel_trace)

    print(
        f"Param Count:  {bandit.param_count} -> {bandit.subspace_param_count}\n"
        f"Train acc:    {res['train_acc'][0]:.2%} -> {res['train_acc'][-1]:.2%}\n"
        f"Test acc:     {res['test_acc'][0]:.2%} -> {res['test_acc'][-1]:.2%}\n"
        f"Test acc BMA: {res['test_acc_bma'][-1]:.2%}\n"
        f"Train avg_ll: {res['train_logpdf'][0]:.2f} -> {res['train_logpdf'][-1]:.2f}\n"
        f"Test avg_ll:  {res['test_logpdf'][0]:.2f} -> {res['test_logpdf'][-1]:.2f}\n"
    )

    # * performance viz
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(res["train_acc"], label="train acc")
    ax1.plot(res["test_acc"], label="test acc")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.plot(res["train_logpdf"], label="train logpdf")
    ax2.plot(res["test_logpdf"], label="test logpdf")
    ax2.set_ylim(None, 0)
    ax2.legend()
    # plt.show()

    # * visualization
    if (task_cfg["ds_type"] == "ogbench") and (n_feats == 2):
        # ogbench "pointmaze-medium-navigate-singletask-v0": train traj + reward plot
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

        reward_predictor = jax.vmap(partial(bandit.sub2full_predict_return, bel.mean))
        learn_reward_plotkw = {
            "reward_fn": jax.vmap(reward_predictor),
            "bounds": feature_bounds,
        }
        title = "Learned Reward"
        ax = fig.add_subplot(nrows, ncols, 2)
        plot_reward_heatmap(ax, **learn_reward_plotkw, title=title)

        plt.show()

    if (task_cfg["ds_type"] == "synthetic") and (n_feats == 1) and (T == 1):
        true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Generate x points for plotting
        x = jnp.linspace(feature_bounds[0], feature_bounds[1], 100)
        x_points = x.reshape(-1, 1, 1)  # reshape for single feature
        print(x.shape, x_points.shape)

        # True reward function
        true_utility_fn = partial(true_reward_fn, param_D=true_param_D)
        true_utility_fn = jax.vmap(true_utility_fn)  # vectorize over batch dimension
        true_rewards = true_utility_fn(x_points)

        # Learned reward function
        learned_rewards = reward_predictor(x_points)  # already vmapped

        # Plot true reward
        ax1.plot(x, true_rewards, "b-", label="True Reward")
        ax1.set_title(f"True Reward {true_param_D}")
        ax1.set_xlabel("Feature Value")
        ax1.set_ylabel("Reward")
        ax1.grid(True)
        ax1.legend()

        # Plot learned reward
        ax2.plot(x, learned_rewards, "r-", label="Learned Reward")
        ax2.set_title("Learned Reward")
        ax2.set_xlabel("Feature Value")
        ax2.set_ylabel("Reward")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    if (task_cfg["ds_type"] == "synthetic") and (n_feats == 2) and (T == 1):
        true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
        nrows, ncols = 2, 3
        fig = plt.figure(figsize=(12, 5))

        true_utility_fn = partial(true_reward_fn, param_D=true_param_D)
        true_utility_fn = jax.vmap(jax.vmap(true_utility_fn))
        title = f"True Reward {true_param_D}"
        true_reward_plotkw = {"reward_fn": true_utility_fn, "bounds": feature_bounds}
        ax = fig.add_subplot(nrows, ncols, 1, projection="3d", title=title)
        plot_reward_heatmap(ax, **true_reward_plotkw, plot_3d=True)
        ax = fig.add_subplot(nrows, ncols, 4)
        plot_reward_heatmap(ax, **true_reward_plotkw, plot_3d=False)

        learn_reward_plotkw = {
            "reward_fn": jax.vmap(reward_predictor),
            "bounds": feature_bounds,
        }
        title = "Learned Reward"
        ax = fig.add_subplot(nrows, ncols, 2, projection="3d", title=title)
        plot_reward_heatmap(ax, **learn_reward_plotkw, plot_3d=True)
        ax = fig.add_subplot(nrows, ncols, 5)
        plot_reward_heatmap(ax, **learn_reward_plotkw, plot_3d=False)

        plt.show()


if __name__ == "__main__":
    main()
