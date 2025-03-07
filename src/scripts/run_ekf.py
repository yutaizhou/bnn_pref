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

from bnn_pref.alg.bandit_env import BanditEnvironment
from bnn_pref.alg.ekf_subspace import SubspaceNeuralBandit
from bnn_pref.alg.train_utils import bandit_pipeline, summarize_results
from bnn_pref.data.synthetic import make_synthetic_data
from bnn_pref.utils.metrics import compute_accuracy_nn, compute_pref_ranking_acc
from bnn_pref.utils.plotting import plot_reward_heatmap
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg):
    seed = get_random_seed() if cfg["seed"] == -1 else cfg["seed"]
    key = jr.key(seed)
    # check RLHF paper
    data_kw = cfg["data"]
    ekf_kw = cfg["ekf"]
    # combine these into one multiline print statement
    print(
        f"Seed: {seed}\n"
        f"N={data_kw['n_demos']}, Q={data_kw['n_queries']} (warm_obs={ekf_kw['warm_obs']}), T={data_kw['length']}, D={data_kw['n_feats']}\n"
        f"EKF: rnd_proj={ekf_kw['cls']['rnd_proj']}, warm_epochs={ekf_kw['cls']['warm_epochs']}, warm_burns={ekf_kw['cls']['warm_burns']}"
    )

    # * generate true params + preference data
    output = make_synthetic_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    true_param_D, true_reward_fn = output["true_param"], output["true_reward_fn"]
    feature_bounds = (train_data.queries_Q2D.min(), train_data.queries_Q2D.max())

    # * build + run bandit alg
    key, key1, key2 = jr.split(key, 3)
    env = BanditEnvironment(
        key1,
        X=train_data.queries_Q2D,
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

    key, key1 = jr.split(key, 2)
    pref_predictor = jax.vmap(partial(bandit.apply_model, bel.mean))
    reward_predictor = jax.vmap(partial(bandit.predict_reward, bel.mean))
    train_acc = compute_accuracy_nn(pref_predictor, train_data)
    test_acc = compute_accuracy_nn(pref_predictor, test_data)
    # pref_acc = compute_pref_ranking_acc(reward_predictor, test_data)
    print(f"Param Count: {bandit.full_params_count} -> {bandit.subspace_params_count}")
    print(f"Train acc: {train_acc:.2%}")
    print(f"Test acc:  {test_acc:.2%}")
    # print(f"{pref_acc=:.2%}")

    if data_kw["n_feats"] == 2 and data_kw["length"] == 1:
        # fig, axs = plt.subplots(1, 3, figsize=(12, 5))
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
