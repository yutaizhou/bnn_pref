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
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig

from bnn_pref.alg.ensemble import DeepEnsemble
from bnn_pref.alg.trainer import alg_pipeline
from bnn_pref.data import dataset_creators
from bnn_pref.data.data_env import PreferenceEnv
from bnn_pref.utils.hydra_resolvers import *
from bnn_pref.utils.metrics import compute_acc_ensemble, compute_logpdf_ensemble
from bnn_pref.utils.plotting import plot_reward_heatmap
from bnn_pref.utils.print_utils import print_ensemble_cfg
from bnn_pref.utils.utils import get_random_seed

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
jnp.set_printoptions(precision=2)


@hydra.main(version_base=None, config_name="configPref", config_path="../cfg")
def main(cfg):
    seed = get_random_seed(cfg["seed"])
    key = jr.key(seed)
    data_cfg = cfg["data"]
    sgd_cfg = cfg["sgd"]
    task_cfg = cfg["task"]

    # * generate true params + preference data
    output = dataset_creators[task_cfg["ds_type"]](key, cfg)
    train_prefs, test_prefs = output["train_prefs"], output["test_prefs"]
    Q, _, T, n_feats = train_prefs.queries_Q2TD.shape
    print_ensemble_cfg(seed, cfg, n_feats=n_feats, length=T)

    # * build + run bandit alg
    key, key_pipe, key_bma = jr.split(key, 3)
    env = PreferenceEnv(
        X=train_prefs.queries_Q2TD,
        Y=jax.nn.one_hot(train_prefs.responses_Q1.squeeze(), num_classes=2),
    )

    ts_trace, bandit = alg_pipeline(key_pipe, DeepEnsemble, env, sgd_cfg, data_cfg)

    # * compute metrics
    def eval_bel(carry, ts: TrainState):
        # t: int = ts.step[0]  # (M,)
        # key = jr.fold_in(key_bma, t)
        fn = jax.vmap(ts.apply_fn, in_axes=(0, None), out_axes=1)  # fn(param, input)
        fn = partial(fn, {"params": ts.params})
        train_logpdf = compute_logpdf_ensemble(fn, train_prefs)
        test_logpdf = compute_logpdf_ensemble(fn, test_prefs)

        train_acc = compute_acc_ensemble(fn, train_prefs)
        test_acc = compute_acc_ensemble(fn, test_prefs)
        # train_acc_bma = compute_acc_nn_bma(key, pred_fn, ts, train_prefs)
        # test_acc_bma = compute_acc_nn_bma(key, pred_fn, ts, test_prefs)

        result = {
            # * logpdf
            "train_logpdf": train_logpdf,
            "test_logpdf": test_logpdf,
            # * acc
            "train_acc": train_acc,
            "test_acc": test_acc,
            # "train_acc_bma": train_acc_bma,
            # "test_acc_bma": test_acc_bma,
        }
        return (), result

    *_, res = jax.lax.scan(eval_bel, init=(), xs=ts_trace)

    print(
        f"Param Count:  {bandit.model_params_count} -> {bandit.ensemble_params_count}\n"
        f"Train acc:    {res['train_acc'][0]:.2%} -> {res['train_acc'][-1]:.2%}\n"
        f"Test acc:     {res['test_acc'][0]:.2%} -> {res['test_acc'][-1]:.2%}\n"
        # f"Test acc BMA: {res['test_acc_bma'][-1]:.2%}\n"
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
    plt.show()


if __name__ == "__main__":
    main()
