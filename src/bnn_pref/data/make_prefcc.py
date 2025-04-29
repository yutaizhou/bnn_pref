import jax
import jax.numpy as jnp
import jax.random as jr
import torch
from einops import rearrange
from tensordict import TensorDict

from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.data.traj_utils import normalize, rebalance, segment_traj, split_dataset
from bnn_pref.utils.type import ArrayDict


def segment_arraydict(trajs: ArrayDict, sz: int) -> ArrayDict:
    """
    segment each traj in traj dict into chunks of size sz
    """
    assert sz > 0, f"segment size {sz=} must be positive"
    # print(f"  Whole trajs: {trajs['observations'].shape}")
    seg_fn = lambda x: segment_traj(x, sz)
    trajs["observations"] = seg_fn(trajs["observations"])
    trajs["rewards"] = rearrange(seg_fn(trajs["rewards"]), "S sz 1-> S sz")
    trajs["returns"] = trajs["rewards"].sum(1)
    # print(f"  Segments:    {trajs['observations'].shape}")
    return trajs


def make_prefcc_data(key, cfg) -> ArrayDict:
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]
    path = task_cfg["tensordict_path"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    sz = data_cfg["segment_size"]

    # * load trajectory data, sort by return
    td = torch.load(path, weights_only=False)
    if sz > td.shape[1]:
        sz = td.shape[1]

    trajs = process_prefcc_data(td)

    # * optionally normalize observations
    if task_cfg["name"] not in [
        "Reacher-v4",
        "reacher-easy-v0",
        "reacher-hard-v0",
    ]:
        trajs.update({"observations": normalize(trajs["observations"], axis=(0, 1))})

    # * optional pruning
    key, key_rebalance = jr.split(key, 2)
    trajs = rebalance(
        key_rebalance,
        task_cfg["name"],
        ds=trajs,
        n_bins=data_cfg["n_bins"],
        max_count_per_bin=data_cfg["max_count_per_bin"],
        tokeep=data_cfg["tokeep"],
    )

    # * optionally segment trajectories: (N, T, ...) -> (N * n_chunks, sz, ...)
    # * can be done before or after splitting into train/test
    if sz != -1:
        trajs = segment_arraydict(trajs, sz)

    # * split into train/test
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_dataset(key_split, trajs, demo_train_frac)

    # if sz != -1:
    #     train_trajs = segment_arraydict(train_trajs, sz)
    #     test_trajs = segment_arraydict(test_trajs, sz)

    # * turn train/test trajs into preference data
    key, key_train, key_test = jr.split(key, 3)
    train_prefs: QueryIndexAndResponses = create_pref_data(
        key_train,
        ranked_returns=train_trajs["returns"],
        n_queries=nq_train,
        noisy_label=data_cfg["noisy_label"],
        bt_beta=data_cfg["bt_beta"],
    )
    test_prefs: QueryIndexAndResponses = create_pref_data(
        key_test,
        ranked_returns=test_trajs["returns"],
        n_queries=nq_test,
    )

    return {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }


def process_prefcc_data(td: TensorDict, rank: bool = False) -> ArrayDict:
    """
    Tensordict only contains obs, act, rew, and are already sorted by returns.
    """
    ds = {
        "observations": jnp.asarray(td["obs"]),  # (N, T, D)
        # "actions": jnp.asarray(td["actions"]),  # (N, T, A)
        "rewards": jnp.asarray(td["rewards"]),  # (N, T)
    }
    ds["returns"] = ds["rewards"].sum(axis=1)  # (N,)

    # * sort trajectories by return (ascending)
    if rank:
        sorted_idxes = jnp.argsort(ds["returns"])
        ds = jax.tree.map(lambda x: x[sorted_idxes], ds)

    return ds


if __name__ == "__main__":
    from hydra import compose, initialize

    from bnn_pref.utils.utils import get_random_seed

    with initialize(version_base=None, config_path="../../cfg"):
        cfg = compose(config_name="config", overrides=["data=lunar"])

    key = jr.key(get_random_seed())
    output = make_prefcc_data(key, cfg)
    train_data, test_data = output["train_prefs"], output["test_prefs"]
    print(train_data.queries_Q2TD.shape, train_data.responses_Q1.shape)
    print(test_data.queries_Q2TD.shape, test_data.responses_Q1.shape)
    print()
