from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bnn_pref.data.pref_utils import QueryIndexAndResponses, create_pref_data
from bnn_pref.data.traj_utils import (
    _segment_traj_masked,
    normalize,
    split_subsample_rank_ds,
)
from bnn_pref.utils.type import ArrayDict

TRAJ_LEN = 100
PROPRIO_DIM = 7


def load_soar_task_trajs(
    pref_proprio_dir: Path,
    scene_slug: str,
    task_slug: str,
) -> ArrayDict:
    """
    Load one (scene, task) from processed soar_pref_proprio/.

    Returns failures first, then successes, so returns are ascending.
    """
    task_dir = pref_proprio_dir / scene_slug / task_slug
    fail_fp = task_dir / "failure" / "proprio.npy"
    succ_fp = task_dir / "success" / "proprio.npy"
    if not (fail_fp.exists() and succ_fp.exists()):
        raise FileNotFoundError(
            f"Missing success/failure proprio under {task_dir}"
        )

    fail_NTD = np.load(fail_fp)  # (Nf, T, 7)
    succ_NTD = np.load(succ_fp)  # (Ns, T, 7)
    assert fail_NTD.shape[1:] == (TRAJ_LEN, PROPRIO_DIM)
    assert succ_NTD.shape[1:] == (TRAJ_LEN, PROPRIO_DIM)

    observations_NTD = jnp.concatenate(
        [jnp.asarray(fail_NTD), jnp.asarray(succ_NTD)],
        axis=0,
    )  # (N, T, 7)
    Nf, Ns = fail_NTD.shape[0], succ_NTD.shape[0]
    returns_N = jnp.concatenate(
        [jnp.zeros(Nf), jnp.ones(Ns)],
        axis=0,
    )  # (N,)
    masks_NT = jnp.ones((Nf + Ns, TRAJ_LEN), dtype=jnp.bool_)

    return {
        "observations": observations_NTD,
        "returns": returns_N,
        "masks": masks_NT,
    }


def segment_soar_arraydict(trajs: ArrayDict, sz: int) -> ArrayDict:
    """
    Segment SOAR proprio trajs; inherit parent traj binary return (0/1).

    Unlike segment_arraydict_masked, does not need per-step rewards.
    """
    assert sz > 0, f"segment size {sz=} must be positive"
    obs_NTD = trajs["observations"]  # (N, T, D)
    masks_NT = trajs["masks"]  # (N, T)
    returns_N = trajs["returns"]  # (N,)
    N = returns_N.shape[0]

    seg_obs_list = []
    seg_ret_list = []
    for i in range(N):
        seg_i = _segment_traj_masked(
            traj=obs_NTD[i : i + 1],
            mask=masks_NT[i : i + 1],
            segment_size=sz,
        )  # (S_i, sz, D)
        if seg_i.shape[0] == 0:
            continue
        seg_obs_list.append(seg_i)
        seg_ret_list.append(jnp.full((seg_i.shape[0],), returns_N[i]))

    observations_STD = jnp.concatenate(seg_obs_list, axis=0)  # (S, sz, D)
    returns_S = jnp.concatenate(seg_ret_list, axis=0)  # (S,)
    masks_ST = jnp.ones((observations_STD.shape[0], sz), dtype=jnp.bool_)

    return {
        "observations": observations_STD,
        "returns": returns_S,
        "masks": masks_ST,
    }


def make_soar_data(key, cfg) -> ArrayDict:
    task_cfg = cfg["task"]
    data_cfg = cfg["data"]

    pref_proprio_dir = Path(task_cfg["pref_proprio_dir"])
    scene_slug = task_cfg["scene_slug"]
    task_slug = task_cfg["task_slug"]
    demo_train_frac = data_cfg["demo_train_frac"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    sz = data_cfg["segment_size"]

    #* load single-task trajectories (failure=0, success=1)
    trajs = load_soar_task_trajs(
        pref_proprio_dir=pref_proprio_dir,
        scene_slug=scene_slug,
        task_slug=task_slug,
    )

    #* normalize proprio across all trajs + timesteps
    trajs.update(
        {"observations": normalize(trajs["observations"], axis=(0, 1))}
    )

    #* optionally segment whole trajs (parent return inherited by each segment)
    if sz != -1:
        trajs = segment_soar_arraydict(trajs=trajs, sz=sz)

    #* split into train/test, rank by return (ascending)
    key, key_split = jr.split(key, 2)
    train_trajs, test_trajs = split_subsample_rank_ds(
        key=key_split,
        ds=trajs,
        train_frac=demo_train_frac,
        n_segments_train=data_cfg["n_segments_train"],
        n_segments_test=data_cfg["n_segments_test"],
    )

    #* preference queries: higher return preferred (success > failure)
    key, key_train, key_test = jr.split(key, 3)
    train_prefs: QueryIndexAndResponses = create_pref_data(
        key=key_train,
        ranked_returns=train_trajs["returns"],
        n_queries=nq_train,
        noisy_label=data_cfg["noisy_label"],
        bt_beta=data_cfg["bt_beta"],
        cross_outcome_only=data_cfg.get("cross_outcome_only", True),
    )
    test_prefs: QueryIndexAndResponses = create_pref_data(
        key=key_test,
        ranked_returns=test_trajs["returns"],
        n_queries=nq_test,
        noisy_label=False,
        bt_beta=data_cfg["bt_beta"],
        cross_outcome_only=data_cfg.get("cross_outcome_only", True),
    )

    return {
        "train_trajs": train_trajs,
        "train_prefs": train_prefs,
        "test_trajs": test_trajs,
        "test_prefs": test_prefs,
    }
