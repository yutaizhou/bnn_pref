def print_ekf_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]  # only synthetic task has T and D in task_kw
    ekf_kw = cfg["ekf"]
    ekf_cls_cfg = ekf_kw["cls"]

    n_demos = data_kw["n_demos"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]
    length = length if length is not None else task_kw["length"]

    nq_train, nq_test = data_kw["nq_train"], data_kw["nq_test"]
    nq_init = ekf_kw["nq_init"]
    nq_update = ekf_kw["nq_update"]
    n_updates = (nq_train - nq_init) if nq_update == -1 else nq_update

    niters = ekf_cls_cfg["niters"]
    batch_size = ekf_cls_cfg["batch_size"]
    warm_burns = ekf_cls_cfg["warm_burns"]
    thinning = ekf_cls_cfg["thinning"]
    sub_dim = ekf_cls_cfg["sub_dim"]
    rnd_proj = ekf_cls_cfg["rnd_proj"]
    n_eff_iterates = (niters - warm_burns) // thinning

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  n_demos={n_demos}, nq_train={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"  Samples for init/update = {nq_init}/{n_updates}\n"
        f"EKF:\n"
        f"  active={ekf_kw['active']}\n"
        f"  init: bs={batch_size}, niters={niters}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
    )


def print_mcmc_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    mcmc_kw = cfg["mcmc"]

    length = length if length is not None else task_kw["length"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]
    nq_train, nq_test = data_kw["nq_train"], data_kw["nq_test"]

    n_samples = mcmc_kw["n_samples"]
    burn_in = mcmc_kw["burn_in"]
    thinning = mcmc_kw["thinning"]
    normalize = mcmc_kw["normalize"]

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={data_kw['n_demos']}, Q={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"MCMC:\n"
        f"  n_samples={n_samples}, burn_in={burn_in}, thinning={thinning}, normalize={normalize}"
    )


def print_sgd_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    ekf_kw = cfg["ekf"]

    nq_train, nq_test = data_kw["nq_train"], data_kw["nq_test"]
    length = length if length is not None else task_kw["length"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]

    niters = ekf_kw["cls"]["niters"]
    batch_size = ekf_kw["cls"]["batch_size"]
    lr = ekf_kw["learning_rate"]

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={data_kw['n_demos']}, Q={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"SGD:\n"
        f"  niters={niters}, batch_size={batch_size}, lr={lr}"
    )
