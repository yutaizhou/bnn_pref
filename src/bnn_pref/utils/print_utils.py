def get_param_count_msg(cfg, alg: str, res_m: dict) -> str:
    if alg == "ekf":
        ekf_M = cfg["ekf"]["M"]
        param_count = res_m["param_count"][0].item()
        subspace_param_count = res_m["subspace_param_count"][0].item()
        return f"({param_count:,d} -> {subspace_param_count:,d})"
    elif alg == "sgd":
        sgd_M = cfg["sgd"]["M"]
        param_count = res_m["param_count"][0].item()
        ensemble_param_count = res_m["ensemble_param_count"][0].item()
        return f"({param_count:,d} x {sgd_M:d} -> {ensemble_param_count:,d})"
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def print_ekf_cfg(seed, cfg, length=None, n_feats=None):
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]  # only synthetic task has T and D in task_cfg
    alg_cfg = cfg["ekf"]
    ekf_cls_cfg = alg_cfg["cls"]

    n_demos = data_cfg["n_demos"]
    n_feats = n_feats if n_feats is not None else task_cfg["n_feats"]
    length = length if length is not None else task_cfg["length"]

    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    assert nsteps > 0, "nsteps must be positive"

    niters = ekf_cls_cfg["niters"]
    batch_size = ekf_cls_cfg["batch_size"]
    warm_burns = ekf_cls_cfg["warm_burns"]
    thinning = ekf_cls_cfg["thinning"]
    sub_dim = ekf_cls_cfg["sub_dim"]
    rnd_proj = ekf_cls_cfg["rnd_proj"]
    n_eff_iterates = (niters - warm_burns) // thinning

    if task_cfg["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_cfg['ds_type']}: f={task_cfg['f']}, fhat={task_cfg['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_cfg['ds_type']}: {task_cfg['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  n_demos={n_demos}, nq_train={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"  Samples for init/update = {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  active={alg_cfg['active']}\n"
        f"  init: bs={batch_size}, niters={niters}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
    )


def print_ensemble_cfg(seed, cfg, length=None, n_feats=None):
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    alg_cfg = cfg["sgd"]

    n_demos = data_cfg["n_demos"]
    length = length if length is not None else task_cfg["length"]
    n_feats = n_feats if n_feats is not None else task_cfg["n_feats"]

    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]
    assert nsteps > 0, "nsteps must be positive"

    if task_cfg["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_cfg['ds_type']}: f={task_cfg['f']}, fhat={task_cfg['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_cfg['ds_type']}: {task_cfg['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={n_demos}, Q={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"  Samples for init/update = {nq_init}/{nsteps}\n"
        f"Ensemble:\n"
        f"  active={alg_cfg['active']}\n"
        f"  n_models={alg_cfg['M']}\n"
    )


def print_mcmc_cfg(seed, cfg, length=None, n_feats=None):
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    mcmc_cfg = cfg["mcmc"]

    length = length if length is not None else task_cfg["length"]
    n_feats = n_feats if n_feats is not None else task_cfg["n_feats"]
    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]

    n_samples = mcmc_cfg["n_samples"]
    burn_in = mcmc_cfg["burn_in"]
    thinning = mcmc_cfg["thinning"]
    normalize = mcmc_cfg["normalize"]

    if task_cfg["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_cfg['ds_type']}: f={task_cfg['f']}, fhat={task_cfg['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_cfg['ds_type']}: {task_cfg['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={data_cfg['n_demos']}, Q={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"MCMC:\n"
        f"  n_samples={n_samples}, burn_in={burn_in}, thinning={thinning}, normalize={normalize}"
    )


def print_sgd_cfg(seed, cfg, length=None, n_feats=None):
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    ekf_cfg = cfg["ekf"]

    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    length = length if length is not None else task_cfg["length"]
    n_feats = n_feats if n_feats is not None else task_cfg["n_feats"]

    niters = ekf_cfg["cls"]["niters"]
    batch_size = ekf_cfg["cls"]["batch_size"]
    lr = ekf_cfg["learning_rate"]

    if task_cfg["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_cfg['ds_type']}: f={task_cfg['f']}, fhat={task_cfg['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_cfg['ds_type']}: {task_cfg['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={data_cfg['n_demos']}, Q={nq_train}, T={length}, D={n_feats} (Q Test = {nq_test})\n"
        f"SGD:\n"
        f"  niters={niters}, batch_size={batch_size}, lr={lr}"
    )
