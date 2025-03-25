def print_ekf_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]  # only synthetic task has T and D in task_kw
    ekf_kw = cfg["ekf"]
    ekf_cls_cfg = ekf_kw["cls"]

    n_demos, n_queries = data_kw["n_demos"], data_kw["n_queries"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]
    length = length if length is not None else task_kw["length"]

    warm_obs = ekf_kw["warm_obs"]
    n_steps = ekf_kw["n_steps"]
    Q_train = data_kw["n_queries_train"]
    Q_test = data_kw["n_queries_test"]
    # Q_train = int(n_queries * data_kw["train_frac"])
    # Q_test = n_queries - Q_train
    n_updates = (Q_train - warm_obs) if n_steps == -1 else n_steps

    n_iterates = ekf_cls_cfg["n_iterates"]
    batch_size = ekf_cls_cfg["batch_size"]
    warm_burns = ekf_cls_cfg["warm_burns"]
    thinning = ekf_cls_cfg["thinning"]
    sub_dim = ekf_cls_cfg["sub_dim"]
    rnd_proj = ekf_cls_cfg["rnd_proj"]
    n_eff_iterates = (n_iterates - warm_burns) // thinning

    if task_kw["ds_type"] == "synthetic":
        # todo fix this fhat thing
        task_str = f"{task_kw['ds_type']}: f={task_kw['f']}, fhat={task_kw['fhat']} (fhat ignored for ekf runs)"
    else:
        task_str = f"{task_kw['ds_type']}: {task_kw['name']}"

    print(
        f"Seed: {seed}\n"
        f"Data:\n"
        f"  {task_str}\n"
        f"  N={n_demos}, Q={n_queries}, T={length}, D={n_feats} (Q Train/Test = {Q_train}/{Q_test})\n"
        f"  Samples for init/update = {warm_obs}/{n_updates}\n"
        f"EKF:\n"
        f"  init: bs={batch_size}, n_iterates={n_iterates}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
    )


def print_mcmc_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    mcmc_kw = cfg["mcmc"]

    n_queries = data_kw["n_queries"]
    length = length if length is not None else task_kw["length"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]

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
        f"  N={data_kw['n_demos']}, Q={n_queries}, T={length}, D={n_feats}\n"
        f"MCMC:\n"
        f"  n_samples={n_samples}, burn_in={burn_in}, thinning={thinning}, normalize={normalize}"
    )


def print_sgd_cfg(seed, cfg, length=None, n_feats=None):
    data_kw = cfg["data"]
    task_kw = cfg["task"]
    ekf_kw = cfg["ekf"]

    n_queries = data_kw["n_queries"]
    length = length if length is not None else task_kw["length"]
    n_feats = n_feats if n_feats is not None else task_kw["n_feats"]

    n_iterates = ekf_kw["cls"]["n_iterates"]
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
        f"  N={data_kw['n_demos']}, Q={n_queries}, T={length}, D={n_feats}\n"
        f"SGD:\n"
        f"  n_iterates={n_iterates}, batch_size={batch_size}, lr={lr}"
    )
