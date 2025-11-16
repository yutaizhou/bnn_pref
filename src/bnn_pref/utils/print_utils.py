def get_param_count_msg(cfg, alg: str, res_m: dict) -> str:
    def get_item(x):
        if hasattr(x, "ndim"):  # np or jnp array
            return x[0].item()
        else:
            return x

    if alg == "ekf":
        param_count = get_item(res_m["param_count"])
        subspace_param_count = get_item(res_m["subspace_param_count"])
        return f"({param_count:,d} -> {subspace_param_count:,d})"
    elif alg == "sgd":
        sgd_M = cfg["sgd"]["M"]
        param_count = get_item(res_m["param_count"])
        ensemble_param_count = get_item(res_m["ensemble_param_count"])
        return f"({param_count:,d} x {sgd_M:d} -> {ensemble_param_count:,d})"
    elif alg == "do":
        param_count = get_item(res_m["param_count"])
        return f"({param_count:,d})"
    elif alg == "llmcmc":
        param_count = get_item(res_m["param_count"])
        last_layer_param_count = get_item(res_m["last_layer_param_count"])
        return f"({param_count:,d} -> {last_layer_param_count:,d})"
    elif alg == "laplace":
        param_count = get_item(res_m["param_count"])
        return f"({param_count:,d})"
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def get_run_cfg_msg(seed, cfg):
    data_cfg = cfg["data"]
    ekf_cfg = cfg["ekf"]
    sgd_cfg = cfg["sgd"]
    do_cfg = cfg["do"]
    llmcmc_cfg = cfg["llmcmc"]
    laplace_cfg = cfg["laplace"]

    nq_train, nq_test = data_cfg["nq_train"], data_cfg["nq_test"]
    nq_init, nsteps = data_cfg["nq_init"], data_cfg["nsteps"]

    n_eff_iterates = (ekf_cfg["niters_init"] - ekf_cfg["warm_burns"]) // ekf_cfg[
        "thinning"
    ]

    print(
        f"Run:\n"
        f"  Seed: {seed} x {cfg['seeds']} (seed_vmap={cfg['seed_vmap']})\n"
        f"  Sanity: {cfg['sanity']} ({cfg['sanity_frac']} real frac)\n"
        f"  Network: {cfg['network']['hidden_sizes']}\n"
        f"Data:\n"
        f"  prune: {data_cfg['n_bins']} bins, {data_cfg['max_count_per_bin']} max_count_per_bin, {data_cfg['tokeep']} tokeep\n"
        f"  noisy_label: {data_cfg['noisy_label']} (beta={data_cfg['bt_beta']})\n"
        f"  Train/Test: {nq_train}/{nq_test}\n"
        f"  Init/Update: {nq_init}/{nsteps}\n"
        f"EKF:\n"
        f"  M={ekf_cfg['M']}, use_vmap={ekf_cfg['use_vmap']}\n"
        f"  prior / dynamics / obs noise: {ekf_cfg['prior_noise']} / {ekf_cfg['dynamics_noise']} / {ekf_cfg['obs_noise']}\n"
        f"  init: bs={ekf_cfg['bs']}, niters={ekf_cfg['niters_init']}[{ekf_cfg['warm_burns']}::{ekf_cfg['thinning']}] ({n_eff_iterates} eff), sub_dim={ekf_cfg['sub_dim']}, rnd_proj={ekf_cfg['rnd_proj']}\n"
        f"Ensemble:\n"
        f"  M={sgd_cfg['M']}, use_vmap={sgd_cfg['use_vmap']}\n"
        f"  init: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters_init']}\n"
        f"  update: bs={sgd_cfg['bs']}, niters={sgd_cfg['niters_update']}\n"
        f"Dropout:\n"
        f"  M={do_cfg['M']}, use_vmap={do_cfg['use_vmap']}\n"
        f"  init: bs={do_cfg['bs']}, niters={do_cfg['niters_init']}\n"
        f"  update: bs={do_cfg['bs']}, niters={do_cfg['niters_update']}\n"
        f"Last-Layer MCMC:\n"
        f"  M={llmcmc_cfg['M']}, use_vmap={llmcmc_cfg['use_vmap']}\n"
        f"  warmups={llmcmc_cfg['mcmc_warmups_init']} -> {llmcmc_cfg['mcmc_warmups_update']}, steps={llmcmc_cfg['mcmc_steps']}\n"
        f"Laplace:\n"
        f"  M={laplace_cfg['M']}, use_vmap={laplace_cfg['use_vmap']}\n"
        f"  init: bs={laplace_cfg['bs']}, niters={laplace_cfg['niters_init']}\n"
        f"  update: bs={laplace_cfg['bs']}, niters={laplace_cfg['niters_update']}\n"
        f"  prior_prec={laplace_cfg['prior_prec']}\n"
    )


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

    niters_init = ekf_cls_cfg["niters_init"]
    batch_size = ekf_cls_cfg["batch_size"]
    warm_burns = ekf_cls_cfg["warm_burns"]
    thinning = ekf_cls_cfg["thinning"]
    sub_dim = ekf_cls_cfg["sub_dim"]
    rnd_proj = ekf_cls_cfg["rnd_proj"]
    n_eff_iterates = (niters_init - warm_burns) // thinning

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
        f"  init: bs={batch_size}, niters={niters_init}[{warm_burns}::{thinning}] ({n_eff_iterates} eff), {sub_dim=}, {rnd_proj=}\n"
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
