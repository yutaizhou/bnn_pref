from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: int(x * y))
OmegaConf.register_new_resolver("subtract", lambda x, y: int(x - y))


def get_nsteps(nq_train, nq_init, nq_update) -> int:
    return nq_train - nq_init if nq_update == -1 else nq_update


OmegaConf.register_new_resolver("get_nsteps", get_nsteps)
