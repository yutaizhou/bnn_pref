from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: int(x * y))
OmegaConf.register_new_resolver("subtract", lambda x, y: int(x - y))
