from omegaconf import OmegaConf

OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
