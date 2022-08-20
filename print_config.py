import hydra
from omegaconf import DictConfig, OmegaConf

import hoechstgan  # required to register interpolations


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    app()
