import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hoechstgan.util.logging import ModelLogger


@hydra.main(config_path="conf", config_name="train")
def app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print(cfg.name)
    with ModelLogger(cfg) as logger:
        for i in range(10):
            wandb.log({"i": i})


if __name__ == "__main__":
    app()
