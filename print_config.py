import hydra
from omegaconf import DictConfig, OmegaConf

from hoechstgan.data.split_dataset import get_dataset_splits


@hydra.main(config_path="conf", config_name="train")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    get_dataset_splits(cfg)


if __name__ == "__main__":
    app()
