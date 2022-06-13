import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()
