import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="train")
def app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    app()
