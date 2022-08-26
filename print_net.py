import hydra
from omegaconf import DictConfig
from hoechstgan.models import create_model


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def print_net(cfg: DictConfig) -> None:
    model = create_model(cfg)
    model.setup(cfg)


if __name__ == "__main__":
    print_net()
