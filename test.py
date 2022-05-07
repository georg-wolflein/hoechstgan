import hydra
from omegaconf import DictConfig
import time

from hoechstgan.models import create_model
from hoechstgan.data import create_dataset
from hoechstgan.util.logging import ModelLogger


@hydra.main(config_path="conf", config_name="test")
def test(cfg: DictConfig) -> None:
    dataset = create_dataset(cfg)

    model = create_model(cfg)
    model.setup(cfg)

    data = next(iter(dataset))
    model.set_input(data)
    model.test()


if __name__ == "__main__":
    test()
