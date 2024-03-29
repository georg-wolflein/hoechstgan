from omegaconf import DictConfig

from .base_model import BaseModel


def create_model(cfg: DictConfig) -> BaseModel:
    from .pix2pix import Pix2PixModel
    from .hoechstgan import HoechstGANModel
    from .regression import RegressionModel
    model = {
        "pix2pix": Pix2PixModel,
        "hoechstgan": HoechstGANModel,
        "regression": RegressionModel
    }[cfg.gan]
    return model(cfg)
