from omegaconf import DictConfig

from .base_model import BaseModel


def create_model(cfg: DictConfig) -> BaseModel:
    from .pix2pix import Pix2PixModel
    model = Pix2PixModel
    return model(cfg)
