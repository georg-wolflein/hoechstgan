import torch
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from PIL import Image


class BaseTransform(torch.nn.Module, ABC):
    def __init__(self, opt: DictConfig):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, img: Image.Image) -> Image.Image:
        raise NotImplementedError
