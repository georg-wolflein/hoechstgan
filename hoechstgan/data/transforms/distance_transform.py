import typing
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage
import torch


class DistanceTransform(torch.nn.Module):
    def __init__(self, opt: DictConfig):
        super().__init__()
        self.opt = opt

    def forward(self, img):
        return ndimage.distance_transform_edt(img)


__export__ = DistanceTransform
