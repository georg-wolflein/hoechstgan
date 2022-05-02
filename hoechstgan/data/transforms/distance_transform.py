from omegaconf import DictConfig
from scipy import ndimage
import torch
import numpy as np
from PIL import Image


class DistanceTransform(torch.nn.Module):
    def __init__(self, opt: DictConfig):
        super().__init__()
        self.opt = opt

    def forward(self, img):
        img = ndimage.distance_transform_edt(img)
        if img.max() != 0.:
            img = img / img.max() * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img


__export__ = DistanceTransform
