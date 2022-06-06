import importlib
from PIL import Image
import typing
from matplotlib import transforms
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torchvision import transforms as T


def find_transform_by_name(name: str):
    module_name = f"{__name__}.{name}"
    lib = importlib.import_module(module_name)
    return lib.__export__


def get_transform(cfg: DictConfig, opt: DictConfig) -> typing.Callable[[Image.Image], torch.Tensor]:
    transforms = opt.transforms
    grayscale = opt.num_channels == 1
    funcs = []
    if grayscale:
        funcs.append(T.Grayscale(1))
    for transform in transforms:
        if isinstance(transform, str):
            name = transform
            opt = DictConfig({})
        else:
            name, opt = next(iter(transform.items()))
        funcs.append(find_transform_by_name(name)(opt))
    funcs.append(T.ToTensor())
    if grayscale:
        funcs.append(T.Normalize((0.5,), (0.5,)))
    else:
        funcs.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(funcs)
