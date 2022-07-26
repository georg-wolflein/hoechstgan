from abc import ABC, abstractmethod
from omegaconf import DictConfig

from .registry import Registry

composite_factory = Registry()


class Composite(ABC):

    @abstractmethod
    def __call__(self, *inputs, epoch):
        raise NotImplementedError


@composite_factory.register_as("default")
class TakeFirstComposite(Composite):

    def __init__(self, cfg: DictConfig):
        pass

    def __call__(self, *inputs, epoch: int):
        return inputs[0]


@composite_factory.register_as("linear")
class LinearComposite(Composite):

    def __init__(self, cfg: DictConfig):
        self.num_epochs = cfg.num_epochs

    def __call__(self, *inputs, epoch: int):
        a, b = inputs
        coef = epoch / (self.num_epochs - 1)
        return (1. - coef) * a + coef * b
