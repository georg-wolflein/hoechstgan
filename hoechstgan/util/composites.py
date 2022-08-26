from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy as np

from .registry import Registry

composite_factory = Registry()


class Composite(ABC):

    @abstractmethod
    def __call__(self, *inputs, epoch: float):
        raise NotImplementedError


@composite_factory.register_as("default")
class TakeFirstComposite(Composite):

    def __init__(self, cfg: DictConfig, /, **kwargs):
        pass

    def __call__(self, *inputs, epoch: float):
        return inputs[0]


@composite_factory.register_as("linear")
class LinearComposite(Composite):

    def __init__(self, cfg: DictConfig, /, *, from_epoch=None, to_epoch=None, **kwargs):
        if from_epoch is None:
            from_epoch = 0
        if to_epoch is None:
            to_epoch = cfg.num_epochs
        self.from_epoch = float(from_epoch)
        self.to_epoch = float(to_epoch)

    def __call__(self, *inputs, epoch: float):
        a, b = inputs
        if epoch < self.from_epoch:
            coef = 0.
        elif epoch > self.to_epoch:
            coef = 1.
        else:
            coef = (epoch - self.from_epoch) / \
                (self.to_epoch - self.from_epoch)
        return (1. - coef) * a + coef * b


@composite_factory.register_as("sigmoid")
class SigmoidComposite(Composite):

    def __init__(self, cfg: DictConfig, /, *, from_epoch=None, to_epoch=None, **kwargs):
        if from_epoch is None:
            from_epoch = 0
        if to_epoch is None:
            to_epoch = cfg.num_epochs
        self.from_epoch = float(from_epoch)
        self.to_epoch = float(to_epoch)

    def _sigmoid(self, x):
        # x should be between 0 and 1; we're clamping the original sigmoid function between -5 and 5
        assert 0. <= x <= 1.
        x = (x - 0.5) * 10.
        return 1 / (1 + np.exp(-x))

    def __call__(self, *inputs, epoch: float):
        a, b = inputs
        if epoch < self.from_epoch:
            coef = 0.
        elif epoch > self.to_epoch:
            coef = 1.
        else:
            coef = (epoch - self.from_epoch) / \
                (self.to_epoch - self.from_epoch)
            coef = self._sigmoid(coef)
        return (1. - coef) * a + coef * b
