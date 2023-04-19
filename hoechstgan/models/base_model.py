import itertools
import os
import shutil
import typing
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from pathlib import Path

from hoechstgan.util.logging import get_current_run_id

from . import networks


def _count_params(params: typing.Iterable) -> int:
    # Count number of parameters, counting shared parameters only once
    return sum({param.data_ptr(): param.numel()
                for param in params
                if param.requires_grad  # filter only trainable
                }.values())


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, cfg).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the BaseModel class.

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, cfg)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.cfg = cfg
        self.gpus = cfg.gpus
        self.is_train = cfg.is_train
        self.device = torch.device(f"cuda:{self.gpus[0]}") \
            if self.gpus else torch.device('cpu')  # get device name: CPU or GPU
        # Save all the checkpoints to save_dir
        self.save_dir = Path(cfg.checkpoints_dir) / get_current_run_id(cfg)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints directory: {self.save_dir}")
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration."""
        pass

    def setup(self, cfg):
        """Load and print networks; create schedulers."""
        if self.is_train:
            self.schedulers = [networks.get_scheduler(
                optimizer, cfg) for optimizer in self.optimizers]
        if not self.is_train or cfg.load_checkpoint:
            prefix = "latest"
            if cfg.load_checkpoint == "epoch":
                prefix = int(cfg.initial_epoch)
            self.load_networks(prefix)
        self.print_networks(cfg.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self, **kwargs):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward(**kwargs)
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.cfg.learning_rate.policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                try:
                    errors_ret[name] = float(getattr(self, 'loss_' + name))
                except AttributeError:
                    pass
        return errors_ret

    def save_networks(self, epoch):
        if isinstance(epoch, int):
            epoch = f"{epoch:03d}"
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_net_{name}.pth"
                save_path = self.save_dir / save_filename
                print(f"Saving network: {save_path}")
                net = getattr(self, "net" + name)

                if len(self.gpus) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpus[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        print(f"Loading network from epoch: {epoch}")
        if isinstance(epoch, int):
            epoch = f"{epoch:03d}"
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = self.save_dir / load_filename
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f"Loading module from {load_path}")
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = _count_params(net.parameters())
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    def num_parameters_by_net(self) -> dict:
        return {
            name: _count_params(getattr(self, "net" + name).parameters())
            for name in self.model_names if isinstance(name, str)
        }

    @property
    def num_parameters(self):
        # Ensure we don't count parameters multiple times
        return _count_params(itertools.chain.from_iterable(getattr(self, "net" + name).parameters()
                                                           for name in self.model_names
                                                           if isinstance(name, str)))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_state_dict(self):
        return {
            "net" + name: getattr(self, "net" + name).state_dict()
            for name in self.model_names if isinstance(name, str)
        }

    def load_state_dict(self, state_dict):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, "net" +
                        name).load_state_dict(state_dict["net" + name])

    def share_memory(self, *args, **kwargs):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, "net" + name).share_memory(*args, **kwargs)
