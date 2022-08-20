from abc import ABC, abstractmethod
import itertools
import typing
from omegaconf import OmegaConf
import torch
from .base_model import BaseModel
from . import networks


def make_input_substitution(cfg: OmegaConf):
    if not cfg.generator.substitute_input:
        return None

    num_epochs = cfg.learning_rate.n_epochs_inital + cfg.learning_rate.n_epochs_decay

    def substitute_input(outputs: dict, real_inputs: dict, key: str, epoch: int):
        out = outputs[key]
        if key == "fake_B":
            coef = epoch / (num_epochs-1)
            return coef * out + (1.-coef) * real_inputs["real_B"]

    return substitute_input


class Generator:

    def __init__(self, gan):
        self.gan = gan
        self.cfg = cfg = gan.cfg
        self.generator_gt_losses = {k: weight for (k, weight) in cfg.loss.generator.ground_truth.items()
                                    if weight != 0.}
        self.netG = networks.define_G(cfg)
        if self.cfg.is_train:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=cfg.learning_rate.initial,
                                                betas=(cfg.beta1, 0.999))
            gan.optimizers.append(self.optimizer_G)

    @property
    def loss_names(self):
        def get_losses(i):
            return [f"G{i}_ground_truth",
                    *(f"G{i}_ground_truth_{x}" for x in self.generator_gt_losses.keys())]
        return ["G", "G_GAN",
                *get_losses(1), *get_losses(2)]

    @property
    def model_names(self):
        return ["G"]

    def backward(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        gan = self.gan
        gan.D.compute_G_GAN_loss()  # will set gan.loss_G_GAN

        # Second, G(A) = B
        def gt_losses(i):
            G_gt_losses = []
            for loss_name, weight in self.generator_gt_losses.items():
                L = gan.criteria_ground_truth[loss_name.lower()]
                if i == 1:
                    L = L(gan.fake_B, gan.real_B) * weight
                elif i == 2:
                    L = L(gan.fake_C, gan.real_C) * weight
                setattr(gan, f"loss_G{i}_ground_truth_{loss_name.lower()}", L)
                G_gt_losses.append(L)
            setattr(gan, f"loss_G{i}_ground_truth", sum(G_gt_losses))
        gt_losses(1)
        gt_losses(2)
        # combine loss and calculate gradients
        gan.loss_G_ground_truth = gan.loss_G1_ground_truth + gan.loss_G2_ground_truth
        gan.loss_G = gan.loss_G_GAN + gan.loss_G_ground_truth
        # TODO: test divide by 2 using coefficient below
        gan.loss_G = gan.loss_G * gan.cfg.loss.generator.coefficient
        gan.loss_G.backward()

    def update(self):
        # D requires no gradients when optimizing G
        self.gan.D.set_requires_grad(False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights


class AbstractDiscriminator(ABC):

    def __init__(self, gan):
        self.gan = gan
        self.cfg = gan.cfg
        self.nets = []
        self.optimizers = []

    @property
    def net_suffixes(self):
        if len(self.nets) > 1:
            return list(map(str, range(1, len(self.nets)+1)))
        else:
            return [""] * len(self.nets)

    @property
    def loss_names(self):
        def get_losses(i):
            return [f"D{i}_real", f"D{i}_fake"]
        return ["G_GAN",  # yes, we compute loss_G_GAN in the Discriminator class
                "D", *itertools.chain.from_iterable(map(get_losses, self.net_suffixes))]

    @property
    def model_names(self):
        if self.cfg.is_train:  # at test time, don't load D
            return [f"D{i}" for i in self.net_suffixes]
        return []

    def set_requires_grad(self, requires_grad):
        for net in self.nets:
            self.gan.set_requires_grad(net, requires_grad)

    def update(self):
        self.set_requires_grad(True)  # enable backprop for D
        for opt in self.optimizers:  # set D's gradients to zero
            opt.zero_grad()
        self.backward()                # calculate gradients for D
        for opt in self.optimizers:  # update D's weights
            opt.step()

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def compute_G_GAN_loss(self):
        pass


class SeparateDiscriminator(AbstractDiscriminator):

    def __init__(self, gan):
        super().__init__(gan)
        cfg = self.cfg
        if self.cfg.is_train:
            self.netD1 = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.B.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
                                           cfg.norm, cfg.initialization, cfg.initialization_scale, cfg.gpus)
            self.netD2 = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.C.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
                                           cfg.norm, cfg.initialization, cfg.initialization_scale, cfg.gpus)
            self.nets.append(self.netD1)
            self.nets.append(self.netD2)
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                                 lr=cfg.learning_rate.initial,
                                                 betas=(cfg.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                 lr=cfg.learning_rate.initial,
                                                 betas=(cfg.beta1, 0.999))
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)
            gan.optimizers.extend(self.optimizers)

    @property
    def loss_names(self):
        return super().loss_names + ["G1_GAN", "G2_GAN"]

    def backward(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        gan = self.gan
        fake_AB = torch.cat((gan.real_A, gan.fake_B), 1)
        fake_AC = torch.cat((gan.real_A, gan.fake_C), 1)
        pred_fake_AB = gan.netD1(fake_AB.detach())
        pred_fake_AC = gan.netD2(fake_AC.detach())
        gan.loss_D1_fake = gan.criterionGAN(pred_fake_AB, False)
        gan.loss_D2_fake = gan.criterionGAN(pred_fake_AC, False)
        # Real
        real_AB = torch.cat((gan.real_A, gan.real_B), 1)
        real_AC = torch.cat((gan.real_A, gan.real_C), 1)
        pred_real_AB = gan.netD1(real_AB)
        pred_real_AC = gan.netD2(real_AC)
        gan.loss_D1_real = gan.criterionGAN(pred_real_AB, True)
        gan.loss_D2_real = gan.criterionGAN(pred_real_AC, True)
        # combine loss and calculate gradients
        gan.loss_D1 = (gan.loss_D1_fake + gan.loss_D1_real) * 0.5
        gan.loss_D2 = (gan.loss_D2_fake + gan.loss_D2_real) * 0.5
        gan.loss_D = gan.loss_D1 + gan.loss_D2  # TODO: divide by 2?
        gan.loss_D = gan.loss_D * self.cfg.loss.discriminator.coefficient
        gan.loss_D.backward()

    def compute_G_GAN_loss(self):
        # G(A) should fool the discriminator
        gan = self.gan
        fake_AB = torch.cat((gan.real_A, gan.fake_B), 1)
        fake_AC = torch.cat((gan.real_A, gan.fake_C), 1)
        pred_fake_AB = gan.netD1(fake_AB)
        pred_fake_AC = gan.netD2(fake_AC)
        gan.loss_G1_GAN = gan.criterionGAN(pred_fake_AB, True)
        gan.loss_G2_GAN = gan.criterionGAN(pred_fake_AC, True)
        gan.loss_G_GAN = gan.loss_G1_GAN + gan.loss_G2_GAN


class JointDiscriminator(AbstractDiscriminator):

    def __init__(self, gan):
        super().__init__(gan)
        cfg = self.cfg
        if self.cfg.is_train:
            self.netD = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.B.num_channels + cfg.dataset.outputs.C.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
                                          cfg.norm, cfg.initialization, cfg.initialization_scale, cfg.gpus)
            self.nets.append(self.netD)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=cfg.learning_rate.initial,
                                                betas=(cfg.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
            gan.optimizers.extend(self.optimizers)

    def backward(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        gan = self.gan
        fake_ABC = torch.cat((gan.real_A, gan.fake_B, gan.fake_C), 1)
        pred_fake_ABC = gan.netD(fake_ABC.detach())
        gan.loss_D_fake = gan.criterionGAN(pred_fake_ABC, False)
        # Real
        real_ABC = torch.cat((gan.real_A, gan.real_B, gan.real_C), 1)
        pred_real_ABC = gan.netD(real_ABC)
        gan.loss_D_real = gan.criterionGAN(pred_real_ABC, True)
        # combine loss and calculate gradients
        gan.loss_D = (gan.loss_D_fake + gan.loss_D_real) * 0.5
        gan.loss_D = gan.loss_D * self.cfg.loss.discriminator.coefficient
        gan.loss_D.backward()

    def compute_G_GAN_loss(self):
        # G(A) should fool the discriminator
        gan = self.gan
        fake_ABC = torch.cat((gan.real_A, gan.fake_B, gan.fake_C), 1)
        pred_fake_ABC = gan.netD(fake_ABC)
        gan.loss_G_GAN = gan.criterionGAN(pred_fake_ABC, True)


DISCRIMINATOR_FACTORY = {
    "separate": SeparateDiscriminator,
    "joint": JointDiscriminator
}


class HoechstGANModel(BaseModel):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.G = Generator(self)
        self.D = DISCRIMINATOR_FACTORY[cfg.discriminator.type](self)
        self.loss_names = self.G.loss_names + self.D.loss_names
        self.visual_names = [
            "real_A", "fake_B", "real_B", "fake_C", "real_C"]
        self.model_names = self.G.model_names + self.D.model_names

        for container in (self.D, self.G):
            for name in container.model_names:
                setattr(self, f"net{name}", getattr(container, f"net{name}"))

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define loss functions
            self.criterionGAN = networks.GANLoss("vanilla").to(self.device)
            self.criteria_ground_truth = {
                "l1": torch.nn.L1Loss(),
                "l2": torch.nn.MSELoss(),
                "kl": torch.nn.KLDivLoss()  # TODO: check that order is correct (i.e. KL(fake, real))
            }
            # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # This is done by the Generator/Discriminator classes by populating self.optimizers

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.real_A = input["A"].to(self.device)
        self.real_B = input["B"].to(self.device)
        self.real_C = input["C"].to(self.device)

    def forward(self, **kwargs):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake_C = self.netG(self.real_A, **kwargs)  # G(A)
        # print(list(map(torch.Tensor.size, (self.real_A, self.fake_B, self.fake_C))))

    def optimize_parameters(self, **kwargs):
        real_inputs = {
            "real_A": self.real_A,
            "real_B": self.real_B,
            "real_C": self.real_C,
        }
        # compute fake images: G(A)
        self.forward(**kwargs, real_inputs=real_inputs)
        # update D
        self.D.update()
        # update G
        self.G.update()
