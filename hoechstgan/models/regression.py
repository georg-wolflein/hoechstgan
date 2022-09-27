from omegaconf import OmegaConf
import torch
from .base_model import BaseModel
from . import networks


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
        return ["G",
                *get_losses(1), *get_losses(2)]

    @property
    def model_names(self):
        return ["G"]

    def backward(self):
        """Calculate GAN and L1 loss for the generator"""
        gan = self.gan

        # G(A) = B
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
        gan.loss_G = gan.loss_G_ground_truth
        # TODO: test divide by 2 using coefficient below
        gan.loss_G = gan.loss_G * gan.cfg.loss.generator.coefficient
        gan.loss_G.backward()

    def update(self):
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights


class RegressionModel(BaseModel):
    """Simple regression baseline without GAN loss."""

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.G = Generator(self)
        self.loss_names = self.G.loss_names
        self.visual_names = [
            "real_A", "fake_B", "real_B", "fake_C", "real_C"]
        self.model_names = self.G.model_names

        self.netG = self.G.netG

        if self.is_train:
            # define loss functions
            self.criteria_ground_truth = {
                "l1": torch.nn.L1Loss(),
                "l2": torch.nn.MSELoss(),
                "kl": torch.nn.KLDivLoss()  # TODO: check that order is correct (i.e. KL(fake, real))
            }
            # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # This is done by the Generator class by populating self.optimizers

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
        # update G
        self.G.update()
