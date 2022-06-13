import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """This is the pix2pix model, as described in https://arxiv.org/pdf/1611.07004.pdf.

    Implementation inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
    """

    """
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """

    def __init__(self, cfg):
        BaseModel.__init__(self, cfg)
        self.generator_gt_losses = cfg.loss.generator.ground_truth.keys()
        self.loss_names = ["G_GAN", "G_ground_truth",
                           *(f"G_ground_truth_{x}" for x in self.generator_gt_losses),
                           "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.is_train:
            self.model_names = ['G', 'D']
        else:  # at test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(cfg.dataset.input.num_channels, cfg.dataset.outputs.B.num_channels, cfg.generator.filters,
                                      cfg.norm, cfg.generator.dropout, cfg.initialization, cfg.initialization_scale, cfg.gpus)

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.B.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
                                          cfg.norm, cfg.initialization, cfg.initialization_scale, cfg.gpus)

            # define loss functions
            self.criterionGAN = networks.GANLoss("vanilla").to(self.device)
            self.criteria_ground_truth = {
                "l1": torch.nn.L1Loss(),
                "l2": torch.nn.MSELoss(),
                "kl": torch.nn.KLDivLoss()  # TODO: check that order is correct (i.e. KL(fake, real))
            }
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=cfg.learning_rate.initial, betas=(cfg.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=cfg.learning_rate.initial, betas=(cfg.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.real_A = input["A"].to(self.device)
        self.real_B = input["B"].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        G_gt_losses = []
        for loss_name, weight in self.cfg.loss.generator.ground_truth.items():
            L = self.criteria_ground_truth[loss_name.lower()]
            L = L(self.fake_B, self.real_B) * weight
            setattr(self, f"loss_G_ground_truth_{loss_name.lower()}", L)
            G_gt_losses.append(L)
        self.loss_G_ground_truth = sum(G_gt_losses)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_ground_truth
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
