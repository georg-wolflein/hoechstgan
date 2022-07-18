import itertools
from omegaconf import OmegaConf
import torch
from .base_model import BaseModel
from . import networks


class HoechstGANModel(BaseModel):
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
        super().__init__(cfg)
        self.generator_gt_losses = cfg.loss.generator.ground_truth.keys()
        self.loss_names = ["G", "D"] + \
            list(itertools.chain(*[[f"G{i}_GAN", f"G{i}_ground_truth",
                                    *(f"G{i}_ground_truth_{x}" for x in self.generator_gt_losses),
                                    f"D{i}_real", f"D{i}_fake"]
                                   for i in (1, 2)]))
        self.visual_names = [
            "real_A", "fake_B", "real_B", "fake_C", "real_C"]
        if self.is_train:
            self.model_names = ['G', 'D1', 'D2']
        else:  # at test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(cfg.dataset.input.num_channels, cfg.dataset.outputs.B.num_channels, cfg.generator.filters,
                                      cfg.norm, cfg.generator.dropout, cfg.initialization, cfg.initialization_scale, cfg.gpus,
                                      encoders=OmegaConf.to_container(
                                          cfg.generator.encoders),
                                      decoders=OmegaConf.to_container(
                                          cfg.generator.decoders),
                                      outputs=OmegaConf.to_container(
                                          cfg.generator.outputs),
                                      verbose=cfg.verbose)

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.B.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
                                           cfg.norm, cfg.initialization, cfg.initialization_scale, cfg.gpus)
            self.netD2 = networks.define_D(cfg.dataset.input.num_channels + cfg.dataset.outputs.C.num_channels, cfg.discriminator.filters, cfg.discriminator.layers,
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
            # self.optimizer_D = torch.optim.Adam(
            #     list(self.netD1.parameters()) + list(self.netD2.parameters()), lr=cfg.learning_rate.initial, betas=(cfg.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(
            ), lr=cfg.learning_rate.initial, betas=(cfg.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(
            ), lr=cfg.learning_rate.initial, betas=(cfg.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)

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

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_AC = torch.cat((self.real_A, self.fake_C), 1)
        pred_fake_AB = self.netD1(fake_AB.detach())
        pred_fake_AC = self.netD2(fake_AC.detach())
        self.loss_D1_fake = self.criterionGAN(pred_fake_AB, False)
        self.loss_D2_fake = self.criterionGAN(pred_fake_AC, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_AC = torch.cat((self.real_A, self.real_C), 1)
        pred_real_AB = self.netD1(real_AB)
        pred_real_AC = self.netD2(real_AC)
        self.loss_D1_real = self.criterionGAN(pred_real_AB, True)
        self.loss_D2_real = self.criterionGAN(pred_real_AC, True)
        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        self.loss_D = self.loss_D1 + self.loss_D2  # TODO: divide by 2?
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_AC = torch.cat((self.real_A, self.fake_C), 1)
        pred_fake_AB = self.netD1(fake_AB)
        pred_fake_AC = self.netD2(fake_AC)
        self.loss_G1_GAN = self.criterionGAN(pred_fake_AB, True)
        self.loss_G2_GAN = self.criterionGAN(pred_fake_AC, True)

        # Second, G(A) = B

        def gt_losses(i):
            G_gt_losses = []
            for loss_name, weight in self.cfg.loss.generator.ground_truth.items():
                L = self.criteria_ground_truth[loss_name.lower()]
                if i == 1:
                    L = L(self.fake_B, self.real_B) * weight
                elif i == 2:
                    L = L(self.fake_C, self.real_C) * weight
                setattr(self, f"loss_G{i}_ground_truth_{loss_name.lower()}", L)
                G_gt_losses.append(L)
            setattr(self, f"loss_G{i}_ground_truth", sum(G_gt_losses))
        gt_losses(1)
        gt_losses(2)
        # combine loss and calculate gradients
        self.loss_G1 = self.loss_G1_GAN + self.loss_G1_ground_truth
        self.loss_G2 = self.loss_G2_GAN + self.loss_G2_ground_truth
        self.loss_G = self.loss_G1 + self.loss_G2  # TODO: divide by 2?
        self.loss_G.backward()

    # def backward_G1(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     # First, G(A) should fake the discriminator
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake_AB = self.netD1(fake_AB)
    #     self.loss_G1_GAN = self.criterionGAN(pred_fake_AB, True)
    #     # Second, G(A) = B

    #     G_gt_losses = []
    #     for loss_name, weight in self.cfg.loss.generator.ground_truth.items():
    #         L = self.criteria_ground_truth[loss_name.lower()]
    #         L = L(self.fake_B, self.real_B) * weight
    #         setattr(self, f"loss_G1_ground_truth_{loss_name.lower()}", L)
    #         G_gt_losses.append(L)
    #     self.loss_G1_ground_truth = sum(G_gt_losses)
    #     # combine loss and calculate gradients
    #     self.loss_G1 = self.loss_G1_GAN + self.loss_G1_ground_truth
    #     self.loss_G1.backward()

    # def backward_G2(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     # First, G(A) should fake the discriminator
    #     fake_AC = torch.cat((self.real_A, self.fake_C), 1)
    #     pred_fake_AC = self.netD1(fake_AC)
    #     self.loss_G2_GAN = self.criterionGAN(pred_fake_AC, True)
    #     # Second, G(A) = C

    #     G_gt_losses = []
    #     for loss_name, weight in self.cfg.loss.generator.ground_truth.items():
    #         L = self.criteria_ground_truth[loss_name.lower()]
    #         L = L(self.fake_C, self.real_C) * weight
    #         setattr(self, f"loss_G2_ground_truth_{loss_name.lower()}", L)
    #         G_gt_losses.append(L)
    #     self.loss_G2_ground_truth = sum(G_gt_losses)
    #     # combine loss and calculate gradients
    #     self.loss_G2 = self.loss_G2_GAN + self.loss_G2_ground_truth
    #     self.loss_G2.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.optimizer_D1.zero_grad()     # set D's gradients to zero
        self.optimizer_D2.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D1.step()          # update D's weights
        self.optimizer_D2.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD1, False)
        self.set_requires_grad(self.netD2, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights
