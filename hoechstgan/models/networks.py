from typing import Callable
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from ..util.module import PairedSerializedModuleDict
from ..util.composites import composite_factory
from ..util.dropout import Dropout


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, cfg):
    """Return a learning rate scheduler

    cfg.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <cfg.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    lr = cfg.learning_rate
    if lr.policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + cfg.initial_epoch -
                             lr.n_epochs_initial) / float(lr.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr.policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr.decay_iters, gamma=0.1)
    elif lr.policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr.policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr.n_epochs_initial, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr.policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(cfg: DictConfig):
    input_nc = cfg.dataset.input.num_channels
    output_nc = cfg.dataset.outputs.B.num_channels
    filters = cfg.generator.filters
    use_dropout = cfg.generator.dropout
    dropout_eval_mode = cfg.generator.dropout_eval_mode
    encoders = OmegaConf.to_container(cfg.generator.encoders, resolve=True)
    decoders = OmegaConf.to_container(cfg.generator.decoders, resolve=True)
    outputs = OmegaConf.to_container(cfg.generator.outputs, resolve=True)
    composites = OmegaConf.to_container(cfg.generator.composites, resolve=True)

    norm_layer = get_norm_layer(norm_type=cfg.norm)

    def make_net(factory, **kwargs):
        defaults = dict(input_nc=input_nc, output_nc=output_nc, num_downs=8, filters=filters,
                        norm_layer=norm_layer, use_dropout=use_dropout, dropout_eval_mode=dropout_eval_mode)
        defaults.update(**kwargs)
        return factory(**defaults)

    def except_keys(d: dict, *keys):
        return {k: v for (k, v) in d.items() if k not in keys}

    def force_tuple(x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
        else:
            return x,

    RESERVED_KEYS = ("from", "to")

    encoders = {
        (enc["from"], enc["to"]):
        make_net(UnetEncoder, **except_keys(enc, *RESERVED_KEYS))
        for enc in encoders
    }
    decoders = {
        (force_tuple(dec["from"]), dec["to"]):
        make_net(UnetDecoder, **except_keys(dec, *RESERVED_KEYS))
        for dec in decoders
    }
    composites = {
        (force_tuple(comp[cfg.phase]["from"]), comp["to"]):
            composite_factory[comp[cfg.phase].get("schedule", "default")](
                cfg, **comp[cfg.phase].get("args", dict()))
        for comp in composites
    }
    reals = ["A", *cfg.dataset.outputs.keys()]

    net = UnetGenerator(encoders, decoders, composites, reals, outputs)
    if cfg.verbose:
        net.describe()
    return init_net(net, cfg.initialization, cfg.initialization_scale, cfg.gpus)


def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def make_unet_layers(UnetBlock, input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout, dropout_eval_mode="dropout"):
    yield UnetBlock(output_nc, ngf, input_nc=input_nc, outermost=True,
                    norm_layer=norm_layer)  # outermost layer
    # gradually increase the number of filters from ngf to ngf * 8
    yield UnetBlock(ngf, ngf * 2, input_nc=None,
                    norm_layer=norm_layer)
    yield UnetBlock(ngf * 2, ngf * 4, input_nc=None,
                    norm_layer=norm_layer)
    yield UnetBlock(ngf * 4, ngf * 8, input_nc=None,
                    norm_layer=norm_layer)
    # intermediate layers with ngf * 8 filters
    for _ in range(num_downs - 5):
        yield UnetBlock(ngf * 8, ngf * 8, input_nc=None,
                        norm_layer=norm_layer, use_dropout=use_dropout, dropout_eval_mode=dropout_eval_mode)
    yield UnetBlock(ngf * 8, ngf * 8, input_nc=None,
                    norm_layer=norm_layer, innermost=True)  # innermost layer


class UnetEncoder(nn.ModuleList):

    def __init__(self, input_nc, output_nc, num_downs, filters=64, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a Unet generator encoder
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            filters (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__(list(make_unet_layers(UnetDown,
                                               input_nc, output_nc, num_downs, filters, norm_layer, **kwargs)))

    def forward(self, x):
        intermediate_outputs = []
        for layer in self:
            x = layer(x)
            intermediate_outputs.append(x)
        return intermediate_outputs  # last intermediate output is result


class UnetDecoder(nn.ModuleList):
    """See UnetEncoder."""

    def __init__(self, input_nc, output_nc, num_downs, filters=64, norm_layer=nn.BatchNorm2d, use_dropout=False, dropout_eval_mode="dropout", **kwargs):
        super().__init__(reversed(list(make_unet_layers(UnetUp,
                                                        input_nc, output_nc, num_downs, filters, norm_layer, use_dropout, dropout_eval_mode, **kwargs))))


# class UnetGenerator(nn.Module):
#     """Unet-based generator"""

#     def __init__(self, encoder: UnetEncoder, decoder: UnetDecoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         x, xs = self.encoder(x)
#         xs.pop()
#         for up_layer in self.decoder.layers:
#             x = up_layer(x)
#             if len(xs) > 0:
#                 x_prev = xs.pop()
#                 x = torch.cat((x, x_prev), 1)
#         return x


class UnetGenerator(nn.Module):
    """Unet-based generator"""

    def __init__(self, encoders: dict, decoders: dict, composites: list, reals: list, outputs: list):
        super().__init__()

        self.encoders = PairedSerializedModuleDict(encoders)
        self.decoders = PairedSerializedModuleDict(decoders)
        self.reals = reals
        self.outputs = outputs  # names of outputs
        self.composites = composites

    def _decode(self, decoder, *latents):
        # Merge latents layer-wise (concatenate across channel dimension)
        outputs = [torch.cat(layer_outputs, axis=-3)  # tensors are of shape BCWH
                   for layer_outputs in zip(*latents)]
        *xs, x = outputs  # outputs is list of intermediate outputs, where last one is the latent code
        for up_layer in decoder:
            x = up_layer(x)
            if len(xs) > 0:
                *xs, x_prev = xs
                x = torch.cat((x, x_prev), 1)
        return x

    def forward(self, real_A, /,
                dry_run: bool = False,
                verbose: bool = None,
                latent_substitutions: dict = {},
                input_substitutions: dict = {},
                real_inputs: dict = {},
                epoch: float = None):
        if dry_run and verbose is None:
            verbose = True
        log = print if verbose else lambda _: None
        log("Dry run of generator:")
        outputs = {
            "real_A": real_A,
            **real_inputs
        }
        latents = {}
        progress = True
        while progress:
            progress = False
            for (enc_from, enc_to), encoder in self.encoders.items():
                if enc_to not in latents and enc_from in outputs:
                    latent = None
                    log(f"  Encoding {enc_from} -> {enc_to}")
                    if not dry_run:
                        output = outputs[enc_from]
                    if enc_from in input_substitutions:
                        log(f"    (using substituted {enc_from})")
                        if not dry_run:
                            output = input_substitutions[enc_from]
                    if not dry_run:
                        latent = encoder(output)
                        if enc_to in latent_substitutions:
                            log(f"  Substituting latent code for {enc_to}")
                            latent = latent_substitutions[enc_to](latent)
                    latents[enc_to] = latent
                    progress = True
            for (comp_from, comp_to), composite in self.composites.items():
                if comp_to not in outputs and all(map(outputs.keys().__contains__, comp_from)):
                    output = None
                    log(f"  Compositing {','.join(comp_from)} -> {comp_to}")
                    if not dry_run:
                        output = composite(
                            *(outputs[c] for c in comp_from), epoch=epoch)
                    outputs[comp_to] = output
                    progress = True
            for (dec_from, dec_to), decoder in self.decoders.items():
                if dec_to not in outputs and all(map(latents.keys().__contains__, dec_from)):
                    output = None
                    log(f"  Decoding {','.join(dec_from)} -> {dec_to}")
                    if not dry_run:
                        output = self._decode(
                            decoder, *map(latents.__getitem__, dec_from))
                    outputs[dec_to] = output
                    progress = True
        log(f"  Done, generated following outputs: {', '.join(outputs)}")
        return tuple(outputs[k] for k in self.outputs)

    def describe(self):
        self.forward(None, dry_run=True,
                     real_inputs={
                         f"real_{x}": None for x in self.reals
                     })


class UnetDown(nn.Sequential):

    def __init__(self, outer_nc, inner_nc, input_nc=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, **kwargs):
        """Unet down block.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
        """
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)  # , True)
        downnorm = norm_layer(inner_nc)

        if outermost:
            down = [downconv]
        elif innermost:
            down = [downrelu, downconv]
        else:
            down = [downrelu, downconv, downnorm]

        super().__init__(*down)


class UnetUp(nn.Sequential):

    def __init__(self, outer_nc, inner_nc, input_nc=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, dropout_eval_mode="dropout", **kwargs):
        """Unet up block.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            dropout_eval_mode (str) -- if use_dropout, what mode to use for dropout during evaluation
        """
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        uprelu = nn.ReLU()  # (True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            up = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [Dropout(0.5, eval_mode=dropout_eval_mode)]

        super().__init__(*up)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
