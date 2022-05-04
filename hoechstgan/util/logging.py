from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np

from . import tensor2im


class ModelLogger:
    def __init__(self, cfg: DictConfig):
        wandb.init(project="hoechstgan", name=cfg.name,
                   config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    def log(self, step, epoch, iters, losses, t_comp, t_data, visuals=None):
        log_dict = {"epoch": epoch, "iters": iters}

        # Log losses
        message = f"({epoch=:d}, {iters=:d}, {t_comp=:f}, {t_data=:f}): "
        message += ", ".join(f"{k:s}={v:.3f}" for (k, v) in losses.items())
        print(message)

        log_dict.update({f"loss/{k}": v for (k, v) in losses.items()})

        # Log images
        if visuals:
            imgs = [tensor2im(img) for img in visuals.values()]
            imgs = wandb.Image(np.concatenate(imgs, axis=1))
            log_dict["_".join(visuals.keys())] = imgs
        wandb.log(log_dict, step=step)
