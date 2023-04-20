from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import sys

from . import tensor2im

WANDB_ENTITY = "georgw7777"
WANDB_PROJECT = "hoechstgan"


def get_current_run_id(cfg: DictConfig) -> str:
    if cfg.wandb_id:
        return cfg.wandb_id
    if wandb.run:
        return wandb.run.id
    return cfg.name


def get_api() -> wandb.Api:
    return wandb.Api(dict(entity=WANDB_ENTITY, project=WANDB_PROJECT))


class ClientLogger:
    def __init__(self, model_logger: "ModelLogger", client_id: int):
        self.model_logger = model_logger
        self.client_id = client_id
        self.cfg = model_logger.cfg

    def log(self, *args, **kwargs):
        return self.model_logger.log(*args, client_id=self.client_id, **kwargs)


class ModelLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __enter__(self):
        cfg = {
            **OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
            "overrides": " ".join(x
                                  for x in sys.argv[1:]
                                  if not x.startswith("+experiment=") and not x.startswith("gpus="))
        }
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                   name=self.cfg.name, id=self.cfg.wandb_id,
                   config=cfg)
        self.cfg.wandb_id = wandb.run.id
        return self

    def __exit__(self, *args, **kwargs):
        wandb.finish()

    def for_client(self, client_id: int) -> ClientLogger:
        return ClientLogger(self, client_id)

    def log(self, step, epoch, iters, losses, t_comp, t_data, visuals=None, client_id=None):
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
        log_dict = {
            f"client_{client_id}/{k}": v for (k, v) in log_dict.items()}
        wandb.log(log_dict, step=step)
