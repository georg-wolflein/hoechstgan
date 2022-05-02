from omegaconf import DictConfig, OmegaConf
import wandb

from . import tensor2im


class ModelLogger:
    def __init__(self, cfg: DictConfig):
        wandb.init(project="hoechstgan", name=cfg.name,
                   config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        # wandb.run._label(repo="hoechstgan")
        self.current_epoch = 0

    def log(self, step, epoch, iters, losses, t_comp, t_data, visuals=None):
        log_dict = {"epoch": epoch, "iters": iters}

        # Log losses
        message = f"({epoch=:d}, {iters=:d}, {t_comp=:f}, {t_data=:f}): "
        message += ", ".join(f"{k:s}={v:.3f}" for (k, v) in losses.items())
        print(message)

        log_dict.update({f"loss/{k}": v for (k, v) in losses.items()})

        # Log images
        if visuals:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, "epoch")
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            log_dict.update(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                log_dict["result"] = result_table
        wandb.log(log_dict, step=step)
