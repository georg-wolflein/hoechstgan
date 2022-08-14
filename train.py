import hydra
from omegaconf import DictConfig
import time
from tqdm import tqdm

from hoechstgan.models import create_model
from hoechstgan.data import create_dataset
from hoechstgan.util.logging import ModelLogger


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    assert cfg.visualize_freq % cfg.log_freq == 0

    dataset = create_dataset(cfg)
    dataset_size = len(dataset)
    print(f"Number of training images: {dataset_size}")

    with ModelLogger(cfg) as model_logger:
        model = create_model(cfg)
        model.setup(cfg)
        step = cfg.initial_epoch * (dataset_size // cfg.dataset.batch_size)

        # outer loop for different epochs
        for epoch in range(cfg.initial_epoch, cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0  # number of training iterations in current epoch
            # Inner loop within one epoch
            for data in tqdm(dataset, desc=f"Epoch {epoch}", total=int(dataset_size / cfg.dataset.batch_size)):
                iter_start_time = time.time()  # timer for computation per iteration

                step += cfg.dataset.batch_size
                epoch_iter += cfg.dataset.batch_size

                model.set_input(data)  # preprocess data
                # Compute loss functions, get gradients, update weights
                model.optimize_parameters(epoch=epoch)

                if step % cfg.log_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    model.compute_visuals()
                    visuals = None
                    if step % cfg.visualize_freq == 0:
                        visuals = model.get_current_visuals()
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / \
                        cfg.dataset.batch_size
                    model_logger.log(step, epoch, epoch_iter,
                                     losses, t_comp, t_data,
                                     visuals)

                if step % cfg.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print(f"Saving latest model ({epoch=}, {step=})")
                    save_suffix = f"iter_{step if cfg.save_by_iter else 'latest'}"
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % cfg.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print(f"Saving model ({epoch=}, {step=})")
                model.save_networks("latest")
                model.save_networks(epoch)

            model.update_learning_rate()  # update learning rates at end of every epoch
            print(
                f"End of epoch {epoch} / {cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay:d} \t Time Taken: {time.time() - epoch_start_time} sec")


if __name__ == "__main__":
    train()
