#import sys
# sys.path.append('/opt/conda/lib/python3.9/site-packages')

import hydra
from omegaconf import DictConfig
import time
from tqdm import tqdm
import copy

from hoechstgan.models import create_model
from hoechstgan.data import create_dataset
from hoechstgan.util.logging import ModelLogger
from federation.federate import *


def setup_model(cfg, gpu):
    cfg = copy.deepcopy(cfg)
    cfg.gpus = [gpu]
    model = create_model(cfg)
    model.setup(cfg)
    # model.print_networks(cfg.verbose)
    return model, cfg


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    assert cfg.visualize_freq % cfg.log_freq == 0

    dataset = create_dataset(cfg)
    dataset_size = len(dataset)
    print(f"Number of training images: {dataset_size}")

    global_disc1_param = {}
    global_disc2_param = {}
    global_gen_param = {}

    dataset1, dataset2, dataset3, dataset4 = create_4_clients(cfg)
    datasets = [dataset1, dataset2, dataset3, dataset4]
    lendata = [len(dataset1), len(dataset2), len(dataset3), len(dataset4)]

    with ModelLogger(cfg) as model_logger:
        models, model_cfgs = zip(*[setup_model(cfg, gpu) for gpu in cfg.gpus])
        # model = create_model(cfg)
        # model.setup(cfg)
        # model.print_networks(cfg.verbose)
        step = cfg.initial_epoch * (dataset_size // cfg.dataset.batch_size)

        # Get global parameters
        model = models[0]
        global_disc1_param = model.D.netD1.state_dict()
        global_disc2_param = model.D.netD2.state_dict()
        global_gen_param = model.G.netG.state_dict()

        # outer loop for different epochs
        for epoch in range(cfg.initial_epoch, cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0  # number of training iterations in current epoch
            # Inner loop within one epoch

            clients_gen_param = []
            clients_disc1_param = []
            clients_disc2_param = []

            # Iterate over clients
            for client, dataset in enumerate(datasets):

                # Find the model to use
                model_id = client % len(models)
                model = models[model_id]
                model_cfg = model_cfgs[model_id]
                print(
                    f"Training client {client} with model {model_id} on device {model_cfg.gpus}")

                # Perform one epoch of training
                for data in tqdm(dataset, desc=f"Epoch {epoch}", total=dataset_size // cfg.dataset.batch_size):
                    iter_start_time = time.time()  # timer for computation per iteration

                    step += cfg.dataset.batch_size
                    epoch_iter += cfg.dataset.batch_size
                    # epoch number, but continuous (i.e. epoch 1 on step 3/10 will be epoch 1.3)
                    continuous_epoch = epoch + \
                        min(1., epoch_iter / dataset_size)

                    client_training(global_disc1_param, global_disc2_param, global_gen_param, clients_gen_param,
                                    clients_disc1_param, clients_disc2_param, model_cfg, epoch, data, continuous_epoch, model)

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
                    '''
                    if step % cfg.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                        print(f"Saving latest model ({epoch=}, {step=})")
                        save_suffix = f"iter_{step if cfg.save_by_iter else 'latest'}"
                        model.save_networks(save_suffix)'''

                print("client " + str(client + 1) + " has been trained")
            iter_data_time = time.time()

            global_disc1_param, global_disc2_param, global_gen_param = fed_avg(
                clients_disc1_param, clients_disc2_param, clients_gen_param)

            if epoch % cfg.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print(f"Saving model ({epoch=}, {step=})")
                model.G.netG.load_state_dict(global_gen_param)
                model.D.netD1.load_state_dict(global_disc1_param)
                model.D.netD2.load_state_dict(global_disc2_param)

                model.save_networks("latest")
                model.save_networks(epoch)

            model.update_learning_rate()  # update learning rates at end of every epoch
            print(
                f"End of epoch {epoch} / {cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay:d} \t Time Taken: {time.time() - epoch_start_time} sec")


if __name__ == "__main__":
    train()
