#import sys
# sys.path.append('/opt/conda/lib/python3.9/site-packages')

import hydra
from omegaconf import DictConfig
import time
from tqdm import tqdm
import copy
import torch.multiprocessing as mp
import contextlib

from hoechstgan.models import create_model
from hoechstgan.data import create_dataset
from hoechstgan.util.logging import ModelLogger, ClientLogger
from federation.federate import *


def setup_model(cfg, gpu):
    cfg = copy.deepcopy(cfg)
    cfg.gpus = [gpu]
    model = create_model(cfg)
    model.setup(cfg)
    # model.share_memory()
    # model.print_networks(cfg.verbose)
    return model, cfg


def train_one_epoch(cfg, model, dataset, epoch, model_logger):
    epoch_iter = 0  # number of training iterations in current epoch
    iter_data_time = time.time()    # timer for data loading per iteration

    dataset_size = len(dataset)
    step = epoch * (dataset_size // cfg.dataset.batch_size)

    # Perform one epoch of training
    # for data in tqdm(dataset, desc=f"Epoch {epoch}", total=dataset_size // cfg.dataset.batch_size):
    for data in dataset:
        iter_start_time = time.time()  # timer for computation per iteration

        step += cfg.dataset.batch_size
        epoch_iter += cfg.dataset.batch_size
        # Epoch number, but continuous (i.e. epoch 1 on step 3/10 will be epoch 1.3)
        continuous_epoch = epoch + \
            min(1., epoch_iter / dataset_size)

        # Train model
        model.set_input(data)  # preprocess data
        # Compute loss functions, get gradients, update weights
        model.optimize_parameters(epoch=continuous_epoch)

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
        iter_data_time = time.time()


def share_params(params):
    return {
        model: {
            k: v.to("cpu").share_memory_() for k, v in state_dict.items()
        } for (model, state_dict) in params.items()
    }


def worker(input_queue, output_queue, model, model_cfg):
    print(f"Starting worker on {model_cfg.gpus}")
    while True:
        data = input_queue.get()
        if data is None:
            print(f"Stopping worker on {model_cfg.gpus}")
            return
        epoch, client_dataset, global_params, model_logger = data

        print(
            f"Loading global parameters into client model on device {model_cfg.gpus}")
        model.load_state_dict(global_params)

        print(f"Training client on device {model_cfg.gpus}")
        train_one_epoch(model_cfg, model, client_dataset, epoch, model_logger)

        print(f"Sending parameters from client on device {model_cfg.gpus}")
        output_queue.put(share_params(model.get_state_dict()))


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    assert cfg.visualize_freq % cfg.log_freq == 0
    cfg.dataset.num_threads = 0

    dataset = create_dataset(cfg)
    dataset_size = len(dataset)
    print(f"Number of training images: {dataset_size}")

    dataset1, dataset2, dataset3, dataset4 = create_4_clients(cfg)
    datasets = [dataset1, dataset2, dataset3, dataset4]

    with ModelLogger(cfg) as model_logger:
        models, model_cfgs = zip(*[setup_model(cfg, gpu) for gpu in cfg.gpus])
        # model = create_model(cfg)
        # model.setup(cfg)
        # model.print_networks(cfg.verbose)

        # Split dataset into clients
        datasets_by_worker = [(i % len(models), dataset)
                              for i, dataset in enumerate(datasets)]
        with contextlib.ExitStack() as client_logger_stack:
            client_loggers = [client_logger_stack.enter_context(ClientLogger(model_logger, i))
                              for i in range(len(datasets))]
            # Get global parameters
            model = models[0]
            global_params = model.get_state_dict()
            model.print_networks(cfg.verbose)

            # Initialize multiprocessing
            mp.set_start_method("spawn", force=True)
            input_queues = [mp.Queue() for _ in range(len(models))]
            output_queue = mp.Queue()
            processes = []

            # Start worker processes
            print("Starting workers")
            for model, model_cfg, input_queue in zip(models, model_cfgs, input_queues):
                p = mp.Process(target=worker, args=(
                    input_queue, output_queue, model, model_cfg))
                p.start()
                processes.append(p)

            # Outer loop for different epochs
            for epoch in range(cfg.initial_epoch, cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay + 1):
                epoch_start_time = time.time()  # timer for entire epoch
                print(f"Starting epoch {epoch}")

                # Distribute tasks
                print("Distributing tasks")
                for client_id, ((model_id, dataset), client_logger) in enumerate(zip(datasets_by_worker, client_loggers)):
                    print(f"Sending dataset {client_id} worker {model_id}")
                    input_queues[model_id].put(
                        (epoch, dataset, share_params(global_params), client_logger))

                # Receive client parameters
                print("Waiting for clients to finish training")
                client_params = [output_queue.get()
                                 for _ in range(len(datasets_by_worker))]
                print("Received client parameters")

                # Aggregate client parameters
                print("Aggregating client parameters")
                global_params = fed_avg(client_params)
                print("Deleting client parameters")
                del client_params

                if epoch % cfg.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                    print(f"Saving model ({epoch=})")
                    model.load_state_dict(global_params)
                    model.save_networks("latest")
                    model.save_networks(epoch)

                for model in models:
                    model.update_learning_rate()  # update learning rates at end of every epoch
                print(
                    f"End of epoch {epoch} / {cfg.learning_rate.n_epochs_initial + cfg.learning_rate.n_epochs_decay:d} \t Time Taken: {time.time() - epoch_start_time} sec")

        # Terminate processes
        for input_queue in input_queues:
            input_queue.put(None)  # send stop signal
        for p in processes:
            p.join()


if __name__ == "__main__":
    train()
