import itertools
import pandas as pd
import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import DictConfig
from tqdm import tqdm
from skimage.morphology import label

from hoechstgan.data import create_dataset


def compute_mask_stats(mask):
    mask = mask != 0
    return {
        "cells": label(mask != 0, connectivity=1).max(),
        "present": mask.max() > 0,
        "area": mask.sum() / mask.size,
    }


def compute_stats(cfg: DictConfig) -> None:
    dataset = create_dataset(cfg)
    dataset_size = len(dataset)

    def yield_from_dict(d):
        iters = {k: iter(v) for (k, v) in d.items()}
        try:
            while True:
                yield {k: next(v) for (k, v) in iters.items()}
        except StopIteration:
            pass

    def compute():
        data_generator = itertools.chain.from_iterable(
            map(yield_from_dict, dataset))
        data_generator = itertools.islice(data_generator, 500)
        for data in tqdm(data_generator, total=dataset_size, desc=f"processing {cfg.phase} dataset"):
            yield dict(
                (f"{channel} {stat}", value)
                for channel in ("Hoechst", "CD3", "CD8")
                for (stat, value) in compute_mask_stats(data[f"{channel}_mask"].numpy().squeeze(axis=0)).items()
            )

    df = pd.DataFrame(list(compute()))
    df = df.agg(["count", "sum", "mean", "std", "min", "max"])
    df.to_csv(f"dataset_stats_{cfg.phase}.csv")
    print(df)


if __name__ == "__main__":
    initialize(config_path="conf", version_base="1.2")
    for phase in "train", "test":
        cfg = compose("config.yaml", overrides=["+experiment=compute_stats",
                                                f"phase={phase}",
                                                f"is_train={str(phase == 'train').lower()}"])
        compute_stats(cfg)
        break
