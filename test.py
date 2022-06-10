
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from PIL import Image
import itertools
from functools import partial
from skimage.morphology import label
import argparse

from hoechstgan.data import create_dataset
from hoechstgan.models import create_model
from hoechstgan.util.dataset import get_channel_file_from_metadata
from hoechstgan.util.logging import WANDB_ENTITY, WANDB_PROJECT, get_api


def load_run_cfg(run_id: str) -> DictConfig:
    api = get_api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    cfg = DictConfig(run.config)
    cfg.wandb_id = run_id
    return cfg


def compute_mask_intensity_ratio(img, mask) -> float:
    numerator = img[mask].mean()
    denominator = img[~mask].mean()
    return numerator / denominator if denominator != 0 else 0


def get_img(tensor):
    return (tensor.cpu().numpy().squeeze() + 1) / 2.


def perform_test(data, model):
    model.set_input(data)
    model.test()

    for json_path, real_A, fake_B, real_B in zip(data["json_files"], model.real_A, model.fake_B, model.real_B):
        json_path = Path(json_path)
        with json_path.open("r") as f:
            meta = json.load(f)

        def get_mask(mask_type: str):
            mask_path = json_path.with_name(get_channel_file_from_metadata(
                meta, "CD3", "mask", mask_type=mask_type))
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = mask != 0
            return mask

        mask = get_mask("cells")

        l_real = label(mask, connectivity=1)
        num_cells_real = l_real.max()

        return {
            "metrics": {
                "real cells": num_cells_real,
                "real MIR": compute_mask_intensity_ratio(get_img(real_B), mask),
                "fake MIR": compute_mask_intensity_ratio(get_img(fake_B), mask)
            },
            "visuals": {
                "real_A": get_img(real_A),
                "fake_B": get_img(fake_B),
                "real_B": get_img(real_B),
                "mask": mask.astype(float)
            }
        }


def test_model(cfg: DictConfig, metric: str = "fake MIR"):
    model = create_model(cfg)
    model.setup(cfg)

    dataset = create_dataset(cfg)
    res = itertools.islice(
        map(partial(perform_test, model=model), dataset), 30)

    if metric is not None:
        res = reversed(sorted(res, key=lambda x: x["metrics"][metric]))
    res = list(res)

    def subplot(rows, cols, row, col):
        return plt.subplot(rows, cols, cols * row + col + 1)

    plt.figure(figsize=(12, len(res) * 2.1))
    plt.suptitle(f"Experiment: {cfg.name}")
    for row, r in enumerate(res):
        visuals = r["visuals"]
        metrics = r["metrics"]
        splot = partial(subplot, len(res), len(visuals) + 1)
        for col, (name, visual) in enumerate(visuals.items()):
            splot(row, col)
            plt.imshow(visual, cmap="gray")
            plt.title(name)
            plt.axis("off")
        splot(row, col + 1)
        for y, (k, v) in zip(np.linspace(.3, .7, len(metrics)), reversed(metrics.items())):
            plt.text(0., y, k)
            plt.text(1., y, f"{v:.3f}" if isinstance(v, float) else str(v))
        plt.axis("off")
    plt.savefig(Path(__file__).with_name(f"test_{cfg.name}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str, help="wandb run ID")
    args = parser.parse_args()
    cfg = load_run_cfg(args.run)
    test_model(cfg)
