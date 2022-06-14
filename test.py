
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from PIL import Image
import itertools
from functools import partial
from skimage.morphology import label
import argparse
from hydra import initialize, compose

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


def perform_test(data, model, cfg):
    model.set_input(data)
    model.test()

    outputs = cfg.dataset.outputs
    real_outputs = {i: getattr(model, f"real_{i}") for i in outputs}
    fake_outputs = {i: getattr(model, f"fake_{i}") for i in outputs}

    for i, (json_path, real_A) in enumerate(zip(data["json_files"], model.real_A)):
        metrics = {}
        visuals = {"real_A": get_img(real_A)}
        json_path = Path(data["json_files"][0])
        with json_path.open("r") as f:
            meta = json.load(f)

        for out, out_cfg in outputs.items():
            real_X = real_outputs[out][i]
            fake_X = fake_outputs[out][i]
            channel = {
                "Cy3": "CD3",
                "Cy5": "CD8"
            }[out_cfg.props.channel]

            def get_mask():
                mask_path = json_path.with_name(get_channel_file_from_metadata(
                    meta, channel=channel, mode="mask", mask_type="cells"))
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                mask = mask != 0
                return mask

            mask = get_mask()

            l_real = label(mask, connectivity=1)
            num_cells_real = l_real.max()

            mir_real = compute_mask_intensity_ratio(get_img(real_X), mask)
            mir_fake = compute_mask_intensity_ratio(get_img(fake_X), mask)
            mir_ratio = mir_fake / mir_real if mir_real > 0 else np.inf

            metrics.update({
                f"{channel}+ cells": num_cells_real,
                f"{channel} real MIR": mir_real,
                f"{channel} fake MIR": mir_fake,
                f"{channel} relative MIR": mir_ratio
            })
            visuals.update({
                f"fake_{out}": get_img(fake_X),
                f"real_{out}": get_img(real_X),
                f"mask_{out}": mask.astype(float)
            })

        yield {
            "metrics": metrics,
            "visuals": visuals
        }


def test_model(cfg: DictConfig, metric="CD8 relative MIR"):
    model = create_model(cfg)
    model.setup(cfg)

    dataset = create_dataset(cfg)
    res = itertools.chain(*[perform_test(data, model=model, cfg=cfg)
                          for data in itertools.islice(dataset, 30)])

    if metric is not None:
        res = reversed(sorted(res, key=lambda x: x["metrics"][metric]))
    res = list(res)

    def subplot(rows, cols, row, col):
        return plt.subplot(rows, cols, cols * row + col + 1)

    plt.figure(figsize=(len(res[0]["visuals"]) * 2.2 + 3, len(res) * 2.2))
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
        for y, (k, v) in zip(np.linspace(.1, .9, len(metrics)), reversed(metrics.items())):
            plt.text(0., y, k)
            plt.text(1., y, f"{v:.3f}" if isinstance(v, float) else str(v))
        plt.axis("off")
    plt.savefig(Path(__file__).with_name(
        f"test_{cfg.name}_{cfg.wandb_id}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str, help="wandb run ID")
    args = parser.parse_args()
    cfg = load_run_cfg(args.run)
    initialize(config_path="conf", version_base="1.2")
    test_cfg = compose("test_overrides.yaml")
    cfg = OmegaConf.merge(cfg, test_cfg)
    test_model(cfg)
