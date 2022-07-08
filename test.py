
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
from tqdm import tqdm
import pandas as pd
import torch

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


def perform_test(data, model, cfg, latent_substitutions=None):
    model.set_input(data)
    model.test()

    outputs = cfg.dataset.outputs
    real_outputs = {i: getattr(model, f"real_{i}") for i in outputs}
    fake_outputs = {i: getattr(model, f"fake_{i}") for i in outputs}

    if latent_substitutions:
        model.test(latent_substitutions=latent_substitutions)
        fake_outputs_sub = {i: getattr(model, f"fake_{i}")
                            for i in outputs}

    for i, (json_path, real_A) in enumerate(zip(data["json_files"], model.real_A)):
        json_path = Path(json_path)
        metrics = {"file": json_path.stem}
        visuals = {"real_A": get_img(real_A)}
        with json_path.open("r") as f:
            meta = json.load(f)

        for out, out_cfg in outputs.items():
            real_X = get_img(real_outputs[out][i])
            fake_X = get_img(fake_outputs[out][i])
            channel = {
                "Cy3": "CD3",
                "Cy5": "CD8"
            }[out_cfg.props.channel]

            # Get mask
            mask_path = json_path.with_name(get_channel_file_from_metadata(
                meta, channel=channel, mode="mask", mask_type="cells"))
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = mask != 0

            l_real = label(mask, connectivity=1)
            num_cells_real = l_real.max()

            mir_real = compute_mask_intensity_ratio(real_X, mask)
            mir_fake = compute_mask_intensity_ratio(fake_X, mask)
            mir_ratio = mir_fake / mir_real if mir_real > 0 else np.inf

            metrics.update({
                f"{channel}+ cells": num_cells_real,
                f"{channel} real MIR": mir_real,
                f"{channel} fake MIR": mir_fake,
                f"{channel} relative MIR": mir_ratio
            })
            visuals.update({
                f"fake_{out}": fake_X,
                **({f"fake_{out}_sub": get_img(fake_outputs_sub[out][i])} if latent_substitutions else {}),
                f"real_{out}": real_X,
                f"mask_{out}": mask.astype(float)
            })

        yield {
            "metrics": metrics,
            "visuals": visuals
        }


def setup_model(cfg: DictConfig, overrides: list = [], is_train: bool = False, verbose: bool = True, gpus: list = None):
    cfg = cfg.copy()
    cfg.is_train = is_train
    cfg.phase = "train" if is_train else "test"
    cfg.verbose = verbose
    if gpus is not None:
        cfg.gpus = gpus
    model = create_model(cfg)
    model.setup(cfg)
    return model


def test_model(cfg: DictConfig, metric="CD8 relative MIR"):
    model_stats = dict()

    # Load model in train mode to gather some stats
    model = setup_model(cfg, is_train=True, verbose=False, gpus=[])
    model_stats.update({f"num_params_{k}": v
                        for (k, v) in {
                            **model.num_parameters_by_net(),
                            "total": model.num_parameters
                        }.items()})
    del model

    def substitute(latent):
        return [torch.zeros_like(x) for x in latent]
        return [torch.normal(0., 1., x.shape).to(model.device) for x in latent]
    latent_substitutions = {"latent_B": substitute}

    # Load model again in test mode
    model = setup_model(cfg, is_train=False, verbose=True)
    dataset = create_dataset(cfg)
    res_generator = itertools.chain.from_iterable(perform_test(data, model=model, cfg=cfg, latent_substitutions=latent_substitutions)
                                                  for data in dataset)
    res = itertools.islice(res_generator, 30)

    if metric is not None:
        res = reversed(sorted(res, key=lambda x: x["metrics"][metric]))
    res = list(res)

    def subplot(rows, cols, row, col):
        return plt.subplot(rows, cols, cols * row + col + 1)

    plt.figure(figsize=(len(res[0]["visuals"]) * 2.2 + 6,
                        (len(res) + 5) * 2.2))
    plt.suptitle(f"Experiment: {cfg.name}")
    for row, r in enumerate(res):
        visuals = r["visuals"]
        metrics = r["metrics"]
        splot = partial(subplot, len(res), len(visuals) + 5)
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

    metrics = [r["metrics"]
               for r in tqdm(itertools.chain(res, res_generator), total=len(dataset))]
    df = pd.DataFrame(metrics)
    df = df.replace([np.inf, -np.inf], np.nan)
    print(df.describe())
    df.to_csv(f"test_{cfg.name}_{cfg.wandb_id}_metrics.csv", index=False)

    model_stats.update({f"metric_mean_{k}": v for (k, v) in df.mean().items()})
    df_stats = pd.DataFrame(model_stats.items(), columns=["key", "value"])
    df_stats.to_csv(f"test_{cfg.name}_{cfg.wandb_id}_stats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str, help="wandb run ID")
    args = parser.parse_args()
    cfg = load_run_cfg(args.run)
    initialize(config_path="conf", version_base="1.2")
    test_cfg = compose("test_overrides.yaml")
    cfg = OmegaConf.merge(cfg, test_cfg)
    test_model(cfg)
