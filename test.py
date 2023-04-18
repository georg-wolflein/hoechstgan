
from collections import defaultdict
import shutil
from omegaconf import DictConfig, ListConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageColor
import itertools
from functools import partial
from skimage.morphology import label
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import argparse
from hydra import initialize, compose
from tqdm import tqdm
import pandas as pd
import torch
import wandb

from hoechstgan.data import create_dataset
from hoechstgan.models import create_model
from hoechstgan.util.dataset import get_channel_file_from_metadata
from hoechstgan.util.logging import WANDB_ENTITY, WANDB_PROJECT, get_api

OUT_DIR = Path(__file__).parent / "vis"
OUT_DIR.mkdir(exist_ok=True)

CHANNELS = {
    "Cy3": "CD3",
    "Cy5": "CD8",
    "CD3": "CD3",
    "CD8": "CD8"
}

CFG_DEFAULTS = {
    "generator.composites": ListConfig([]),
    "loss.generator.coefficient": 1.,
    "loss.discriminator.coefficient": 1.,
    "discriminator.type": "separate",
    "generator.dropout_eval_mode": "dropout"
}


def fix_cfg(cfg: DictConfig) -> DictConfig:
    for key, value in CFG_DEFAULTS.items():
        *ancestors, child = key.split(".")
        item = cfg
        for ancestor in ancestors:
            if ancestor not in item:
                item[ancestor] = DictConfig({})
            item = item[ancestor]
        if child not in item:
            item[child] = value
    return cfg


def load_run_cfg(run_id: str) -> DictConfig:
    api = get_api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    cfg = DictConfig(run.config)
    cfg.wandb_id = run_id
    cfg = fix_cfg(cfg)
    return cfg, run


def compute_mask_intensity_ratio(img, mask) -> float:
    if (mask == 0).all():
        return np.nan, np.nan, np.nan
    numerator = img[mask].mean()
    denominator = img[~mask].mean()
    return numerator / denominator if denominator != 0 else 0, numerator, denominator


def get_mir_stats(real_X, fake_X, mask) -> dict:
    mir_real, mir_real_num, mir_real_den = compute_mask_intensity_ratio(
        real_X, mask)
    mir_fake, mir_fake_num, mir_fake_den = compute_mask_intensity_ratio(
        fake_X, mask)
    mir_ratio = mir_fake / mir_real if mir_real != 0 else np.inf

    fake_X_norm = fake_X + real_X.mean() - fake_X.mean()
    mir_fake_norm, mir_fake_norm_num, mir_fake_norm_den = compute_mask_intensity_ratio(
        fake_X_norm, mask)
    mir_ratio_norm = mir_fake_norm / mir_real if mir_real != 0 else np.inf
    # real_X_norm = real_X - real_X.mean()
    # fake_X_norm = fake_X - fake_X.mean()
    # mir_real_norm, mir_real_norm_num, mir_real_norm_den = compute_mask_intensity_ratio(
    #     real_X_norm, mask)
    # mir_fake_norm, mir_fake_norm_num, mir_fake_norm_den = compute_mask_intensity_ratio(
    #     fake_X_norm, mask)
    # mir_ratio_norm = mir_fake_norm / mir_real_norm if mir_real_norm != 0 else np.inf

    return {
        f"real MIR": mir_real,
        f"real MIR numerator": mir_real_num,
        f"real MIR denominator": mir_real_den,
        f"fake MIR": mir_fake,
        f"fake MIR numerator": mir_fake_num,
        f"fake MIR denominator": mir_fake_den,
        f"fake MIR norm": mir_fake_norm,
        f"fake MIR norm numerator": mir_fake_norm_num,
        f"fake MIR norm denominator": mir_fake_norm_den,
        f"relative MIR": mir_ratio,
        f"relative MIR norm": mir_ratio_norm,
        f"log relative MIR": np.log(mir_ratio),
        f"log relative MIR norm": np.log(mir_ratio_norm),
        f"real fake PSNR": peak_signal_noise_ratio(real_X, fake_X, data_range=1.),
        f"real fake SSIM": structural_similarity(real_X, fake_X, data_range=1.),
        f"real mask PSNR": peak_signal_noise_ratio(mask, real_X, data_range=1.),
        f"real mask SSIM": structural_similarity(mask, real_X, data_range=1.),
        f"fake mask PSNR": peak_signal_noise_ratio(mask, fake_X, data_range=1.),
        f"fake mask SSIM": structural_similarity(mask, fake_X, data_range=1.)
    }


def get_img(tensor):
    return (tensor.cpu().numpy().squeeze() + 1) / 2.


def compute_percent_tp_cells(img, mask):
    thresholds = np.arange(0.1, 1, 0.1)
    L = label(mask, connectivity=1)
    L, img = L.flatten(), img.flatten()
    num_cells = L.max()
    if num_cells == 0:
        return zip(thresholds, [1.] * len(thresholds))
    masks = np.expand_dims(L, -1) == np.arange(1, L.max() + 1)
    pixels_per_cell = masks.sum(axis=0)
    mult_mask = np.expand_dims(img, -1) * masks
    mean_intensity_per_cell = mult_mask.sum(axis=0) / pixels_per_cell
    thresholds_mask = mean_intensity_per_cell >= np.expand_dims(thresholds, -1)
    num_cells_per_threshold = np.sum(thresholds_mask, axis=1)
    return zip(thresholds, num_cells_per_threshold / num_cells)


def perform_test(data, model, cfg, latent_substitutions=None, compute_tp_thresholds=False, do_input_substitution=False):
    model.set_input(data)
    model.test()

    outputs = cfg.dataset.outputs
    real_outputs = {i: getattr(model, f"real_{i}") for i in outputs}
    fake_outputs = {i: getattr(model, f"fake_{i}") for i in outputs}
    fake_outputs_sub = dict()

    if latent_substitutions:
        model.test(latent_substitutions=latent_substitutions)
        fake_outputs_sub["latent"] = {i: getattr(model, f"fake_{i}")
                                      for i in outputs}

    if do_input_substitution:
        key = "composite_B" if any(c["to"] == "composite_B"
                                   for c in cfg.generator.composites) else "fake_B"
        for mode in "zeros", "normal", "real", "other_real":
            if mode == "zeros":
                sub = torch.zeros_like(model.fake_B)
            elif mode == "normal":
                sub = torch.randn_like(model.fake_B)
            elif mode == "real":
                sub = model.real_B
            elif mode == "other_real":
                # Permute batch (akin to random shuffling, because dataset is already shuffled)
                indices = torch.arange(model.real_B.shape[0]) + 1
                indices = indices % model.real_B.shape[0]
                sub = model.real_B[indices]
            input_substitutions = {
                key: sub
            }
            model.test(input_substitutions=input_substitutions)
            fake_outputs_sub[f"input_{mode}"] = {i: getattr(model, f"fake_{i}")
                                                 for i in outputs}

    for i, (json_path, real_A) in enumerate(zip(data["json_files"], model.real_A)):
        json_path = Path(json_path)
        metrics = {"file": json_path.stem}
        visuals = {"real_A": get_img(real_A)}
        with json_path.open("r") as f:
            meta = json.load(f)

        def load_mask(channel):
            mask_path = json_path.with_name(get_channel_file_from_metadata(
                meta, channel=channel, mode="mask", mask_type="cells"))
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = mask != 0
            return mask

        # Load Hoechst mask
        visuals["mask_A"] = load_mask("Hoechst").astype(float)

        for out, out_cfg in outputs.items():
            real_X = get_img(real_outputs[out][i])
            fake_X = get_img(fake_outputs[out][i])
            channel = CHANNELS[out_cfg.props.channel]

            # Get mask
            mask = load_mask(channel)

            l_real = label(mask, connectivity=1)
            num_cells_real = l_real.max()

            metrics.update({
                f"{channel}+ cells": num_cells_real,
                **{f"{channel} {k}": v
                   for k, v in get_mir_stats(real_X, fake_X, mask).items()}
            })

            for sub_mode, sub_outputs in fake_outputs_sub.items():
                sub_fake_X = get_img(sub_outputs[out][i])
                metrics.update({f"sub_{sub_mode} {channel} {k}": v
                                for k, v in get_mir_stats(real_X, sub_fake_X, mask).items()
                                })
            if compute_tp_thresholds:
                for X_name, X in {"real": real_X, "fake": fake_X}.items():
                    metrics.update({
                        f"{channel}+ cells {X_name} TP @{threshold:.1f}": percent_identified
                        for (threshold, percent_identified) in compute_percent_tp_cells(X, mask)
                    })
            visuals.update({
                f"fake_{out}": fake_X,
                **({f"fake_{out}_sub_{sub_mode}": get_img(sub_outputs[out][i])
                    for sub_mode, sub_outputs in fake_outputs_sub.items()}),
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


def load_dataset(cfg: DictConfig, phase: str = None):
    cfg = cfg.copy()
    if phase is not None:
        cfg.phase = phase
    return create_dataset(cfg)


def get_fig(fig: plt.Figure) -> wandb.Image:
    img = wandb.Image(fig)
    fig.gca().clear()
    fig.clear()
    return img


def test_model(
    cfg: DictConfig,
    run: wandb.wandb_sdk.wandb_run.Run,
    metric="CD3 relative MIR",
    do_latent_substitution: bool = False,
    do_input_substitution: bool = False,
    save_sample_patches: bool = False,
    max_dataset_size: int = 150000,
    filename_suffix: str = None,
    update_wandb_stats: bool = None,
    epoch: str = None
):
    # Determine epoch
    if epoch is not None:
        if epoch == "latest":
            cfg.load_checkpoint = "latest"
        else:
            # Assume it's an int
            epoch = int(epoch)
            cfg.initial_epoch = epoch
            cfg.load_checkpoint = "epoch"
    print("Running test for epoch", epoch)

    summary_stats = dict()
    # Only update stats if not sub/samples
    if update_wandb_stats is None:
        update_wandb_stats = not do_latent_substitution and not save_sample_patches
    # update_wandb_stats = False
    if filename_suffix is None:
        filename_suffix = "sub" if do_latent_substitution else ""

    # Load model in train mode to gather some stats
    model = setup_model(cfg, is_train=True, verbose=False, gpus=[])
    summary_stats.update({f"num_params_{k}": v
                          for (k, v) in {
                              **model.num_parameters_by_net(),
                              "total": model.num_parameters
                          }.items()})
    del model

    latent_substitutions = {}
    if do_latent_substitution:
        def substitute(latent):
            return [torch.zeros_like(x) for x in latent]
            return [torch.normal(0., 1., x.shape).to(model.device) for x in latent]
        latent_substitutions = {"latent_B": substitute}

    # Load model again in test mode
    model = setup_model(cfg, is_train=False, verbose=True)

    for phase in ("test", "train"):
        dataset = load_dataset(cfg, phase=phase)
        dataset_size = len(dataset)
        # Test set is ~150000, so we choose only 150000 samples
        dataset_size = min(dataset_size, max_dataset_size)
        # dataset_size = 1000
        res_generator = itertools.chain.from_iterable(perform_test(data, model=model, cfg=cfg, latent_substitutions=latent_substitutions, do_input_substitution=do_input_substitution)
                                                      for data in dataset)
        res_generator = itertools.islice(res_generator, dataset_size)
        res = itertools.islice(res_generator, 30)

        if metric is not None:
            res = reversed(sorted(res, key=lambda x: x["metrics"][metric]))
        res = list(res)

        def subplot(rows, cols, row, col):
            return plt.subplot(rows, cols, cols * row + col + 1)

        fig = plt.figure(figsize=(len(res[0]["visuals"]) * 2.2 + 6,
                                  (len(res) + 5) * 2.2))
        plt.suptitle(f"Experiment: {cfg.name} ({phase} dataset)")
        for row, r in enumerate(res):
            visuals = r["visuals"]
            metrics = r["metrics"]
            metrics["row"] = row
            splot = partial(subplot, len(res), len(visuals) + 5)
            for col, (name, visual) in enumerate(visuals.items()):
                splot(row, col)
                plt.imshow(visual, cmap="gray")
                plt.title(name)
                plt.axis("off")
            splot(row, col + 1)
            relevant_metrics = {k: v
                                for k, v in metrics.items() if
                                "relative" in k or k in ("file", "row")}
            for y, (k, v) in zip(np.linspace(.1, .9, len(relevant_metrics)), reversed(relevant_metrics.items())):
                plt.text(0., y, k)
                plt.text(1., y, f"{v:.3f}" if isinstance(
                    v, float) else str(v))
            plt.axis("off")
        if not save_sample_patches:
            plt.savefig(
                OUT_DIR / f"{cfg.name}_{cfg.wandb_id}_{phase}_{filename_suffix}_vis.png")
            # run.summary["sample_vis"] = get_fig(fig)

        if save_sample_patches:
            visuals = defaultdict(list)
            metrics = defaultdict(list)
            WHITE = "#ffffff"
            RED = "#c61a09"
            BLUE = "#0da2ff"
            samples_out_dir = OUT_DIR / "samples" / \
                f"{cfg.name}_{cfg.wandb_id}_{phase}_{epoch}"
            shutil.rmtree(samples_out_dir, ignore_errors=True)
            samples_out_dir.mkdir(parents=True, exist_ok=True)
            for r in itertools.islice(res, samples):
                for dk, dv in ("visuals", visuals), ("metrics", metrics):
                    if dk == "visuals":
                        mask_A = r[dk]["mask_A"]
                        mask_B = r[dk]["mask_B"]
                        mask_C = r[dk]["mask_C"]
                        img = np.zeros((*mask_A.shape, 3), dtype=np.uint8)
                        img[mask_A == 1] = ImageColor.getrgb(WHITE)
                        img[mask_B == 1] = ImageColor.getrgb(BLUE)
                        img[mask_C == 1] = ImageColor.getrgb(RED)
                        r[dk]["cells"] = img
                    for k, v in r[dk].items():
                        dv[k].append(v)
            for k, v in visuals.items():
                for i, patch in enumerate(v):
                    img = patch
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(
                        samples_out_dir / f"patch_{i:02d}_{k}.png")
            df = pd.DataFrame(metrics)
            df.to_csv(samples_out_dir / "metrics.csv")
            plt.savefig(samples_out_dir / "vis.png")
            return

        metrics = [r["metrics"]
                   for r in tqdm(itertools.chain(res, res_generator), total=dataset_size, desc=f"processing {phase} dataset")]
        df = pd.DataFrame(metrics)
        df = df.replace([np.inf, -np.inf], np.nan)
        print(df.describe())
        df.to_csv(
            OUT_DIR / f"{cfg.name}_{cfg.wandb_id}_{epoch}_{phase}_{filename_suffix}_metrics.csv", index=False)

        summary_stats.update(
            {f"{phase} mean {k}": v for (k, v) in df.mean().items()})
        summary_stats.update(
            {f"{phase} std {k}": v for (k, v) in df.std(ddof=0).items()})
    if update_wandb_stats:
        run.summary.update(summary_stats)
        run.update()
    df_stats = pd.DataFrame(summary_stats.items(),
                            columns=["key", "value"])
    df_stats.to_csv(
        OUT_DIR / f"{cfg.name}_{cfg.wandb_id}_{epoch}__{filename_suffix}_stats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str, help="wandb run ID")
    parser.add_argument("--latentsub",
                        action="store_true",
                        help="perform latent substition")
    parser.add_argument("--inputsub",
                        action="store_true",
                        help="perform input substition")
    parser.add_argument("--samples", action="store_true",
                        help="save sample patches")
    parser.add_argument("--size", type=int, default=150000,
                        help="max dataset size")
    parser.add_argument("--gpus", type=str,
                        help="comma-separated list of gpus to use",
                        default=None)
    parser.add_argument("--dropout", action="store_true",
                        help="override to force dropout", dest="dropout", default=None)
    parser.add_argument("--no-dropout", action="store_false",
                        help="override to force no dropout", dest="dropout")
    parser.add_argument("--no-update", action="store_false",
                        help="don't update wandb stats", dest="update_wandb_stats")
    parser.add_argument("--dropout-eval-mode", type=str,
                        help="override dropout eval mode", choices=["identity", "dropout", "average"], default=None, dest="dropout_eval_mode")
    parser.add_argument("--suffix", type=str, default=None,
                        help="suffix for output files")
    parser.add_argument("--epoch", type=str, default=None,
                        help="epoch to evaluate")
    parser.set_defaults(update_wandb_stats=None, dropout=None)
    args = parser.parse_args()
    cfg, run = load_run_cfg(args.run)
    initialize(config_path="conf", version_base="1.2")
    test_cfg = compose("test_overrides.yaml")
    cfg = OmegaConf.merge(cfg, test_cfg)
    if args.gpus:
        cfg.gpus = [int(x) for x in args.gpus.split(",")]
        print("Overriding gpus to", cfg.gpus)
        cfg.dataset.num_threads = len(cfg.gpus) * 2
        print("Overriding dataset.num_threads to", cfg.dataset.num_threads)
    if args.dropout is not None:
        print("Overriding dropout to", args.dropout)
        cfg.generator.dropout = args.dropout
    if args.dropout_eval_mode is not None:
        print("Overriding dropout_eval_mode to", args.dropout_eval_mode)
        cfg.generator.dropout_eval_mode = args.dropout_eval_mode

    # Construct function to run test
    run_test = partial(test_model,
                       cfg, run,
                       metric=f"{CHANNELS[cfg.dataset.outputs.B.props.channel]} relative MIR",
                       do_latent_substitution=args.latentsub,
                       do_input_substitution=args.inputsub,
                       save_sample_patches=args.samples,
                       max_dataset_size=args.size,
                       update_wandb_stats=args.update_wandb_stats,
                       filename_suffix=args.suffix)

    # Run test
    if args.epoch is not None and args.epoch.startswith("<"):
        # Run test for all epochs up to the specified one
        print("Running test for all epochs up to", args.epoch)
        until_epoch = int(args.epoch[1:])
        for epoch in range(until_epoch):
            run_test(epoch=epoch)
    else:
        # Run test for the specified epoch
        run_test(epoch=args.epoch)
