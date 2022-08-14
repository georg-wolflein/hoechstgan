from pathlib import Path
import json
from PIL import Image
from omegaconf import DictConfig
import numpy as np

from .transforms import get_transform
from ..util.dataset import get_channel_file_from_metadata


def get_json_paths(cfg: DictConfig) -> list:
    root = Path(cfg.dataset.data_root)

    split_dir = root / cfg.phase
    index_file = split_dir.joinpath("index.txt")
    if not index_file.exists():
        raise FileNotFoundError(
            f"{index_file} does not exist, fix by running index_dataset.py")
    paths = [split_dir.joinpath(x.strip())
             for x in index_file.open("r")]

    if cfg.dataset.shuffle:
        np.random.seed(42)
        paths = np.random.permutation(paths).tolist()

    print(f"Loaded {len(paths)} samples in {cfg.phase} dataset.")
    return paths


class Dataset:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root = Path(self.cfg.dataset.data_root)
        self.json_paths = get_json_paths(cfg)

    def __getitem__(self, index):
        json_path = self.json_paths[index]

        with json_path.open("r") as f:
            metadata = json.load(f)

            def load_channel_img(ds_cfg):
                path = json_path.parent / \
                    get_channel_file_from_metadata(
                        metadata, **ds_cfg.props)
                img = Image.open(path).convert("RGB")
                return img

            def load_and_transform_channel_img(ds_cfg):
                img = load_channel_img(ds_cfg)
                transform = get_transform(self.cfg, ds_cfg)
                return transform(img)

            assert "A" not in self.cfg.dataset.outputs

            imgs = {k: load_and_transform_channel_img(v)
                    for (k, v)
                    in {"A": self.cfg.dataset.input,
                        **self.cfg.dataset.outputs}.items()}

            return {**imgs, "json_files": str(json_path)}

    def __len__(self):
        return len(self.json_paths)
