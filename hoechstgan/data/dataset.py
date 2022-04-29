from pathlib import Path
import json
from PIL import Image
from omegaconf import DictConfig

from .transforms import get_transform


def _get_channel_file(metadata: dict, channel_name: str, channel_type: str) -> str:
    for img in metadata["images"]:
        if img["channel"] == channel_name and img["type"] == channel_type:
            return img["file"]
    raise Exception(
        f"Did not find {channel_type=} and {channel_name=} in metadata ({metadata})")


class Dataset:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        root = Path(self.cfg.dataset.data_root)  # / cfg.training.phase
        self.json_paths = sorted(x for x in root.glob("*.json"))

    def __getitem__(self, index):
        json_path = self.json_paths[index]

        with json_path.open("r") as f:
            metadata = json.load(f)

            def load_channel_img(ds_cfg):
                path = json_path.parent / \
                    _get_channel_file(metadata, ds_cfg.channel, ds_cfg.mode)
                img = Image.open(path).convert("RGB")
                return img, str(path)

            A, path_A = load_channel_img(self.cfg.dataset.input)
            B, path_B = load_channel_img(self.cfg.dataset.output)

            A_transform = get_transform(self.cfg, self.cfg.dataset.input)
            B_transform = get_transform(self.cfg, self.cfg.dataset.output)

            A = A_transform(A)
            B = B_transform(B)

            return {"A": A, "B": B, "path_A": path_A, "path_B": path_B}

    def __len__(self):
        return len(self.json_paths)
