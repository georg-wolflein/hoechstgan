from pathlib import Path
import json
from PIL import Image
from omegaconf import DictConfig

from hoechstgan.data.split_dataset import get_dataset_splits

from .transforms import get_transform
from ..util.dataset import get_channel_file_from_metadata


class Dataset:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root = Path(self.cfg.dataset.data_root)
        self.json_paths = get_dataset_splits(cfg)[cfg.phase]

    def __getitem__(self, index):
        json_path = self.root / self.json_paths[index]

        with json_path.open("r") as f:
            metadata = json.load(f)

            def load_channel_img(ds_cfg):
                path = json_path.parent / \
                    get_channel_file_from_metadata(
                        metadata, **ds_cfg.props)
                img = Image.open(path).convert("RGB")
                return img

            A = load_channel_img(self.cfg.dataset.input)
            B = load_channel_img(self.cfg.dataset.output)

            A_transform = get_transform(self.cfg, self.cfg.dataset.input)
            B_transform = get_transform(self.cfg, self.cfg.dataset.output)

            A = A_transform(A)
            B = B_transform(B)

            return {"A": A, "B": B, "json_files": str(json_path)}

    def __len__(self):
        return len(self.json_paths)
