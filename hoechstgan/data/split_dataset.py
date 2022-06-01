import typing
from omegaconf import DictConfig
from pathlib import Path
import json
import numpy as np

from hoechstgan.util.fileutil import find, get_filename


def get_dataset_splits(cfg: DictConfig) -> typing.Dict[str, list]:
    split_total = sum(cfg.dataset.split.values())
    splits_weights = {
        k: v / split_total for (k, v) in cfg.dataset.split.items()
    }
    root = Path(cfg.dataset.data_root)
    args = {k: f"{v:.3f}" for (k, v) in splits_weights.items()}
    try:
        with next(find(root, "split", "txt", **args)).open("r") as f:
            splits = json.load(f)
            print("Reusing existing split...")
    except StopIteration:
        print("Generating new split...")
        split_path = root / get_filename("split", "txt", **args)
        json_paths = sorted(x.name for x in root.glob("*.json"))
        np.random.seed(42)
        json_paths = np.random.permutation(json_paths)
        sample_sizes = (np.array(list(splits_weights.values()))
                        * len(json_paths)).astype(np.int32)
        splits = dict(zip(splits_weights.keys(),
                      map(np.ndarray.tolist, np.split(json_paths, np.cumsum(sample_sizes)[:-1]))))
        with split_path.open("w") as f:
            json.dump(splits, f)
    print(f"{sum(map(len, splits.values()))} samples were split into {', '.join(f'{k}={len(v)}' for (k, v) in splits.items())}.")
    return splits
