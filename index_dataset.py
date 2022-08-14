import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def index_subdir(root: Path):
    with (root / "index.txt").open("w") as f:
        subdirs = [x for x in root.iterdir() if x.is_dir()]
        for i, subdir in enumerate(subdirs, 1):
            if subdir.is_dir():
                for file in tqdm((f for f in os.scandir(subdir)
                                  if f.is_file() and f.name.endswith(".json")),
                                 desc=f"Processing {subdir.name} (#{i}/{len(subdirs)} in {root.name})",
                                 unit="files"):
                    f.write(subdir.name + "/" + file.name + "\n")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def index(cfg: DictConfig):
    root = Path(cfg.dataset.data_root)
    index_subdir(root / "train")
    index_subdir(root / "test")


if __name__ == "__main__":
    index()
