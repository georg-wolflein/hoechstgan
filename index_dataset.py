import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def index_subdir(root: Path):
    with (root / "index.txt").open("w") as f:
        subdirs = [x for x in root.iterdir() if x.is_dir()]
        for subdir in (pbar := tqdm(subdirs)):
            pbar.set_description(
                f"Processing {subdir.relative_to(root.parent)}")
            if subdir.is_dir():
                i = 0
                for file in os.scandir(subdir):
                    if file.is_file() and file.name.endswith(f".json"):
                        f.write(subdir.name + "/" + file.name + "\n")
                        i += 1
                        if i % 10 == 0:
                            pbar.set_description(
                                f"Processing {subdir.relative_to(root.parent)} ({i} files)")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def index(cfg: DictConfig):
    root = Path(cfg.dataset.data_root)
    index_subdir(root / "train")
    index_subdir(root / "test")


if __name__ == "__main__":
    index()
