from PIL import Image
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from hoechstgan.dataset import Dataset


@hydra.main(config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:
    ds = Dataset(cfg)
    img = ds[0]["B"].numpy().squeeze()
    img = img - img.min()
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).convert("RGB").save("/app/test.png")


if __name__ == "__main__":
    app()
