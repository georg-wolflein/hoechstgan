from omegaconf import DictConfig
import torch
from .dataset import Dataset


class CustomDatasetDataLoader():

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset = Dataset(cfg)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            num_workers=int(cfg.dataset.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        if "inf" in str(self.cfg.dataset.max_size).lower():
            return len(self.dataset)
        else:
            return min(len(self.dataset), int(self.cfg.dataset.max_size))

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if "inf" not in str(self.cfg.dataset.max_size).lower() and \
                    i * self.cfg.dataset.batch_size >= self.cfg.dataset.max_size:
                break
            yield data


def create_dataset(cfg):
    data_loader = CustomDatasetDataLoader(cfg)
    dataset = data_loader.load_data()
    return dataset
