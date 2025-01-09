from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
from hydra import compose, initialize
# from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset
# from src.data.precip_dataloader_inference import (RainfallSpecifiedInference, PreSavedPrecipDataset,
#                                                   xarray_collate_fn, do_nothing_collate_fn)
from src.data.depth import get_dataset, DatasetMode
from src.utils.event.depth_transform import DepthNormalizerBase, get_depth_normalizer

class EventDataModule(LightningDataModule):
    """`LightningDataModule` for the precipitation dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, data_config: dict, augmentation_args: dict, depth_transform_args: dict,
                 batch_size: int, num_workers: int, pin_memory: bool, seed: int,
                 *args, **kwargs) -> None:
        super().__init__()
        self.data_config = data_config
        self.save_hyperparameters(logger=False, ignore=("data_config",))
        # to be defined elsewhere
        self.loader_generator = None
        self.train_dataset = None
        self.train_loader = None
        return


    def setup(self, stage: Optional[str] = None):
        self.loader_generator = torch.Generator().manual_seed(self.hparams.seed)
        # train dataset
        self.train_dataset = get_dataset(
            self.data_config.train,
            base_data_dir= self.data_config.base_dir,
            mode=DatasetMode.TRAIN,
            augmentation_args= self.hparams.augmentation_args,
            depth_transform=self.hparams.depth_transform_args,)
         # TODO validation dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.data_config.train.name == 'mixed':
            raise NotImplementedError("Mixed datasets are not supported yet.")
        else:
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                generator=self.loader_generator,)
        return self.train_loader


if __name__ == '__main__':  # debug
    with initialize(version_base=None, config_path="../../configs/data", job_name="evaluation"):
        config = compose(config_name="event_carla")
    data_module = EventDataModule(data_config=config.data_config, augmentation_args=config.augmentation,
                                  depth_transform_args=config.depth_transform, batch_size=config.batch_size,
                                  num_workers=config.num_workers, pin_memory=config.pin_memory, seed=config.seed)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    print(f"Train DataLoader initialized with {len(train_loader)} batches.")
    # Get the first batch
    first_batch = next(iter(train_loader))
    print(f"First batch: {first_batch}")
