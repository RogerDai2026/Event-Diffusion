# this is the mixed training file for multiple dataset, providing sequential training( one epoch for multiple datasets )

from typing import Any, Dict, Optional, Tuple, List, Union
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from src.data.depth import get_dataset, DatasetMode
from src.utils.event.depth_transform import get_depth_normalizer


class EventDataModule(LightningDataModule):
    def __init__(self, data_config: dict, augmentation_args: dict, depth_transform_args: dict,
                 batch_size: int, num_workers: int, pin_memory: bool, seed: int, *args, **kwargs) -> None:
        super().__init__()
        self.data_config = data_config
        self.save_hyperparameters(logger=False, ignore=("data_config",))

        self.loader_generator: Optional[torch.Generator] = None
        self.depth_transform = None

        # single (default)
        self.train_dataset: Optional[Dataset] = None

        # sequential mode
        self._mode: str = "single"               # "single" (default) or "sequential"
        self._train_datasets: Optional[List[Dataset]] = None
        self._dl_build_count: int = 0            # increments each time train_dataloader() is called

        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    # ---------- helpers ----------
    def _build_one(self, cfg, mode: DatasetMode) -> Dataset:
        return get_dataset(
            cfg,
            base_data_dir=self.data_config.base_dir,
            mode=mode,
            augmentation_args=self.hparams.augmentation_args,
            depth_transform=self.depth_transform,
            io_args=self.data_config.io_args,
        )

    # ---------- Lightning hooks ----------
    def setup(self, stage: Optional[str] = None):
        self.loader_generator = torch.Generator().manual_seed(self.hparams.seed)
        self.depth_transform = get_depth_normalizer(cfg_normalizer=self.hparams.depth_transform_args)

        # --- TRAIN ---
        # Default: single dataset (your current config)
        # Optional: train.name == "sequential" and train.datasets: [cfgA, cfgB, ...]
        train_cfg = self.data_config.train
        if getattr(train_cfg, "name", None) == "sequential":
            self._mode = "sequential"
            ds_cfgs = list(train_cfg.datasets)
            self._train_datasets = [self._build_one(c, DatasetMode.TRAIN) for c in ds_cfgs]
        else:
            self._mode = "single"
            self.train_dataset = self._build_one(train_cfg, DatasetMode.TRAIN)

        # --- VAL ---
        self.val_dataset = self._build_one(self.data_config.val, DatasetMode.EVAL)

        # --- TEST (optional) ---
        if hasattr(self.data_config, "test") and self.data_config.test is not None:
            self.test_dataset = self._build_one(self.data_config.test, DatasetMode.EVAL)

        self.print_dataset_stats()

    def print_dataset_stats(self):
        print('-------------- Dataset Statistics --------------------')
        if self._mode == "single":
            print(f"Train dataset size: {len(self.train_dataset) if self.train_dataset else 0}")
        else:
            total = sum(len(d) for d in (self._train_datasets or []))
            details = ", ".join(str(len(d)) for d in (self._train_datasets or []))
            print(f"Train (sequential) total: {total} | per-ds: [{details}]")
        print(f"Validation dataset size: {len(self.val_dataset) if self.val_dataset else 0}")
        print(f"Test dataset size: {len(self.test_dataset) if self.test_dataset else 0}")
        print('------------------------------------------------------')

    # ---------- DataLoaders ----------
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._mode == "single":
            ds = self.train_dataset
        else:
            # IMPORTANT: set Trainer.reload_dataloaders_every_n_epochs=1 so this advances each epoch
            assert self._train_datasets and len(self._train_datasets) > 0
            idx = self._dl_build_count % len(self._train_datasets)
            ds = self._train_datasets[idx]
            self._dl_build_count += 1
            print(f"[EventDataModule] SEQUENTIAL epoch → using train dataset index {idx} (size={len(ds)})")

        return DataLoader(
            dataset=ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            generator=self.loader_generator,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            generator=self.loader_generator,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            print("No test dataset specified, using validation dataset for testing")
            return self.val_dataloader()
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            generator=self.loader_generator,
        )
