from typing import Any, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import torch
from lightning import LightningModule

class GenericE2DModule(LightningModule, ABC):

    def _fill_missing_keys(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fill missing keys in the batch dictionary. In this example,
        if 'depth_raw_norm' is not provided, compute it using the
        datamodule’s depth_transform method.
        """
        if 'depth_raw_norm' not in batch_dict:
            # Here we assume that the datamodule has a `depth_transform` callable.
            batch_dict['depth_raw_norm'] = self.trainer.datamodule.depth_transform(batch_dict['depth_raw_linear'])
        return batch_dict

    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the condition and target tensors from the batch dictionary.
        This implementation uses a `scaler` which must be provided by subclasses.
        """
        # The scaler method or property is expected to be implemented by the child.
        # condition = self.scaler(batch_dict['rgb_norm'])
        condition = batch_dict['rgb_norm']
        y = batch_dict['depth_raw_norm']
        return condition, y

    @abstractmethod
    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Runs sampling fn when given condition tensor. No further trimming or processing is performance.
        This is intended to use during validation to gather quick samples.
        """
        pass