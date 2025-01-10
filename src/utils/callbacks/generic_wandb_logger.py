import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.helper import wandb_display_grid, cm_, visualize_batch
from src.utils.metrics import calc_mae, calc_bias


class GenericLogger(Callback, ABC):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_log_param_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False, check_freq_via: str = 'global_step',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 report_sample_metrics: bool = True):
        """
        Callback to log images, scores and parameters to wandb.
        :param train_log_img_freq: frequency to log images. Set to -1 to disable
        :param train_log_score_freq: frequency to log scores. Set to -1 to disable
        :param train_log_param_freq: frequency to log parameters. Set to -1 to disable
        :param show_samples_at_start: whether to log samples at the start of training (likely during sanity check)
        :param show_unconditional_samples: whether to log unconditional samples. Deprecated.
        :param check_freq_via: whether to check frequency via 'global_step' or 'epoch'
        :param enable_save_ckpt: whether to save checkpoint
        :param add_reference_artifact: whether to add the checkpoint as a reference artifact
        :param report_sample_metrics: whether to report sample metrics
        """
        super().__init__()

        self.check_freq_via = check_freq_via
        assert self.check_freq_via in ['global_step', 'epoch']
        self.freqs = {'img': train_log_img_freq, 'score': train_log_score_freq, 'param': train_log_param_freq}
        self.next_log_idx = {'img': 0 if show_samples_at_start else train_log_img_freq - 1, 'score': 0, 'param': 0}
        self.show_unconditional_samples = show_unconditional_samples
        self.enable_save_ckpt = enable_save_ckpt
        self.add_reference_artifact = add_reference_artifact
        self.report_sample_metrics = report_sample_metrics

        if self.show_unconditional_samples:
            raise NotImplementedError('Unconditional samples not implemented yet.')

        # to be defined elsewhere
        self.rainfall_dataset = None
        self.progress_bar = None
        self.sampling_pbar_desc = 'Sampling on validation set...'
        # self.first_samples_logged = False
        self.first_batch_visualized = False
        return

    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb.run.summary['logdir'] = trainer.default_root_dir
        self.rainfall_dataset = trainer.datamodule.precip_dataset

        for callback in trainer.callbacks:
            if isinstance(callback, RichProgressBar):
                self.progress_bar = callback
                break
        return

    @rank_zero_only
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                             batch_idx: int) -> None:
        # visualize the first batch in logger
        batch, _ = batch  # discard coordinates
        if not self.first_batch_visualized:
            visualize_batch(**batch)
            self.first_batch_visualized = True
        return

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        if self._check_frequency(trainer, 'score'):
            self._log_score(pl_module, outputs)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_last_ckpt(trainer)

    @rank_zero_only
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._check_frequency(trainer, 'img'):
            self._log_samples(trainer, pl_module, outputs)
            self.save_ckpt(trainer)

    @abstractmethod
    def _log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def _log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass

    def _check_frequency(self, trainer: "pl.trainer", key: str):
        if self.freqs[key] == -1:
            return False
        if self.check_freq_via == 'global_step':
            check_idx = trainer.global_step
        elif self.check_freq_via == 'epoch':
            check_idx = trainer.current_epoch
        if check_idx >= self.next_log_idx[key]:
            self.next_log_idx[key] = check_idx + self.freqs[key]
            return True
        else:
            return False

    def _modify_pbar_desc(self):
        task_id, original_description = None, None
        # Ensure progress bar is active and tasks are initialized
        if self.progress_bar.progress is not None and len(self.progress_bar.progress.tasks) > 0:
            # Look for the current validation task
            for task in self.progress_bar.progress.tasks:
                if "Validation" in task.description or "Sanity Checking" in task.description:
                    task_id = task.id
                    original_description = task.description
                    # Update the description of the active validation progress bar
                    self.progress_bar.progress.update(task_id, description=self.sampling_pbar_desc)
                    self.progress_bar.progress.refresh()
        return task_id, original_description

    def _revert_pbar_desc(self, task_id, original_description):
        # Ensure progress bar is active and tasks are initialized
        if self.progress_bar.progress is not None and len(self.progress_bar.progress.tasks) > 0:
            # Look for the current validation task
            for task in self.progress_bar.progress.tasks:
                if task.id == task_id:
                    # Update the description of the active validation progress bar
                    self.progress_bar.progress.update(task_id, description=original_description)
                    self.progress_bar.progress.refresh()
        return

    def save_ckpt(self, trainer: Trainer):
        if self.enable_save_ckpt:
            save_dir = trainer.logger.save_dir
            ckpt_path = os.path.join(save_dir, 'checkpoints',
                                     f'epoch_{trainer.current_epoch:03d}_step_{wandb.run.step:03d}.ckpt')
            trainer.save_checkpoint(ckpt_path)
            if self.add_reference_artifact:  # Log the checkpoint as a reference artifact
                artifact = wandb.Artifact(name=f'model-ckpt-{wandb.run.id}', type='model')
                artifact.add_reference(f"file://{ckpt_path}")
                artifact.metadata = {'epoch': trainer.current_epoch, 'step': wandb.run.step}
                wandb.run.log_artifact(artifact, aliases=[f'epoch_{trainer.current_epoch:03d}',
                                                          f'step_{wandb.run.step:03d}'])
        return

    def save_last_ckpt(self, trainer: Trainer):
        if self.enable_save_ckpt:
            save_dir = trainer.logger.save_dir
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'last.ckpt')
            trainer.save_checkpoint(ckpt_path)
