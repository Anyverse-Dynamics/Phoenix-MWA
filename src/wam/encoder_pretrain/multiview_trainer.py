"""Encoder training with metadata-driven all-view loading."""

from __future__ import annotations


import torch
from accelerate.logging import get_logger

from wam.encoder_pretrain.data.multiview import (
    SAMPLING_COUNT_KEYS,
    format_stratified_view_sampling_statistics,
)
from .trainer import MultiTeacherDistillationTrainer


logger = get_logger(__name__)


class StratifiedViewMultiTeacherDistillationTrainer(MultiTeacherDistillationTrainer):
    """Add globally reduced sampling-composition metrics to the trainer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_counts = torch.zeros(
            len(SAMPLING_COUNT_KEYS),
            dtype=torch.float64,
            device=self.accelerator.device,
        )

    def _log_dataloader_statistics(self) -> None:
        if self.accelerator.is_main_process:
            logger.info(
                format_stratified_view_sampling_statistics(
                    self.bucket_loaders,
                    self.config,
                    num_processes=self.accelerator.num_processes,
                )
            )

    def _train_micro_step(self) -> dict[str, float]:
        metrics = super()._train_micro_step()
        counts = self.bucket_loaders.last_sampling_counts
        self._sampling_counts += torch.tensor(
            [counts[key] for key in SAMPLING_COUNT_KEYS],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        if not self.accelerator.sync_gradients:
            return metrics

        global_counts = self.accelerator.reduce(self._sampling_counts, reduction="sum")
        total = float(global_counts[-1])
        for index, key in enumerate(SAMPLING_COUNT_KEYS[:-1]):
            metrics[f"sampling_count/{key}"] = float(global_counts[index])
            metrics[f"sampling_fraction/{key}"] = float(global_counts[index]) / total
        metrics["sampling_count/total"] = total
        self._sampling_counts.zero_()
        return metrics
