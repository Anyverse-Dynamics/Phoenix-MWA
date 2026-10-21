# Copyright 2025 starVLA community. All rights reserved.
"""VGGT-Ω / V-JEPA 2.1 ego-video distillation trainer."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from wam.encoder_pretrain.data.clips import (
    SharedClipBucketLoader,
    format_shared_ego_video_dataloader_statistics,
)
from wam.encoder.multi_teacher_distillation import (
    MultiTeacherDistillationModel,
    VideoBatch,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    cfg.output_dir = str(Path(cfg.run_root_dir) / str(cfg.run_id))
    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.trainer.get("resume_from"):
        raise FileExistsError(f"Nonempty run directory requires explicit trainer.resume_from: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return output_dir


def build_accelerator(trainer_cfg) -> Accelerator:
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        gradient_accumulation_steps=int(trainer_cfg.gradient_accumulation_steps),
        mixed_precision=str(trainer_cfg.precision),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp],
    )


def build_wandb_config(cfg) -> dict[str, object]:
    """Keep W&B metadata compact while the full resolved config remains on disk."""

    config = OmegaConf.to_container(cfg, resolve=True)
    if "datasets" not in config or "ego_video_data" not in config["datasets"]:
        return config
    ego_video_data = config["datasets"]["ego_video_data"]
    dataset_summaries = []
    configured_path_count = 0
    for dataset in ego_video_data["datasets"]:
        path_count = len(dataset["paths"]) if "paths" in dataset else 1
        configured_path_count += path_count
        summary = {
            key: dataset[key]
            for key in ("name", "video_key", "video_backend", "target_fps", "skip_empty", "require_success")
            if key in dataset
        }
        summary["configured_path_count"] = path_count
        dataset_summaries.append(summary)
    ego_video_data["datasets"] = dataset_summaries
    ego_video_data["configured_source_count"] = len(dataset_summaries)
    ego_video_data["configured_path_count"] = configured_path_count
    return config


def _center_square(frame) -> torch.Tensor:
    tensor = TVF.pil_to_tensor(frame.convert("RGB")).float().div_(255.0)
    height, width = tensor.shape[-2:]
    size = min(height, width)
    tensor = TVF.center_crop(tensor, [size, size])
    return TVF.resize(tensor, [256, 256], antialias=True)


def batch_to_video_batch(batch: list[dict], device: torch.device) -> VideoBatch:
    frames = torch.stack([torch.stack([_center_square(frame) for frame in sample["frames"]]) for sample in batch])
    positions = torch.as_tensor(np.stack([sample["frame_positions"] for sample in batch]), dtype=torch.long)
    return VideoBatch(
        frames.to(device=device, non_blocking=True),
        positions.to(device=device, non_blocking=True),
    )


def _is_no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim <= 1 or name.endswith(".bias") or "embedding" in name or "template" in name or "token" in name


def _parameter_family(name: str) -> str:
    if "student.dino." in name:
        return "dino"
    if "student.adapter." in name:
        return "adapter"
    if "vggt_objective." in name:
        return "vggt_projector"
    if "vjepa_objective." in name:
        return "vjepa_predictor"
    raise ValueError(f"Trainable parameter is outside the online distillation modules: {name}")


def build_optimizer(model: torch.nn.Module, trainer_cfg) -> torch.optim.Optimizer:
    lr_cfg = trainer_cfg.learning_rate
    peak_lrs = {
        "dino": float(lr_cfg.dino_peak),
        "adapter": float(lr_cfg.adapter_peak),
        "vggt_projector": float(lr_cfg.vggt_projector_peak),
        "vjepa_predictor": float(lr_cfg.vjepa_predictor_peak),
    }
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = _parameter_family(name)
        grouped.setdefault((family, _is_no_decay(name, parameter)), []).append(parameter)
    optimizer_groups = [
        {
            "params": parameters,
            "lr": peak_lrs[family],
            "weight_decay": 0.0 if no_decay else float(trainer_cfg.optimizer.weight_decay),
            "family": family,
            "name": family,
        }
        for (family, no_decay), parameters in grouped.items()
    ]
    return torch.optim.AdamW(
        optimizer_groups,
        betas=tuple(float(value) for value in trainer_cfg.optimizer.betas),
        eps=float(trainer_cfg.optimizer.eps),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, trainer_cfg):
    total_steps = int(trainer_cfg.max_train_steps)
    warmup_steps = int(total_steps * float(trainer_cfg.warmup_fraction))
    lr_cfg = trainer_cfg.learning_rate
    final_lrs = {
        "dino": float(lr_cfg.dino_final),
        "adapter": float(lr_cfg.adapter_final),
        "vggt_projector": float(lr_cfg.vggt_projector_final),
        "vjepa_predictor": float(lr_cfg.vjepa_predictor_final),
    }

    lambdas = []
    for group in optimizer.param_groups:
        peak = float(group["lr"])
        final_ratio = final_lrs[group["family"]] / peak

        def schedule(step: int, final_ratio: float = final_ratio) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return final_ratio + (1.0 - final_ratio) * cosine

        lambdas.append(schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def rng_state() -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def gather_rng_states(accelerator: Accelerator) -> list[dict[str, object]]:
    local = rng_state()
    if accelerator.num_processes == 1:
        return [local]
    states: list[dict[str, object] | None] = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(states, local)
    return [state for state in states if state is not None]


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    bucket_loaders: SharedClipBucketLoader,
    cfg,
    step: int,
    output_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    rng_by_process = gather_rng_states(accelerator)
    if accelerator.is_main_process:
        target = accelerator.unwrap_model(model)
        payload = {
            "format": target.resume_checkpoint_format,
            "step": int(step),
            "world_size": int(accelerator.num_processes),
            "gradient_accumulation_steps": int(accelerator.gradient_accumulation_steps),
            "training_model": target.training_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": bucket_loaders.state_dict(),
            "rng_by_process": rng_by_process,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        atomic_save(payload, output_dir / "checkpoints" / f"step_{step:08d}.pt")
    accelerator.wait_for_everyone()


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    bucket_loaders,
    process_index: int,
    num_processes: int,
    gradient_accumulation_steps: int,
    expected_format: str = "phoenix.encoder_pretrain.resume.v1",
    config=None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != expected_format:
        raise RuntimeError(
            f"Unsupported resume checkpoint for this framework: expected {expected_format!r}, "
            f"got {payload.get('format')!r}"
        )
    if int(payload["world_size"]) != int(num_processes):
        raise RuntimeError(
            f"Checkpoint world_size={payload['world_size']} differs from current world_size={num_processes}"
        )
    if payload["gradient_accumulation_steps"] != gradient_accumulation_steps:
        raise RuntimeError("Resume requires the original gradient accumulation steps")
    if config is not None:
        current = OmegaConf.to_container(config, resolve=True)
        previous = payload["config"]
        for section in ("framework", "datasets", "seed"):
            if current.get(section) != previous.get(section):
                raise RuntimeError(f"Resume configuration changed: {section}")
        ignored = {"resume_from", "logging_frequency", "save_interval"}
        old_trainer = {k: v for k, v in previous["trainer"].items() if k not in ignored}
        new_trainer = {k: v for k, v in current["trainer"].items() if k not in ignored}
        if old_trainer != new_trainer:
            raise RuntimeError("Resume training schedule or optimizer configuration changed")
    model.load_training_state_dict(payload["training_model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    bucket_loaders.load_state_dict(
        payload["sampler"],
        logical_step=int(payload["step"]) * int(gradient_accumulation_steps),
    )
    states = payload["rng_by_process"]
    if process_index >= len(states):
        raise RuntimeError(f"Checkpoint has {len(states)} RNG states for process index {process_index}")
    restore_rng_state(states[process_index])
    return int(payload["step"])


class MultiTeacherDistillationTrainer:
    def __init__(self, cfg, model, bucket_loaders, optimizer, scheduler, accelerator, output_dir) -> None:
        self.config = cfg
        self.model = model
        self.bucket_loaders = bucket_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.completed_steps = 0
        self.micro_step = 0
        self.grad_norm = None
        self._wandb_enabled = False
        self._wandb = None
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}

    def prepare(self) -> None:
        set_seed(int(self.config.seed) + self.accelerator.process_index)
        self.accelerator.dataloader_config.dispatch_batches = False
        self._load_initial_checkpoint()
        self._save_configs()
        self._log_dataloader_statistics()
        self._prepare_distributed()
        self._init_wandb()

    def _log_dataloader_statistics(self) -> None:
        if not self.accelerator.is_main_process:
            return
        logger.info(
            format_shared_ego_video_dataloader_statistics(
                self.bucket_loaders,
                self.config,
                num_processes=self.accelerator.num_processes,
                gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
            )
        )

    def _save_configs(self) -> None:
        if not self.accelerator.is_main_process:
            return
        OmegaConf.save(self.config, self.output_dir / "config.full.yaml", resolve=True)

    def _load_initial_checkpoint(self) -> None:
        resume = self.config.trainer.get("resume_from")
        if not resume:
            return
        self.completed_steps = load_checkpoint(
            resume,
            self.model,
            self.optimizer,
            self.scheduler,
            self.bucket_loaders,
            self.accelerator.process_index,
            self.accelerator.num_processes,
            int(self.config.trainer.gradient_accumulation_steps),
            self.model.resume_checkpoint_format,
            config=self.config,
        )
        self.micro_step = self.completed_steps * int(self.config.trainer.gradient_accumulation_steps)

    def _prepare_distributed(self) -> None:
        self.accelerator.even_batches = False
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
        )
        prepared = self.accelerator.prepare(self.bucket_loaders.loader)
        self.bucket_loaders = self.bucket_loaders.with_loader(prepared)

    def _init_wandb(self) -> None:
        disabled = str(os.environ.get("WANDB_MODE", "")).lower() == "disabled"
        if not bool(self.config.get("tracking", {}).get("enabled", False)) or disabled:
            return
        if self.accelerator.is_main_process:
            import wandb

            self._wandb = wandb
            wandb_run_id = self.config.get("wandb_run_id")
            init_kwargs = {
                "name": str(self.config.run_id),
                "dir": str(self.output_dir),
                "project": str(self.config.wandb_project),
                "entity": self.config.get("wandb_entity"),
                "group": "multi-teacher-distillation",
                "config": build_wandb_config(self.config),
            }
            if wandb_run_id:
                init_kwargs.update(id=str(wandb_run_id), resume="must")
            wandb.init(**init_kwargs)
            self._wandb_enabled = True

    def _teacher_type(self) -> str:
        mode = self.accelerator.unwrap_model(self.model).mode
        if mode == "vggt_only":
            return "vggt"
        if mode == "vjepa_only":
            return "vjepa"
        if mode == "joint":
            first = str(self.config.trainer.get("first_teacher", "vggt"))
            return MultiTeacherDistillationModel.teacher_for_step(self.completed_steps, first)
        raise RuntimeError("dino_only does not have a training objective")

    def train(self) -> None:
        total_steps = int(self.config.trainer.max_train_steps)
        progress = tqdm(
            total=total_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        self.optimizer.zero_grad(set_to_none=True)
        while self.completed_steps < total_steps:
            metrics = self._train_micro_step()
            self.micro_step += 1
            if self.accelerator.sync_gradients:
                self.completed_steps += 1
                progress.update(1)
                self._log(metrics)
                if self.completed_steps % int(self.config.trainer.save_interval) == 0:
                    self._save_checkpoint()
        self._finish()

    def _train_micro_step(self) -> dict[str, float]:
        (length, stride), samples = self.bucket_loaders.next()
        batch = batch_to_video_batch(samples, self.accelerator.device)
        teacher_type = self._teacher_type()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(batch, teacher_type, self.completed_steps)
            self.accelerator.backward(outputs["loss"])
            self.grad_norm = None
            if self.accelerator.sync_gradients:
                self.grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), float(self.config.trainer.gradient_clipping)
                )
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.scheduler.step()
                self.accelerator.unwrap_model(self.model).update_ema()
            self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            key: float(value.detach())
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        if teacher_type != "both":
            metrics[f"loss_{teacher_type}"] = metrics["loss"]
        metrics.update(
            {
                "bucket_T": length,
                "bucket_stride": stride,
                "batch_size": len(samples),
                "teacher_vggt": float(teacher_type in {"vggt", "both"}),
                "teacher_vjepa": float(teacher_type in {"vjepa", "both"}),
                "grad_norm": float(self.grad_norm) if self.grad_norm is not None else 0.0,
            }
        )
        for dataset_name in sorted({sample["dataset_name"] for sample in samples}):
            metrics[f"batch_count/{dataset_name}"] = sum(sample["dataset_name"] == dataset_name for sample in samples)
        return metrics

    def _log(self, metrics: dict[str, float]) -> None:
        if not self.accelerator.is_main_process:
            return
        for key, value in metrics.items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + float(value)
            self._metric_counts[key] = self._metric_counts.get(key, 0) + 1
        if self.completed_steps % int(self.config.trainer.logging_frequency):
            return

        metrics = {
            key: total / self._metric_counts[key]
            for key, total in self._metric_sums.items()
        }
        self._metric_sums.clear()
        self._metric_counts.clear()
        metrics["step"] = self.completed_steps
        for group in self.optimizer.param_groups:
            metrics[f"learning_rate/{group['family']}"] = float(group["lr"])
        if self._wandb_enabled:
            self._wandb.log(metrics, step=self.completed_steps)
        with (self.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _save_checkpoint(self) -> None:
        save_checkpoint(
            self.accelerator,
            self.model,
            self.optimizer,
            self.scheduler,
            self.bucket_loaders,
            self.config,
            self.completed_steps,
            self.output_dir,
        )
        if self.accelerator.is_main_process:
            with (self.output_dir / "summary.jsonl").open("a") as handle:
                handle.write(json.dumps({"steps": self.completed_steps}) + "\n")

    def _finish(self) -> None:
        self._save_checkpoint()
        if self.accelerator.is_main_process:
            artifact = self.accelerator.unwrap_model(self.model).export_ema_student()
            artifact["config"] = OmegaConf.to_container(self.config, resolve=True)
            atomic_save(artifact, self.output_dir / "ema_student_final.pt")
        if self._wandb_enabled:
            self._wandb.finish()
        self.accelerator.end_training()



def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
