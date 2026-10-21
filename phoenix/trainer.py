"""Focused Phoenix trainer: native joint objective, Accelerate, exact progress resume.

The joint objective and trainable module boundaries are shared across single
and distributed training. Checkpoints use the Phoenix training-state format.
"""
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .losses import _joint96_training_loss
from .training import configure_training_rtc, rtc_checkpoint_metadata
from .training_data import EpochBatchSampler


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main_process_action(accelerator, function):
    """Propagate rank-zero publication failures before peers enter a barrier."""
    from accelerate.utils import broadcast_object_list
    status = [None]
    if accelerator.is_main_process:
        try:
            function()
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(status, from_process=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def validate_checkpoint_directory(path, contract):
    root = Path(path)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if complete.get("schema") != "phoenix.training_state.v1" or complete.get("contract") != contract:
        raise ValueError("Phoenix resume contract/topology differs")
    files = complete.get("files", {})
    if not {"weights.pt", "trainer_state.json"}.issubset(files):
        raise ValueError("Incomplete Phoenix checkpoint")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"COMPLETE.json"}:
        raise ValueError("Phoenix checkpoint file set differs")
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (root / name).is_symlink():
            raise ValueError("Invalid checkpoint manifest path")
        if file_sha256(root / name) != digest:
            raise ValueError(f"Phoenix checkpoint checksum differs: {name}")
    state = json.loads((root / "trainer_state.json").read_text())
    if state.get("step") != complete.get("step") or state.get("contract") != contract:
        raise ValueError("Phoenix checkpoint progress differs")
    world = contract["world_size"]
    required_rng = {"random_states_0.pkl"} if world == 1 else {f"random_states_{r}.pkl" for r in range(world)}
    if not required_rng.issubset(files):
        raise ValueError("Missing rank RNG state")
    if not any("optimizer" in name or "optim_states" in name for name in files):
        raise ValueError("Missing optimizer state")
    if not any(name.startswith("custom_checkpoint_") for name in files):
        raise ValueError("Missing learning-rate scheduler state")
    return state


class TrainingModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wam = model

    def forward(self, sample):
        return _joint96_training_loss(self.wam, sample, tiled=False)


@contextmanager
def evaluation_state(model):
    """Validation consumes neither training RNG nor persistent train/eval modes."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for module, training in modes:
            module.training = training


class PhoenixTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, accelerator, *, input_contract):
        self.config, self.accelerator = config, accelerator
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.step, self.epoch, self.next_batch = 0, 0, 0
        self.output = Path(config["output_dir"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.contract = dict(input_contract, world_size=accelerator.num_processes,
                             training=config["training"], validation=config["validation"])
        self.rtc = config["training_time_rtc"]
        configure_training_rtc(model, max_prefix=self.rtc["max_prefix_steps"],
                               no_prefix_probability=self.rtc["no_prefix_probability"])
        model.set_trainable_modules()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        options = config["training"]
        self.optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"],
                                           weight_decay=options["weight_decay"],
                                           betas=tuple(options["betas"]), eps=options["eps"])
        maximum, warmup = options["max_steps"], options["warmup_steps"]
        if warmup is None:
            warmup = int(maximum * 0.05)
        if not 0 <= warmup < maximum:
            raise ValueError("Warmup must be below max_steps")
        minimum_ratio = options["min_learning_rate"] / options["learning_rate"]

        def learning_rate(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, max(0.0, (step - warmup) / max(1, maximum - warmup)))
            return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, learning_rate)
        self.module, self.optimizer = accelerator.prepare(TrainingModule(model), self.optimizer)
        accelerator.register_for_checkpointing(self.scheduler)
        # The inference-compatible weights are saved separately. Skip the huge
        # frozen encoder tensors in Accelerate's ordinary DDP model checkpoint.
        accelerator.register_save_state_pre_hook(lambda models, weights, folder: weights.clear())
        accelerator.register_load_state_pre_hook(lambda models, folder: models.clear())
        self.core.set_runtime_device(accelerator.device)
        self.core.set_trainable_modules()
        self.parameters = [p for p in self.core.parameters() if p.requires_grad]

    @property
    def core(self):
        return self.accelerator.unwrap_model(self.module).wam

    def _loader(self):
        options, accelerator = self.config["training"], self.accelerator
        sampler = EpochBatchSampler(len(self.train_dataset), options["batch_size"],
            rank=accelerator.process_index, world_size=accelerator.num_processes,
            seed=options["seed"], accumulation=options["gradient_accumulation_steps"],
            epoch=self.epoch, start_batch=self.next_batch)
        # Do not prepare this loader again: its batch sampler already shards.
        generator = torch.Generator().manual_seed(options["seed"] + self.epoch + accelerator.process_index)
        loader = DataLoader(self.train_dataset, batch_sampler=sampler,
                            num_workers=options["num_workers"], generator=generator,
                            pin_memory=accelerator.device.type != "cpu")
        return loader, sampler.total_batches

    def restore(self, path):
        state = validate_checkpoint_directory(path, self.contract)
        from .training import initialize_from_pretrained
        payload = torch.load(Path(path) / "weights.pt", map_location="cpu", weights_only=True)
        if payload.get("step") != state["step"]:
            raise ValueError("Weights and optimizer update counts differ")
        initialize_from_pretrained(self.core, payload)
        self.accelerator.load_state(str(path))
        self.step, self.epoch, self.next_batch = state["step"], state["epoch"], state["next_batch"]
        if not 0 < self.step <= self.config["training"]["max_steps"]:
            raise ValueError("Invalid restored optimizer step")
        self._loader()  # Validate resume position before any further forward pass.

    def save(self):
        accelerator = self.accelerator
        destination = self.output / "checkpoints" / f"step_{self.step:06d}"
        def create():
            destination.mkdir(parents=True, exist_ok=False)
        main_process_action(accelerator, create)
        metadata = rtc_checkpoint_metadata(self.core, optimizer_step=self.step,
            max_prefix=self.rtc["max_prefix_steps"],
            no_prefix_probability=self.rtc["no_prefix_probability"])
        def save_weights():
            payload = {"step": self.step, "joint96_metadata": metadata}
            payload["training_inputs"] = self.contract.get("assets", {})
            for name in ("mot", "proprio_encoder", "action_proprio_encoder"):
                module = getattr(self.core, name, None)
                payload[name] = None if module is None else {k: v.detach().cpu() for k, v in module.state_dict().items()}
            temporary = destination / "weights.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(destination / "weights.pt")
            atomic_json(destination / "trainer_state.json", dict(step=self.step, epoch=self.epoch,
                        next_batch=self.next_batch, contract=self.contract))
        main_process_action(accelerator, save_weights)
        accelerator.save_state(str(destination))
        accelerator.wait_for_everyone()
        def complete():
            files = {p.relative_to(destination).as_posix(): file_sha256(p)
                     for p in destination.rglob("*") if p.is_file()}
            atomic_json(destination / "COMPLETE.json", dict(schema="phoenix.training_state.v1",
                        step=self.step, contract=self.contract, files=files))
            validate_checkpoint_directory(destination, self.contract)
            atomic_json(self.output / "latest.json", dict(step=self.step, path=str(destination)))
        main_process_action(accelerator, complete)
        return destination

    def evaluate(self):
        if self.val_dataset is None:
            return None
        accelerator = self.accelerator
        count = min(len(self.val_dataset), self.config["validation"]["samples"])
        # Fixed global panel, partitioned without duplicating tail samples.
        selected = [int((i + 0.5) * len(self.val_dataset) / count) for i in range(count)]
        selected = selected[accelerator.process_index::accelerator.num_processes]
        loader = DataLoader(Subset(self.val_dataset, selected), batch_size=1,
                            num_workers=0, generator=torch.Generator().manual_seed(0))
        totals = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
        with evaluation_state(self.core):
            for sample in loader:
                seed = self.config["validation"]["seed"] + int(sample["sample_id"][0])
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with accelerator.autocast():
                    loss, metrics = _joint96_training_loss(self.core, sample, tiled=False)
                totals += torch.tensor([float(loss), float(metrics["loss_action"]), 1.],
                                       dtype=torch.float64, device=accelerator.device)
        totals = accelerator.reduce(totals, reduction="sum")
        return dict(step=self.step, validation_loss=float(totals[0] / totals[2]),
                    validation_action_loss=float(totals[1] / totals[2]), samples=int(totals[2]))

    def train(self, *, stop_after=None):
        """Optionally bound this invocation while retaining the configured schedule."""
        options, accelerator = self.config["training"], self.accelerator
        last_saved = self.step if self.step else None
        target = options["max_steps"] if stop_after is None else min(options["max_steps"], self.step + stop_after)
        self.optimizer.zero_grad(set_to_none=True)
        while self.step < target:
            loader, total_batches = self._loader()
            if self.next_batch == total_batches:
                self.epoch, self.next_batch = self.epoch + 1, 0
                continue
            for sample in loader:
                with accelerator.accumulate(self.module):
                    with accelerator.autocast():
                        loss, metrics = self.module(sample)
                    finite = torch.tensor(int(torch.isfinite(loss)), device=accelerator.device)
                    if accelerator.reduce(finite, reduction="sum").item() != accelerator.num_processes:
                        raise FloatingPointError("Nonfinite Phoenix training loss")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.parameters, options["max_grad_norm"])
                    self.optimizer.step()
                    skipped = accelerator.optimizer_step_was_skipped
                    self.optimizer.zero_grad(set_to_none=True)
                self.next_batch += 1
                if accelerator.sync_gradients and not skipped:
                    self.scheduler.step()
                    self.step += 1
                    report = dict(step=self.step, epoch=self.epoch, next_batch=self.next_batch,
                                  loss=float(accelerator.reduce(loss.detach(), reduction="mean")),
                                  learning_rate=self.scheduler.get_last_lr()[0])
                    if accelerator.is_main_process:
                        print(json.dumps(report), flush=True)
                    validation_every = self.config["validation"]["every"]
                    if validation_every and self.step % validation_every == 0:
                        report = self.evaluate()
                        if report is not None and accelerator.is_main_process:
                            print(json.dumps(report), flush=True)
                    if self.step % options["save_every"] == 0 or self.step == target:
                        self.save()
                        last_saved = self.step
                    if self.step >= target:
                        break
        if last_saved != self.step:
            self.save()
        accelerator.wait_for_everyone()
        return self.step
