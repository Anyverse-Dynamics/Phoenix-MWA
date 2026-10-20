"""Read existing task-routed BF16 T5 caches without importing the old trainer."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from wam.resources import resource_path

import numpy as np
import pyarrow.parquet as pq
import torch

from wam.datasets.local_robot_contracts import digest, sha256
from wam.datasets.local_robot_package import read_json


def validate_language_augmentation_prob(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("language_augmentation_prob must be a finite number in [0, 1]")
    return float(value)


def variant_index(count, *, seed, identity, augment, augmentation_prob=0.5):
    augmentation_prob = validate_language_augmentation_prob(augmentation_prob)
    if count < 1:
        raise ValueError("empty language route")
    if not augment or count == 1:
        return 0
    number = int(digest([seed, "language", identity]), 16)
    # Preserve every historical 50% selection for old callers and checkpoints.
    if augmentation_prob == 0.5:
        return 0 if number % 2 == 0 else 1 + (number // 2) % (count - 1)
    gate = digest([seed, "language_probability", identity])
    if int(gate, 16) >= augmentation_prob * (16 ** len(gate)):
        return 0
    return 1 + (number // 2) % (count - 1)


def _array(root, desc):
    path = root / desc["file"]
    if path.stat().st_size != int(desc["bytes"]):
        raise ValueError(f"T5 index integrity mismatch: {path}")
    return np.memmap(path, mode="r", dtype=desc["dtype"], shape=(int(desc["count"]),))


class LocalRobotTextCache:
    def __init__(self, registry_path, source, expected_hash):
        path = resource_path(registry_path)
        registry = read_json(path)
        if (registry.get("schema_version") != "task_routed_t5_v1" or registry.get("context_len") != 128
                or registry.get("dtype") != "bfloat16" or registry.get("encoder_id") != "wan22ti2v5b"):
            raise ValueError("unsupported local T5 registry contract")
        provider = registry["providers"][source]
        self.root = resource_path(provider["cache_root"])
        self.manifest = read_json(self.root / "manifest.json")
        if (self.manifest["context_len"], self.manifest["context_dim"], self.manifest["dtype"]) != (128, 4096, "bfloat16"):
            raise ValueError("T5 context contract mismatch")
        for key in ("encoder_id", "tokenizer_id"):
            if self.manifest[key] != registry[key]:
                raise ValueError("T5 encoder identity mismatch")
        if self.manifest["source_index_fingerprints"][source] != provider["task_route_fingerprint"]:
            raise ValueError("T5 source route fingerprint mismatch")
        route = provider["task_routes"]
        rows = pq.read_table(self.root / route["file"]).to_pylist()
        self.routes = {(r["physical_dataset_uid"], int(r["task_index"])): r for r in rows}
        if len(rows) != len(self.routes) or len(rows) != route["row_count"]:
            raise ValueError("duplicate or incomplete physical task routes")
        self.catalog = read_json(self.root / "catalog/manifest.json")
        if self.catalog["prompt_set_fingerprint"] != self.manifest["catalog_prompt_set_fingerprint"]:
            raise ValueError("T5 catalog/provider mismatch")
        catroot = self.root / "catalog"
        src = self.catalog["sources"][source]
        self.offsets = _array(catroot, src["episode_offsets"])
        self.ids = _array(catroot, src["prompt_ids"])
        self.prompt_offsets = _array(catroot, self.catalog["prompt_offsets"])
        desc = self.catalog["prompt_blob"]
        self.prompt_path = catroot / desc["file"]
        self.locators = {k: _array(self.root, d) for k, d in self.manifest["locators"].items()}
        self.parts, self.chunks = {}, OrderedDict()

    def _prompt(self, prompt_id):
        begin, end = map(int, self.prompt_offsets[prompt_id:prompt_id + 2])
        with self.prompt_path.open("rb") as f:
            f.seek(begin)
            return f.read(end - begin).decode("utf-8")

    def resolve(self, binding, *, seed, identity, augment, augmentation_prob=0.5):
        route = self.routes[(binding["physical_dataset_uid"], int(binding["task_index"]))]
        if (route["route_row"] != binding["route_row"] or route["source_fingerprint"] != binding["source_fingerprint"]):
            raise ValueError("T5 binding/physical route mismatch")
        row = int(route["route_row"])
        begin, end = map(int, self.offsets[row:row + 2])
        count = end - begin
        if count != int(binding["variant_count"]) or count != int(route["variant_count"]):
            raise ValueError("T5 variant count mismatch")
        if self._prompt(int(self.ids[begin])) != binding["original_text"]:
            raise ValueError("original task instruction does not match T5 route")
        variant = variant_index(
            count, seed=seed, identity=identity, augment=augment,
            augmentation_prob=augmentation_prob,
        )
        pid = int(self.ids[begin + variant])
        return self.resolve_prompt(pid, variant)

    def resolve_prompt(self, pid, variant):
        """Shared packed BF16 payload reader for task and episode routes."""
        part, chunk, offset, length = [int(self.locators[k][pid]) for k in ("part_ids", "chunk_ids", "token_offsets", "token_lengths")]
        if not 0 < length <= 128:
            raise ValueError("invalid T5 token length")
        if part not in self.parts:
            summary = self.manifest["parts"][part]
            path = self.root / summary["manifest"]
            self.parts[part] = (path.parent, read_json(path))
        key = part, chunk
        if key not in self.chunks:
            directory, metadata = self.parts[part]
            desc = metadata["chunks"][chunk]
            path = directory / desc["file"]
            if path.stat().st_size != int(desc["token_count"]) * 4096 * 2:
                raise ValueError("T5 payload size mismatch")
            self.chunks[key] = np.memmap(path, mode="r", dtype=np.uint16, shape=(int(desc["token_count"]), 4096))
            while len(self.chunks) > 16:
                self.chunks.popitem(last=False)
        packed = self.chunks[key]
        if offset < 0 or offset + length > len(packed):
            raise ValueError("T5 locator outside payload")
        context = torch.zeros(128, 4096, dtype=torch.bfloat16)
        context[:length] = torch.from_numpy(np.asarray(packed[offset:offset + length]).copy()).view(torch.bfloat16)
        if not torch.isfinite(context).all():
            raise ValueError("non-finite T5 context")
        return dict(context=context, context_mask=torch.arange(128) < length,
                    prompt=self._prompt(pid), language_variant=variant)
