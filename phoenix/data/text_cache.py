"""Manifest-bound Wan UMT5 contexts, reused by every window of an instruction."""
import hashlib
import os
from pathlib import Path

import torch

from phoenix.dataset import PROMPT
from .common import atomic_json, check_sources, digest, directory_guard, read_json, sha256


def validate_pack(path, prompt):
    pack = torch.load(path, map_location="cpu", weights_only=True)
    context, mask = pack["context"], pack["context_mask"]
    if (pack["prompt"] != prompt or context.shape != (128, 4096) or context.dtype != torch.bfloat16
            or not torch.isfinite(context).all() or mask.shape != (128,) or mask.dtype != torch.bool):
        raise ValueError("Invalid Phoenix text cache")
    return pack


def build_text_cache(config, *, identity=None, encode=None):
    from .encoders import text_identity, load_text
    manifest = read_json(config["manifest"])
    check_sources(manifest)
    identity = text_identity(config) if identity is None else identity
    contract = dict(schema="phoenix.text_build.v1", manifest_sha256=sha256(config["manifest"]),
                    prompt_template=PROMPT, encoder=identity)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with directory_guard(output / "text.owner"):
        contract_path = output / "contract.json"
        if contract_path.exists():
            if read_json(contract_path) != contract:
                raise ValueError("Text cache belongs to different data/encoder/tokenizer")
        else:
            if any(output.glob("*.pt")) or (output / "text_index.json").exists():
                raise ValueError("Unowned text cache files; use a new output directory")
            atomic_json(contract_path, contract)
        prompts = sorted({PROMPT.format(task=task) for tasks in manifest["tasks"].values() for task in tasks.values()})
        entries = {}
        for prompt in prompts:
            key = hashlib.sha256(prompt.encode()).hexdigest()
            path = output / (key + ".pt")
            receipt_path = output / (key + ".entry.json")
            if not path.exists():
                encode = load_text(config) if encode is None else encode
                context, mask = encode(prompt)
                partial = path.with_suffix(".partial")
                torch.save(dict(context=context.cpu(), context_mask=mask.cpu(), prompt=prompt), partial)
                validate_pack(partial, prompt)
                os.replace(partial, path)
                atomic_json(receipt_path, dict(contract_sha256=digest(contract), prompt_sha256=key,
                                               sha256=sha256(path)))
            receipt = read_json(receipt_path)
            if receipt != dict(contract_sha256=digest(contract), prompt_sha256=key, sha256=sha256(path)):
                raise ValueError("Text cache entry changed or has no matching publication receipt")
            validate_pack(path, prompt)
            entries[key] = dict(path=path.name, sha256=sha256(path))
        check_sources(manifest)
        index = dict(schema="phoenix.text_index.v1", manifest_sha256=contract["manifest_sha256"],
                     prompt_template=PROMPT, encoder=identity, prompts=entries)
        if (output / "text_index.json").exists() and read_json(output / "text_index.json") != index:
            raise ValueError("Completed text cache contents changed")
        atomic_json(output / "text_index.json", index)
        atomic_json(output / "READY.json", dict(index_sha256=sha256(output / "text_index.json"), prompts=len(entries)))
    return dict(status="TEXT_READY", index=str(output / "text_index.json"), prompts=len(entries))
