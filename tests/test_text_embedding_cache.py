import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from wam.datasets.robotwin_joint96_robot37 import RobotwinJoint96OnlineDataset
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.text_embedding_cache import TaskIndexShardedTextEmbeddingCache, prompt_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    tasks = [{"task_index": i, "task": f"task {i}"} for i in range(4)]
    task_file = data / "meta/tasks.jsonl"
    task_file.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    prompts = [DEFAULT_PROMPT.format(task=t["task"]) for t in tasks]
    lengths = [1, 2, 3, 128]
    contexts = [torch.arange(n * 4096, dtype=torch.float32).reshape(n, 4096).div(128).to(torch.bfloat16)
                for n in lengths]
    for start in (0, 2):
        shard = root / "shards" / f"shard_{start:07d}_{start + 2:07d}"
        shard.mkdir(parents=True)
        (shard / "manifest.json").write_text(json.dumps({"start": start, "end": start + 2}))
        ls = np.array(lengths[start:start + 2], dtype=np.int64)
        np.save(shard / "lengths.npy", ls)
        np.save(shard / "offsets.npy", np.concatenate(([0], np.cumsum(ls))))
        np.save(shard / "prompt_sha256.npy", np.array([prompt_sha256(p) for p in prompts[start:start + 2]], dtype="S64"))
        torch.cat(contexts[start:start + 2]).view(torch.uint16).numpy().tofile(shard / "embeddings.uint16.mmap")
    manifest = dict(format="task_index_sharded_mmap_v1", total_tasks=4, context_len=128,
                    embedding_dim=4096, shard_size=2, logical_dtype="bfloat16", storage_dtype="uint16",
                    tasks_sha256=[hashlib.sha256(task_file.read_bytes()).hexdigest()], prompt_template=DEFAULT_PROMPT)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, data, prompts, contexts


def test_exact_copy_receipt():
    path = ROOT / "src/wam/datasets/text_embedding_cache.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "2935525bf2af49acdc09fd462adb00a9a7889f1fe08d5bb25b99ddd1e9dd128e"


def test_lazy_mmap_and_exact_bf16_padding(cache):
    root, _, prompts, contexts = cache
    bank = TaskIndexShardedTextEmbeddingCache(root, expected_context_len=128)
    assert not bank._open_shards
    for i, (prompt, expected) in enumerate(zip(prompts, contexts)):
        context, mask = bank.get(i, prompt)
        assert context.shape == (128, 4096) and context.dtype == torch.bfloat16
        assert mask.shape == (128,) and mask.dtype == torch.bool
        assert mask.sum().item() == len(expected)
        assert torch.equal(context[:len(expected)].view(torch.uint16), expected.view(torch.uint16))
        assert not context[len(expected):].any()
        context.fill_(1)
        assert torch.equal(bank.get(i, prompt)[0][:len(expected)], expected)
    assert set(bank._open_shards) == {0, 1}
    assert all(isinstance(s.embeddings, np.memmap) for s in bank._open_shards.values())


@pytest.mark.parametrize("index", [-1, 4])
def test_invalid_task(cache, index):
    with pytest.raises(IndexError):
        TaskIndexShardedTextEmbeddingCache(cache[0]).get(index, cache[2][0])


def test_wrong_prompt_rejected(cache):
    with pytest.raises(ValueError, match="Prompt hash mismatch"):
        TaskIndexShardedTextEmbeddingCache(cache[0]).get(0, "different")


@pytest.mark.parametrize("field,value,match", [
    ("format", "unknown", "Unsupported"), ("storage_dtype", "float32", "requires"),
    ("logical_dtype", "float16", "requires"), ("context_len", 64, "does not match"),
])
def test_manifest_contract(cache, field, value, match):
    path = cache[0] / "manifest.json"
    manifest = json.loads(path.read_text()); manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        TaskIndexShardedTextEmbeddingCache(cache[0], expected_context_len=128)


def test_truncated_payload(cache):
    path = next((cache[0] / "shards").glob("shard_0000000_*/embeddings.uint16.mmap"))
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 2)
    with pytest.raises(ValueError, match="payload size mismatch"):
        TaskIndexShardedTextEmbeddingCache(cache[0]).get(0, cache[2][0])


def reader(cache):
    root, data, prompts, _ = cache
    ds = RobotwinJoint96OnlineDataset.__new__(RobotwinJoint96OnlineDataset)
    ds._text_cache = None
    ds.text_cache_source_root = Path("/nonexistent/legacy/repository")
    ds.text_embedding_cache_dir = root
    ds.dataset_dirs = [data]
    ds._task_lookup = {f"task {i}": i for i in range(len(prompts))}
    return ds


@pytest.mark.parametrize("hash_as_string", [False, True])
def test_robotwin_uses_local_reader_and_official_padding(cache, monkeypatch, hash_as_string):
    if hash_as_string:
        path = cache[0] / "manifest.json"
        manifest = json.loads(path.read_text()); manifest["tasks_sha256"] = manifest["tasks_sha256"][0]
        path.write_text(json.dumps(manifest))
    def forbid(*args, **kwargs):
        raise AssertionError("external dynamic import attempted")
    monkeypatch.setattr(importlib.util, "spec_from_file_location", forbid)
    ds = reader(cache)
    for i, prompt in enumerate(cache[2]):
        context, mask = ds._get_cached_text_context(prompt)
        assert mask.all() and mask.dtype == torch.bool
        assert torch.equal(context[:len(cache[3][i])], cache[3][i])
    assert isinstance(ds._text_cache, TaskIndexShardedTextEmbeddingCache)
    with pytest.raises(ValueError, match="Unexpected T5 prompt"):
        ds._get_cached_text_context("wrong prefix")


@pytest.mark.parametrize("field,value,match", [
    ("tasks_sha256", ["bad"], "task identities"), ("prompt_template", "bad", "template/width"),
    ("embedding_dim", 8, "template/width"),
])
def test_robotwin_guards_preserved(cache, field, value, match):
    path = cache[0] / "manifest.json"
    manifest = json.loads(path.read_text()); manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        reader(cache)._get_cached_text_context(cache[2][0])


@pytest.mark.parametrize("legacy_arg", [[], ["--source-root", "/nonexistent/repository"]])
def test_export_cli_needs_no_external_source(cache, tmp_path, legacy_arg):
    output = tmp_path / "exported"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, str(ROOT / "tools/import_joint96_posttrain_text.py"),
                    "--cache", str(cache[0]), "--tasks", str(cache[1] / "meta/tasks.jsonl"),
                    "--output", str(output), *legacy_arg], env=env, check=True, timeout=60, capture_output=True)
    report = json.loads((output / "COMPLETE.json").read_text())
    assert report["reencoded"] is False and len(report["files"]) == 4
    for i, prompt in enumerate(cache[2]):
        payload = torch.load(output / (prompt_sha256(prompt) + ".t5_len128.wan22ti2v5b.pt"), weights_only=True)
        expected = TaskIndexShardedTextEmbeddingCache(cache[0]).get(i, prompt)
        assert torch.equal(payload["context"], expected[0]) and torch.equal(payload["mask"], expected[1])


def test_isolated_import_without_fastwam():
    script = '''
import sys
class Deny:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fastwam' or fullname.startswith('fastwam.'):
            raise ImportError('External FastWAM forbidden')
sys.meta_path.insert(0, Deny())
from wam.datasets.robotwin_joint96_robot37 import RobotwinJoint96OnlineDataset
from wam.datasets.text_embedding_cache import TaskIndexShardedTextEmbeddingCache
assert not any(n == 'fastwam' or n.startswith('fastwam.') for n in sys.modules)
'''
    subprocess.run([sys.executable, "-c", script], cwd="/tmp", env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                   check=True, timeout=90, capture_output=True)
