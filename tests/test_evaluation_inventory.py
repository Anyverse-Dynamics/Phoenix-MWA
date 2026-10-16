import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def summary_module():
    spec = importlib.util.spec_from_file_location('plus_summary', ROOT/'tools/summarize_plus.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_classification_identity_and_shards():
    path = ROOT/'third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6'
    full = json.loads(path.read_text())
    keys = [(suite, r['id']-1) for suite, rows in full.items() for r in rows]
    assert len(keys) == len(set(keys)) == 10030
    for shards in (1, 2, 8, 32):
        partitions = [keys[rank::shards] for rank in range(shards)]
        assert len(set(k for part in partitions for k in part)) == 10030
        assert sum(map(len, partitions)) == 10030


def fake_shard(tmp_path, name, task, checkpoint='same', backend='inprocess', renderer='egl'):
    directory = tmp_path/name
    (directory/'results/libero_goal').mkdir(parents=True)
    launch = {'checkpoint_sha256': checkpoint, 'selection_sha256': 'selection',
        'global_expected': 2, 'shards': 2, 'env_backend': backend, 'render_backend': renderer}
    row = {'suite': 'libero_goal', 'category': 'camera', 'task_id': task,
        'successes': 1, 'error': None, 'env_backend': backend, 'render_backend': renderer}
    (directory/'launch.json').write_text(json.dumps(launch))
    (directory/'plan.json').write_text(json.dumps([row]))
    (directory/f'results/libero_goal/task_{task}.json').write_text(json.dumps(row))
    return directory


def test_shard_summary_checks_overlap_identity_and_finality(tmp_path):
    summarize = summary_module().summarize
    a = fake_shard(tmp_path, 'a', 0)
    b = fake_shard(tmp_path, 'b', 1)
    report = summarize([a,b])
    assert report['completed'] == report['successes'] == 2
    assert report['all_tasks_completed'] and not report['completion_markers_present']
    assert not summarize([a])['all_tasks_completed']
    (a/'exits.json').write_text('[-6]')
    assert summarize([a])['process_failures'] == 1
    assert not summarize([a])['technically_complete']
    with pytest.raises(ValueError, match='Duplicate'):
        summarize([a,a])
    c = fake_shard(tmp_path, 'c', 1, checkpoint='another')
    with pytest.raises(ValueError, match='checkpoint'):
        summarize([a,c])


def test_summary_rejects_different_rendering_backends(tmp_path):
    a = fake_shard(tmp_path, 'a', 0)
    b = fake_shard(tmp_path, 'b', 1, backend='subprocess')
    with pytest.raises(ValueError, match='env_backend'):
        summary_module().summarize([a, b])
    receipt = json.loads((b/'launch.json').read_text())
    receipt['env_backend'] = 'inprocess'
    (b/'launch.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='env_backend'):
        summary_module().summarize([b])


def test_rendering_backend_default_preserves_original(monkeypatch):
    from eval.libero_plus.runner import evaluation_backend
    monkeypatch.delenv('EVAL_ENV_BACKEND', raising=False)
    assert evaluation_backend() == 'inprocess'
    monkeypatch.setenv('EVAL_ENV_BACKEND', 'subprocess')
    assert evaluation_backend() == 'subprocess'
    monkeypatch.setenv('EVAL_ENV_BACKEND', 'unknown')
    with pytest.raises(ValueError, match='EVAL_ENV_BACKEND'):
        evaluation_backend()


def test_summary_rejects_mixing_egl_and_osmesa(tmp_path):
    a = fake_shard(tmp_path, 'a', 0, renderer='egl')
    b = fake_shard(tmp_path, 'b', 1, renderer='osmesa')
    with pytest.raises(ValueError, match='render_backend'):
        summary_module().summarize([a, b])
    assert summary_module().summarize([b])['render_backend'] == 'osmesa'


def test_gpu_guard_never_accepts_occupied_or_missing_device():
    from eval.libero_plus.runner import check_gpus
    with patch('subprocess.check_output', return_value='0, 0\n1, 4096\n'):
        assert check_gpus('0') == [0]
        with pytest.raises(RuntimeError, match='occupied'):
            check_gpus('1')
        with pytest.raises(ValueError, match='does not exist'):
            check_gpus('2')
    with pytest.raises(ValueError):
        check_gpus('0,0')


def test_launcher_rejects_reserved_overrides(monkeypatch, tmp_path):
    for key in ('WAM_PRETRAIN_CHECKPOINT','STVAE_CHECKPOINT','DINO_ENCODER_CHECKPOINT',
        'DIFFSYNTH_MODEL_BASE_PATH','LIBERO_DATA_ROOT','LIBERO_TEXT_CACHE',
        'LIBERO_WAN_VAE_CACHE','LIBERO_STVAE_CACHE'):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv('RUN_ID', 'pytest_launch')
    monkeypatch.setenv('WAM_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setenv('MASTER_ADDR', '127.0.0.1')
    monkeypatch.setenv('DRY_RUN', '1')
    for arg in ('output_dir=/tmp/elsewhere','++output_dir=/tmp/elsewhere','task=other','resume=auto'):
        result = subprocess.run(['bash', str(ROOT/'scripts/train_libero.sh'),arg],
            text=True, capture_output=True)
        assert result.returncode != 0 and 'Reserved override' in result.stderr
