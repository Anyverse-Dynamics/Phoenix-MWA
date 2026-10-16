"""File bytes/digests must not gate runtime; architecture still must match."""
import copy
from pathlib import Path
from unittest.mock import patch
import pytest


def test_encoder_digest_api_does_not_read_file(tmp_path):
    from wam.model.modules.codecs.dino_stvae.online_dinov3_stvae_encoder import checkpoint_sha256
    from wam.model.modules.codecs.dino_stvae.provenance import verify_bundle
    path = tmp_path/'encoder.pt'
    path.write_bytes(b'not deserialized by this existence check')
    with patch.object(Path, 'open', side_effect=AssertionError('content read')):
        assert checkpoint_sha256(path) is None
        assert verify_bundle()['hash_validation'] is False
    with pytest.raises(FileNotFoundError):
        checkpoint_sha256(tmp_path/'missing.pt')


def test_weight_restore_ignores_digests_but_not_shapes():
    from policy_fixture import policy
    from wam.training.checkpoint import joint96_checkpoint_metadata, _joint96_identity
    from wam.utils.runtime_contract import semantic_contract
    model = policy()
    original = joint96_checkpoint_metadata(model)
    changed = copy.deepcopy(original)
    encoder = changed['model_paths']['joint_visual_encoder']
    encoder.pop('source_sha256', None)
    encoder['stvae'].pop('teacher_sha256', None)
    encoder['stvae'].pop('stvae_checkpoint_sha256', None)
    assert semantic_contract(_joint96_identity(changed)) == semantic_contract(_joint96_identity(original))
    changed['action_dim'] = 7
    assert semantic_contract(_joint96_identity(changed)) != semantic_contract(_joint96_identity(original))


def test_public_launcher_has_no_required_digest():
    root = Path(__file__).resolve().parents[1]
    text = (root/'scripts/eval_libero_plus.sh').read_text()
    assert 'EVAL_CHECKPOINT_SHA256:?' not in text


def test_direct_preparation_needs_no_hash_manifest(tmp_path):
    import json
    import torch
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from tools.prepare_eval import prepare
    checkpoint=tmp_path/'policy.pt'
    torch.save({'step':60000},checkpoint)
    config=tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'joint96_posttrain':{'expected_pretrain_step':155000}}),config)
    stats=tmp_path/'stats.json';stats.write_text('{}')
    encoder=tmp_path/'encoder.pt';encoder.write_bytes(b'no digest needed during preparation')
    benchmark=tmp_path/'benchmark'
    for item in ('assets','bddl_files','init_files'):(benchmark/item).mkdir(parents=True)
    args=SimpleNamespace(checkpoint=checkpoint,config=config,dino=encoder,stvae=encoder,
        model_base=tmp_path,output=tmp_path/'runtime',benchmark='libero-plus',stats=stats,
        benchmark_root=benchmark,render_backend='egl')
    result=prepare(args)
    assert result.is_file() and 'EVAL_CHECKPOINT_SHA256' not in result.read_text()
    assert json.loads((result.parent/'prepare.json').read_text())['hash_validation'] is False
