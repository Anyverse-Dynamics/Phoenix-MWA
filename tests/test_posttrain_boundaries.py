import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from omegaconf import OmegaConf
from wam.posttrain.eval_contract import validate_evaluation_contract
from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_is_self_contained_and_unchanged():
    from wam.model.modules.codecs.dino_stvae.provenance import verify_bundle
    manifest = verify_bundle()
    assert manifest['hash_validation'] is False


def test_no_pretraining_entry_or_reader():
    assert not (ROOT/'configs/pretrain').exists()
    assert not list((ROOT/'src/wam/datasets').glob('pretrain*.py'))
    assert not list((ROOT/'src/wam/datasets').glob('local_robot*.py'))
    from wam.posttrain.guard import validate_scope
    with pytest.raises(ValueError, match='post-training only'):
        validate_scope(OmegaConf.create({'local_robot_joint96': {'enabled': True}}))


def test_import_guard_blocks_external_fastwam():
    process = subprocess.run([sys.executable, '-B', '-c',
        'from wam.posttrain.guard import install_import_guard; install_import_guard(); import fastwam'],
        cwd=ROOT, text=True, capture_output=True)
    assert process.returncode != 0 and 'bundled encoder' in process.stderr


def test_task_selection_is_exactly_the_existing_1003():
    content = json.loads((ROOT/'assets/libero_plus_balanced10_manifest.json').read_text())
    keys = sorted((r['suite'], r['task_id']) for r in content['tasks'])
    assert len(keys) == len(set(keys)) == content['total_selected'] == 1003
    assert hashlib.sha256(json.dumps(keys).encode()).hexdigest() == 'e8a65b76e955ab6359ccb6e08a2659bdb344c50027c3881799fd5ffc50749c30'


@pytest.mark.parametrize('step', [135000, 145000, 155000])
def test_generic_eval_guard_keeps_step_frame_and_stats_strict(tmp_path, step):
    cfg = OmegaConf.create({'joint96_posttrain': {'expected_pretrain_step': step,
        'libero_frame_contract': LIBERO_FLU_BI_CONTRACT['version']},
        'data': {split: {'processor': {'action_state_transforms': [{'_target_':
            'wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform'}]}}
            for split in ('train','val')}})
    path = tmp_path/'stats.json'
    path.write_text(json.dumps({'frame_contract': LIBERO_FLU_BI_CONTRACT}))
    contract = {'frame_contract': LIBERO_FLU_BI_CONTRACT['version'], 'embodiment_id': 0,
        'domain': 'libero', 'normalization_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = {'joint96_metadata': {'soft_prompt': {'training_contract': contract}}}
    validate_evaluation_contract(cfg, payload, path, expected_pretrain_step=step)
    with pytest.raises(ValueError, match='pretraining recipe'):
        validate_evaluation_contract(cfg, payload, path, expected_pretrain_step=step+5000)
    contract['normalization_sha256'] = 'wrong'
    validate_evaluation_contract(cfg, payload, path, expected_pretrain_step=step)
    contract['frame_contract'] = 'wrong'
    with pytest.raises(ValueError):
        validate_evaluation_contract(cfg, payload, path, expected_pretrain_step=step)


def test_inference_core_is_bundled():
    assert (ROOT/'src/wam/inference/frame_joint.py').is_file()
