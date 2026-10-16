from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from tools.prepare_release_eval import Release, prepare, relocate_acceptance, relocate_metadata, sha256


def fake_release(tmp_path, files):
    root = tmp_path / 'release'
    root.mkdir()
    manifest = {'schema': 'phoenix.eval_release.v1', 'complete': True, 'files': {}}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest['files'][name] = {'sha256': sha256(path), 'bytes': len(content)}
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    return root, manifest


@pytest.mark.parametrize('issue', ['missing_complete', 'false_complete', 'INCOMPLETE.json', 'INCOMPLETE'])
def test_release_rejects_unfinished_export(tmp_path, issue):
    root, manifest = fake_release(tmp_path, {})
    if issue == 'missing_complete':
        del manifest['complete']
    elif issue == 'false_complete':
        manifest['complete'] = False
    else:
        (root / issue).write_text('in progress')
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        Release(root)


def test_release_rejects_modified_files_and_path_escape(tmp_path):
    root, _ = fake_release(tmp_path, {'model/policy.pt': b'policy'})
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'secret')
    (root / 'outside.pt').symlink_to(outside)
    for name in ('../outside.pt', '/outside.pt', 'outside.pt'):
        with pytest.raises(ValueError):
            Release(root).file(name)
    (root / 'model/policy.pt').write_bytes(b'POLICY')
    assert Release(root).file('model/policy.pt').read_bytes() == b'POLICY'


def test_relocated_acceptance_keeps_failed_parity_and_checks(tmp_path):
    source = {'checkpoint': '/old/policy/step_045000.pt', 'original_policy_sha256': 'a' * 64,
              'passed': False, 'inference_passed': True,
              'samples': [{'finite_actions': True, 'checks': {'numerical': {'passed': False}}}]}
    before = deepcopy(source)
    model = {'source_sha256': 'a' * 64, 'sha256': 'b' * 64}
    runtime = relocate_acceptance(source, model, tmp_path / 'policy.pt', 'c' * 64)
    assert source == before
    assert runtime['passed'] is False
    assert runtime['samples'] == before['samples']
    assert runtime['source_checkpoint'] == source['checkpoint']
    assert runtime['release_policy_sha256'] == model['sha256']
    relocate_acceptance(source, {**model, 'source_sha256': 'd' * 64}, tmp_path / 'policy.pt', 'c' * 64)


def test_downloaded_metadata_restores_mtime_without_touching_release(tmp_path):
    content = b'{"remarks":"place the object"}\n'
    relative = 'model/metadata/task/meta/episodes.jsonl'
    root, _ = fake_release(tmp_path, {relative: content})
    original = '/old/data/tasks/task/meta/episodes.jsonl'
    record = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content),
              'mtime_ns': 1779949617791111448}
    package = {'metadata': {original: record}}
    before = (root / relative).stat().st_mtime_ns
    output = tmp_path / 'runtime'
    mapping = relocate_metadata(Release(root), 'model', package, output)
    restored = Path(mapping['mappings'][original])
    assert restored.read_bytes() == content and restored.stat().st_mtime_ns == record['mtime_ns']
    assert (root / relative).stat().st_mtime_ns == before


def test_libero_runtime_preserves_algorithm_and_requires_no_private_init(tmp_path):
    cfg = {'init_checkpoint': '/private/pretrain.pt', 'resume': None,
           'joint96_posttrain': {'expected_pretrain_step': 135000},
           'model': {'video_scheduler': {'infer_shift': 6}, 'action_scheduler': {'infer_shift': 1},
                     'joint_visual_codec': {'expected_stvae_step': 150000}}}
    files = {'model/policy.pt': b'policy', 'model/train_config.yaml': OmegaConf.to_yaml(cfg).encode(),
             'encoders/dino.pt': b'dino', 'encoders/stvae.pt': b'stvae',
             **{f'model/{name}': b'{}' for name in
                ('dataset_stats.json', 'normalization_source.json', 'initialization.json')}}
    root, manifest = fake_release(tmp_path, files)
    manifest['models'] = {'libero': {'checkpoint': 'model/policy.pt', 'config': 'model/train_config.yaml',
        'sha256': manifest['files']['model/policy.pt']['sha256'], 'step': 35000, 'pretrain_step': 135000}}
    manifest['encoders'] = {name: {'path': f'encoders/{name}.pt',
        'sha256': manifest['files'][f'encoders/{name}.pt']['sha256']} for name in ('dino', 'stvae')}
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    benchmark = tmp_path / 'benchmark'
    for name in ('assets', 'bddl_files', 'init_files'):
        (benchmark / name).mkdir(parents=True)
    args = SimpleNamespace(release_root=root, output=tmp_path / 'runtime', model='libero',
        model_base=tmp_path, libero_benchmark_root=benchmark, policy_python='/python',
        render_backend='osmesa')
    receipt = prepare(args)
    runtime = OmegaConf.load(receipt['config'])
    assert runtime.init_checkpoint is None
    assert runtime.model.video_scheduler.infer_shift == 6 and runtime.model.action_scheduler.infer_shift == 1
    assert runtime.model.joint_visual_codec.expected_stvae_step == 150000
    assert (args.output / 'policy.pt').resolve() == root / 'model/policy.pt'
    assert not receipt['evaluation_launched'] and not receipt['private_initialization_required']
    assert (root / 'model/train_config.yaml').read_bytes() == files['model/train_config.yaml']
    assert 'JOINT96_EVAL_CHECKPOINT_FILENAME=policy.pt' in Path(receipt['env']).read_text()
