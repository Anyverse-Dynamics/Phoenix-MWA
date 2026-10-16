import pytest

from tools.prepare_eval import metadata_mapping


def test_relocated_metadata_mapping(tmp_path):
    target = tmp_path / 'task_a' / 'meta' / 'tasks.jsonl'
    target.parent.mkdir(parents=True)
    target.write_text('{}\n')
    source = '/old_dataset/task_a/meta/tasks.jsonl'
    assert metadata_mapping({'metadata': {source: {}}}, tmp_path) == {
        source: str(target.resolve())}


def test_missing_metadata_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        metadata_mapping({'metadata': {'/old_dataset/task_a/meta/tasks.jsonl': {}}}, tmp_path)


def test_invalid_metadata_layout_fails(tmp_path):
    with pytest.raises(ValueError, match='layout'):
        metadata_mapping({'metadata': {'/old_dataset/task_a/other/tasks.jsonl': {}}}, tmp_path)


def test_preparer_emits_mapping_accepted_by_runtime(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from tools.prepare_eval import prepare
    from wam.posttrain.robocasa_paths import resolve_source

    checkpoint = tmp_path / 'policy.pt'
    torch.save({'step': 45000}, checkpoint)
    config = tmp_path / 'config.yaml'
    config.write_text('joint96_posttrain:\n  expected_pretrain_step: 110000\n')
    package = tmp_path / 'package'
    (package / 'text').mkdir(parents=True)
    source = '/old_dataset/task_a/meta/tasks.jsonl'
    (package / 'package.json').write_text(json.dumps({'metadata': {source: {}}}))
    (package / 'norm.json').write_text('{}')
    (package / 'text' / 'COMPLETE.json').write_text('{}')
    metadata = tmp_path / 'metadata'
    target = metadata / 'task_a' / 'meta' / 'tasks.jsonl'
    target.parent.mkdir(parents=True)
    target.write_text('{}\n')
    output = tmp_path / 'prepared'
    prepare(SimpleNamespace(checkpoint=checkpoint, config=config, dino=checkpoint,
        stvae=checkpoint, model_base=tmp_path, output=output, benchmark='robocasa-gr1',
        package=package, resources=tmp_path, scene_seeds=config, render_backend='egl',
        sim_python=None, metadata_root=metadata))
    monkeypatch.setenv('ROBOCASA_PATH_MAP', str(output / 'paths.json'))
    assert resolve_source(source) == target.resolve()
    assert 'ROBOCASA_PATH_MAP=' in (output / 'eval.env.sh').read_text()
