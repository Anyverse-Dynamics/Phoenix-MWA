"""Phoenix serialization namespace and read-only release audit regressions."""
import copy
import pytest


def test_checkpoint_namespace():
    from policy_fixture import policy
    from wam.training.checkpoint import joint96_checkpoint_metadata, _joint96_identity
    metadata = joint96_checkpoint_metadata(policy())
    assert metadata['schema'] == 'phoenix.local_joint96.weights.v1'
    _joint96_identity(metadata)
    legacy = copy.deepcopy(metadata)
    legacy['schema'] = 'intern' + 'w0.local_joint96.weights.v1'
    with pytest.raises(ValueError, match='incompatible'):
        _joint96_identity(legacy)


def test_audit_is_read_only(tmp_path):
    from tools.audit_public_source import inspect
    path = tmp_path / 'example.txt'
    text = '-----BEGIN ' + 'PRIVATE KEY-----'
    path.write_text(text)
    assert ('private_key', 1) in inspect(path, tmp_path)
    assert path.read_text() == text
