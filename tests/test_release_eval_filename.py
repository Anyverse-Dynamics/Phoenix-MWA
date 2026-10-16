from pathlib import Path

import pytest

from eval.libero_plus.runner import evaluation_checkpoint_path


def test_checkpoint_filename_supports_release_and_legacy_names():
    root = Path('/release/libero/policy')
    assert evaluation_checkpoint_path(root, 35000) == root / 'step_035000.pt'
    assert evaluation_checkpoint_path(root, 35000, 'policy.pt') == root / 'policy.pt'


@pytest.mark.parametrize('filename', ['', '.', '..', '/tmp/policy.pt', '../policy.pt',
                                     'nested/policy.pt', r'..\policy.pt', 'policy\x00.pt'])
def test_checkpoint_filename_rejects_paths(filename):
    with pytest.raises(ValueError, match='single filename'):
        evaluation_checkpoint_path('/release/libero/policy', 35000, filename)
