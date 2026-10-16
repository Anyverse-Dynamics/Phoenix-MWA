"""A venv interpreter must not be resolved into its base environment."""
import os
from pathlib import Path
import sys
import pytest
from tools.prepare_eval import interpreter_path


@pytest.mark.skipif(os.name == 'nt', reason='Unix venv executable symlinks')
def test_preserve_virtual_environment_python(tmp_path):
    launcher = tmp_path / 'venv' / 'bin' / 'python'
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable).resolve())
    assert interpreter_path(launcher) == launcher.absolute()
    assert interpreter_path(launcher) != launcher.resolve()


def test_missing_interpreter(tmp_path):
    with pytest.raises(FileNotFoundError):
        interpreter_path(tmp_path / 'missing')
