import pytest
from eval.libero_plus.rendering import render_backend, install_render_fix


def test_osmesa_is_explicit_and_never_installs_egl_fix(monkeypatch):
    for key in ('EVAL_RENDER_BACKEND', 'MUJOCO_GL', 'PYOPENGL_PLATFORM'):
        monkeypatch.setenv(key, 'osmesa')
    assert render_backend() == 'osmesa'
    install_render_fix()  # No robosuite/EGL import is needed.


def test_backend_conflicts_and_unknown_backends_fail(monkeypatch):
    monkeypatch.setenv('EVAL_RENDER_BACKEND', 'osmesa')
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    with pytest.raises(ValueError, match='conflicts'):
        render_backend()
    monkeypatch.setenv('EVAL_RENDER_BACKEND', 'other')
    with pytest.raises(ValueError, match='must be'):
        render_backend()
