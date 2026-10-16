from types import SimpleNamespace

from eval.libero_plus.egl_rebind import free_with_owner_context


def test_old_context_free_binds_owner_and_restores_live_scene():
    state = {'context': 22}
    calls = []
    def bind(*args):
        state['context'] = args[-1]
        calls.append(('restore', state['context']))
        return True
    egl = SimpleNamespace(EGL_DRAW=1, EGL_READ=2,
        eglGetCurrentContext=lambda: state['context'],
        eglGetCurrentDisplay=lambda: 100,
        eglGetCurrentSurface=lambda key: 0,
        eglMakeCurrent=bind)
    def make_current():
        state['context'] = 11
        calls.append(('bind', 11))
    owner = SimpleNamespace(gl_ctx=SimpleNamespace(_context=11, make_current=make_current))
    def original_free(value):
        assert value is owner and state['context'] == 11
        calls.append(('free', 11))
        state['context'] = 0
    free_with_owner_context(owner, original_free, egl)
    assert state['context'] == 22
    assert calls == [('bind', 11), ('free', 11), ('restore', 22)]


def test_free_does_not_restore_the_context_just_destroyed():
    calls = []
    egl = SimpleNamespace(EGL_DRAW=1, EGL_READ=2,
        eglGetCurrentContext=lambda: 11, eglGetCurrentDisplay=lambda: 100,
        eglGetCurrentSurface=lambda key: 0,
        eglMakeCurrent=lambda *args: calls.append('incorrect_restore'))
    owner = SimpleNamespace(gl_ctx=SimpleNamespace(_context=11, make_current=lambda: None))
    free_with_owner_context(owner, lambda value: calls.append('free'), egl)
    assert calls == ['free']
