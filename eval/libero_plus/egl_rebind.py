"""Process-local EGL context ownership fix; no change to MuJoCo or observations."""
import ctypes
from functools import wraps

STATS = {'render_calls': 0, 'read_calls': 0, 'context_restores': 0,
         'null_context_restores': 0, 'owner_context_frees': 0, 'installed': False}

def handle(value):
    return ctypes.cast(value, ctypes.c_void_p).value or 0


def free_with_owner_context(owner, original_free, egl):
    """A dead scene must not delete GL buffers belonging to the active scene.

    robosuite 1.4 frees MjrContext before making its GL context current.
    Cyclic garbage collection after reset can therefore free same-numbered
    buffers in a newer scene. Preserve and restore the caller's EGL binding.
    """
    gl = getattr(owner, 'gl_ctx', None)
    own = getattr(gl, '_context', None)
    if not handle(own):
        return original_free(owner)
    previous = egl.eglGetCurrentContext()
    display = egl.eglGetCurrentDisplay()
    draw = egl.eglGetCurrentSurface(egl.EGL_DRAW)
    read = egl.eglGetCurrentSurface(egl.EGL_READ)
    previous_id, own_id = handle(previous), handle(own)
    gl.make_current()
    try:
        original_free(owner)
        STATS['owner_context_frees'] += 1
        if STATS['owner_context_frees'] <= 5:
            print('EGL_CONTEXT_FREED_WITH_OWNER', flush=True)
    finally:
        if previous_id and previous_id != own_id:
            if not egl.eglMakeCurrent(display, draw, read, previous):
                raise RuntimeError('Cannot restore live EGL context after old-scene cleanup')

def install(*, owner_cleanup=False):
    if STATS['installed']:
        return
    from robosuite.utils import binding_utils
    from robosuite.renderers.context import egl_context
    EGL = egl_context.EGL
    original_free = binding_utils.MjRenderContext.__del__
    def owned_free(self):
        return free_with_owner_context(self, original_free, EGL)
    # Experimental diagnostic only: ownership cleanup did not resolve the
    # observed readback abort. Keep the upstream destructor by default.
    if owner_cleanup:
        binding_utils.MjRenderContext.__del__ = owned_free

    def bind(owner):
        expected = handle(owner.gl_ctx._context)
        assert expected, 'Rendering context has already been freed'
        current = handle(EGL.eglGetCurrentContext())
        if current != expected:
            STATS['context_restores'] += 1
            STATS['null_context_restores'] += int(current == 0)
            if STATS['context_restores'] <= 5:
                print(f'EGL_CONTEXT_RESTORED current={current} expected={expected}', flush=True)
        owner.gl_ctx.make_current()
        assert handle(EGL.eglGetCurrentContext()) == expected

    for method, counter in (('render', 'render_calls'), ('read_pixels', 'read_calls')):
        original = getattr(binding_utils.MjRenderContext, method)
        def factory(func, count_key):
            @wraps(func)
            def wrapped(self, *args, **kwargs):
                bind(self)
                STATS[count_key] += 1
                return func(self, *args, **kwargs)
            return wrapped
        setattr(binding_utils.MjRenderContext, method, factory(original, counter))
    STATS['installed'] = True
    print('EGL_CONTEXT_REBIND_INSTALLED process-local render+read_pixels', flush=True)
