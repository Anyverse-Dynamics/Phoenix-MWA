"""Explicit rendering selection: H20 EGL or the original 929 eval OSMesa."""
import os


def render_backend():
    backend = os.environ.get('EVAL_RENDER_BACKEND', os.environ.get('MUJOCO_GL', 'egl'))
    if backend not in ('egl', 'osmesa'):
        raise ValueError('EVAL_RENDER_BACKEND must be egl or osmesa')
    for key in ('MUJOCO_GL', 'PYOPENGL_PLATFORM'):
        value = os.environ.get(key)
        if value and value != backend:
            raise ValueError(f'{key}={value} conflicts with EVAL_RENDER_BACKEND={backend}')
    return backend


def install_render_fix():
    if render_backend() == 'egl':
        from eval.libero_plus import egl_rebind
        egl_rebind.install()


def probe(output):
    from robosuite.utils.binding_utils import GLContext
    from OpenGL import GL
    from eval.libero_plus.runner import dump
    backend = render_backend()
    context = GLContext(max_width=256, max_height=256, device_id=-1)
    try:
        context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        version = GL.glGetString(GL.GL_VERSION).decode()
        context_class = f'{type(context).__module__}.{type(context).__name__}'
        if f'.{backend}_context.' not in context_class:
            raise RuntimeError(f'Expected {backend} context, got {context_class}')
        if backend == 'egl' and ('NVIDIA' not in vendor or 'llvmpipe' in renderer.lower()):
            raise RuntimeError(f'Expected NVIDIA EGL, got {vendor} / {renderer}')
        report = dict(backend=backend, vendor=vendor, renderer=renderer, gl_version=version,
            context_class=context_class, rendering_device='cpu' if backend == 'osmesa' else 'gpu',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            egl_device_id=os.environ.get('MUJOCO_EGL_DEVICE_ID'))
        dump(output/'render_context.json', report)
        if backend == 'egl':
            dump(output/'egl_context.json', report)
        print(f'RENDER_CONTEXT_OK backend={backend} renderer={renderer}', flush=True)
        return report
    finally:
        context.free()
