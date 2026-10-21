"""Read official protocol-4 NumPy init arrays without executing pickle code."""
from pathlib import Path
import pickletools
import zipfile


def safe_init_states(path):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        if sorted(Path(n).name for n in archive.namelist()) != ['data.pkl', 'version']:
            raise ValueError(f'Unsupported init-state archive: {path}')
        data = archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
    ops = list(pickletools.genops(data))
    allowed = {'PROTO', 'FRAME', 'SHORT_BINUNICODE', 'MEMOIZE', 'STACK_GLOBAL',
               'BININT1', 'BININT2', 'BININT', 'TUPLE1', 'SHORT_BINBYTES', 'TUPLE3',
               'REDUCE', 'MARK', 'TUPLE2', 'BINGET', 'LONG_BINGET', 'NEWFALSE',
               'NEWTRUE', 'NONE', 'TUPLE', 'BUILD', 'BINBYTES', 'STOP'}
    if not ops or ops[0][1] != 4 or any(op.name not in allowed for op, _, _ in ops):
        raise ValueError(f'Unsupported init-state serialization: {path}')
    strings = [arg for op, arg, _ in ops if op.name == 'SHORT_BINUNICODE']
    if strings[:5] != ['numpy.core.multiarray', '_reconstruct', 'numpy', 'ndarray', 'dtype']:
        raise ValueError(f'Expected numeric ndarray metadata: {path}')
    if len(strings) != 7 or strings[5] not in {'f8', 'f4'} or strings[6] not in {'<', '>', '='}:
        raise ValueError(f'Unsupported init-state dtype: {path}')
    counts = {name: sum(op.name == name for op, _, _ in ops)
              for name in ('STACK_GLOBAL', 'REDUCE', 'BUILD', 'TUPLE2')}
    if counts != {'STACK_GLOBAL': 3, 'REDUCE': 2, 'BUILD': 2, 'TUPLE2': 1}:
        raise ValueError(f'Unsupported ndarray structure: {path}')
    index = next(i for i, (op, _, _) in enumerate(ops) if op.name == 'TUPLE2')
    shape_ops = ops[index - 2:index]
    if len(shape_ops) != 2 or any(op.name not in {'BININT1', 'BININT2', 'BININT'} for op, _, _ in shape_ops):
        raise ValueError(f'Invalid init-state shape: {path}')
    shape = tuple(arg for _, arg, _ in shape_ops)
    buffers = [(i, arg) for i, (op, arg, _) in enumerate(ops) if op.name == 'BINBYTES']
    if len(buffers) != 1 or ops[buffers[0][0] - 1][0].name != 'NEWFALSE':
        raise ValueError(f'Expected a C-contiguous ndarray: {path}')
    dtype = np.dtype(strings[6] + strings[5])
    if not all(0 < d <= 10000 for d in shape) or len(buffers[0][1]) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f'Init-state buffer size mismatch: {path}')
    states = np.frombuffer(buffers[0][1], dtype=dtype).reshape(shape).copy()
    if states.ndim != 2 or len(states) != 50 or not np.isfinite(states).all():
        raise ValueError(f'Expected 50 finite official initial states: {path}')
    return states
