"""Explicit ARX release clock mapping; never relax the other source readers."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


ABC_TIMESTAMP_POLICY = '(camera_timestamp_ns-reference_origin_ns)/1e9+start_s'
ABC_CAMERA_MAX_OFFSET_NS = 50_000_000


def abc_camera_bounds(record, view):
    """Bound camera time by the published reference interval and alignment budget.

    ABC end_s is a historical reference-clock cutoff, not the final MP4 PTS.
    Native camera samples may lead the published reference interval by up
    to the already-reviewed 50ms. This does not relax decoded-frame matching.
    """
    origin = int(record['reference_origin_ns'])
    begin, end = int(record['original_common_from_ns']), int(record['original_common_to_ns'])
    offset = float(view['start_s'])
    reference_end = (end - origin) / 1e9 + offset
    if (begin >= end or view.get('timestamp_policy') != ABC_TIMESTAMP_POLICY
            or view.get('nominal_fps_only') is not True
            or int(view['reference_origin_ns']) != origin
            or not np.isfinite(offset) or not np.isfinite(reference_end)):
        raise ValueError('ABC common-reference boundary metadata drift')
    slack = ABC_CAMERA_MAX_OFFSET_NS / 1e9
    return max(0., (begin - origin) / 1e9 + offset - slack), reference_end + slack


def nearest_pts(pts, targets, fps):
    pts, targets = np.asarray(pts), np.asarray(targets)
    if len(pts) < 9 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError('ARX video has insufficient or invalid PTS')
    distance = abs(pts[:,None]-targets)
    order = np.argsort(distance,axis=0)
    errors = distance[order[0],np.arange(9)]
    next_errors = distance[order[1],np.arange(9)]
    if np.any(errors > .5/fps) or np.any(next_errors-errors < 1e-9):
        raise ValueError('ARX lacks a unique matching frame within the published half-frame bound')
    chosen=pts[order[0]]
    if np.any(np.diff(chosen)<=0):
        raise ValueError('ARX distinct state samples mapped to repeated video frames')
    return chosen


def arx_video_targets(record, start):
    import av
    source = Path(record['source_root'])
    release = next((p.parent for p in (source, *source.parents) if p.name == 'datasets'), None)
    if release is None:
        raise ValueError('ARX source must be inside a datasets directory')
    original_root = Path(os.environ.get('PHOENIX_ARX_ORIGINAL_ROOT',
                                        str(release.parent / 'arx456_xt_30hz_256')))
    fps=float(record['fps'])
    if fps!=30:
        raise ValueError('ARX clock mapping is restricted to the reviewed 30Hz dedup release')
    result={}
    for view in record['selected_views']:
        path=Path(view['path'])
        original=original_root/path.relative_to(release)
        if not os.path.samefile(path,original):
            raise ValueError('ARX video no longer matches its published original hardlink')
        targets=float(view['start_s'])+(start+np.arange(9)*4)/fps
        pts=[]
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            stream.codec_context.thread_count=1
            container.seek(max(0,int((targets[0]-1/fps)/float(stream.time_base))),stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('ARX video frame lacks PTS')
                stamp=float(frame.pts*stream.time_base)
                pts.append(stamp)
                if stamp>targets[-1]+1/fps:
                    break
        result[view['key']]=nearest_pts(pts,targets,fps)
    return result
