"""Read-only numeric scene descriptions for frozen episode comparison."""
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(type(value).__name__)


def scene_state(env):
    def row(obj, articulation=False):
        pose = obj.get_pose()
        result = dict(name=obj.get_name(), p=plain(pose.p), q=plain(pose.q))
        for key, method in [('velocity', 'get_velocity'), ('angular_velocity', 'get_angular_velocity')]:
            if hasattr(obj, method): result[key] = plain(getattr(obj, method)())
        if articulation:
            for key in ('qpos', 'qvel', 'qf', 'drive_target', 'drive_velocity_target'):
                method = 'get_'+key
                if hasattr(obj, method): result[key] = plain(getattr(obj, method)())
            result['links'] = [x.get_name() for x in obj.get_links()]
        return result
    return dict(actors=[row(x) for x in env.scene.get_all_actors()],
        articulations=[row(x, True) for x in env.scene.get_all_articulations()],
        setup_info=plain(env.info), clutter=plain(getattr(env,'record_cluttered_objects',[])),
        table_xy_bias=plain(env.table_xy_bias),
        cameras=[dict(name=c.get_name(),p=plain(c.get_pose().p),q=plain(c.get_pose().q))
                 for c in env.scene.get_cameras()])
