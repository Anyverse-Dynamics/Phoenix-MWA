"""Align trusted Prometheus recordings and write independent LeRobot v3 shards."""
import pickle
from pathlib import Path

import numpy as np

from .common import atomic_json, digest, directory_guard, inventory, read_json, sha256

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
FEATURES = tuple("observation.images." + key for key in CAMERAS)
JOINT_NAMES = ([f"left_joint_{i}" for i in range(6)] + ["left_gripper"]
               + [f"right_joint_{i}" for i in range(6)] + ["right_gripper"])
ARM_INDICES = [*range(6), *range(7, 13)]


def load_pickle(path):
    # Prometheus uses pickle for local acquisition. Only open trusted recordings.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def nearest_indices(stamps, targets):
    right = np.searchsorted(stamps, targets).clip(0, len(stamps) - 1)
    left = (right - 1).clip(0)
    return np.where(np.abs(stamps[left] - targets) <= np.abs(stamps[right] - targets), left, right)


def clean_robot(stream):
    joints, times = stream["joint"], stream["timestamps"]
    if len(joints) != len(times):
        raise ValueError("Robot state/timestamp counts differ")
    rows = []
    for joint, stamp in zip(joints, times):
        value = np.asarray(joint, dtype=np.float64)
        if value.shape == (7,) and np.isfinite(value).all() and np.isfinite(stamp):
            rows.append((float(stamp), value))
    rows.sort(key=lambda row: row[0])
    unique = {stamp: joint for stamp, joint in rows}
    if len(unique) < 2:
        raise ValueError("Insufficient finite robot observations")
    return np.array(list(unique)), np.stack(list(unique.values()))


def motion_boundary(states, options, *, reverse=False):
    if reverse:
        states = states[::-1]
    delta = np.abs(states - states[0])
    moving = ((np.linalg.norm(delta[:, ARM_INDICES], axis=1) > options["arm_motion_threshold"])
              | (delta[:, [6, 13]].max(axis=1) > options["gripper_motion_threshold"]))
    sustain = int(options["motion_sustain_frames"])
    if sustain < 1 or len(states) < sustain:
        raise ValueError("Invalid motion window")
    runs = np.convolve(moving.astype(np.int64), np.ones(sustain, np.int64), mode="valid")
    starts = np.flatnonzero(runs >= sustain)
    if not len(starts):
        raise ValueError("No sustained motion in episode")
    roll = options["motion_postroll_frames" if reverse else "motion_preroll_frames"]
    return max(0, int(starts[0]) - int(roll))


def plan_episode(path, options):
    root = Path(path).resolve()
    flat = (root / "robot_state_dict.pkl").is_file()
    robot_path = root / ("robot_state_dict.pkl" if flat else "robot/robot_state_dict.pkl")
    stamps_path = root / ("image_timestamps.pkl" if flat else "camera/timestamp.pkl")
    camera_root = root if flat else root / "camera"
    robot, raw_times = load_pickle(robot_path), load_pickle(stamps_path)
    robot_streams = [clean_robot(robot[key]) for key in options["robot_parts"]]
    if len(robot_streams) != 2:
        raise ValueError("Phoenix requires left and right seven-dimensional robot parts")
    camera_times, videos = [], []
    for name in options["camera_streams"]:
        values = raw_times.get(name, raw_times.get(name.removesuffix("_rgb")))
        stamps = np.asarray(values, dtype=np.float64)
        if stamps.ndim != 1 or len(stamps) < 2 or not np.isfinite(stamps).all() or (np.diff(stamps) <= 0).any():
            raise ValueError("Camera timestamps must be finite, increasing milliseconds")
        video = camera_root / (name + ".mp4")
        if not video.is_file():
            raise FileNotFoundError(video)
        camera_times.append(stamps)
        videos.append(video)
    if len(videos) != 3:
        raise ValueError("Exactly three RGB streams are required")
    all_stamps = [row[0] for row in robot_streams] + camera_times
    first, last = max(t[0] for t in all_stamps), min(t[-1] for t in all_stamps)
    count = int(np.floor((last - first) * 30 / 1000)) + 1
    if count < 2:
        raise ValueError("No common recording interval")
    clock = first + np.rint(np.arange(count) * 1000 / 30)
    indices = [nearest_indices(times, clock) for times in all_stamps]
    deviations = [float(np.abs(t[idx] - clock).max()) for t, idx in zip(all_stamps, indices)]
    if (max(deviations[:2]) > options["max_robot_alignment_ms"]
            or max(deviations[2:]) > options["max_rgb_alignment_ms"]):
        raise ValueError("Recording exceeds the configured alignment tolerance")
    states = np.concatenate([joint[index] for (_, joint), index in zip(robot_streams, indices[:2])], axis=1)
    begin = motion_boundary(states, options) if options["trim_static_start"] else 0
    end = len(states) - motion_boundary(states, options, reverse=True) if options["trim_static_end"] else len(states)
    states = states[begin:end]
    if len(states) - 1 < options["min_output_frames"]:
        raise ValueError("Episode is too short after alignment/trimming")
    if (np.abs(states[:, ARM_INDICES]) > options["max_abs_joint"]).any():
        raise ValueError("Joint magnitude exceeds the configured bound")
    if (np.linalg.norm(np.diff(states[:, ARM_INDICES], axis=0), axis=1) > options["max_joint_step_l2"]).any():
        raise ValueError("Discontinuous robot trajectory")
    files = [robot_path, stamps_path, *videos]
    return dict(state=states[:-1].astype(np.float32), action=states[1:].astype(np.float32),
                videos=videos, indices=[idx[begin:end - 1] for idx in indices[2:]],
                source_sha256={str(p): sha256(p) for p in files},
                alignment_max_ms=deviations, dropped_terminal_frames=1)


class SelectedVideoReader:
    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        self.index, self.frame = -1, None
        if not self.cap.isOpened():
            raise ValueError("Cannot open RGB recording")

    def get(self, index):
        if index < self.index:
            raise ValueError("Video indices must be monotonic")
        while self.index < index:
            ok, self.frame = self.cap.read()
            if not ok:
                raise ValueError("Short or corrupt RGB recording")
            self.index += 1
        rgb = self.cv2.cvtColor(self.frame, self.cv2.COLOR_BGR2RGB)
        return self.cv2.resize(rgb, (256, 256), interpolation=self.cv2.INTER_AREA)

    def close(self):
        self.cap.release()


def convert(config):
    if (config["gripper_unit"] not in ("m", "mm", "normalized", "source")
            or type(config["min_output_frames"]) is not int or config["min_output_frames"] <= 4
            or len(set(config["camera_streams"])) != 3 or len(set(config["robot_parts"])) != 2):
        raise ValueError("Declare source units, two robot parts, three cameras, and at least five output frames")
    for key in ("max_robot_alignment_ms", "max_rgb_alignment_ms", "max_abs_joint", "max_joint_step_l2"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid conversion threshold: {key}")
    from importlib.metadata import version
    if version("lerobot") != "0.4.3":
        raise ValueError("Use the dedicated conversion environment with lerobot==0.4.3")
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    if CODEBASE_VERSION != "v3.0" or not hasattr(LeRobotDataset, "finalize"):
        raise ValueError("Native LeRobot v3 writer with finalize() required")
    episodes = read_json(config["episodes"])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Expected a nonempty episode/instruction array")
    source_roots = [str(Path(e["path"]).resolve()) for e in episodes]
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("Duplicate recording episodes")
    output = Path(config["output"]).resolve()
    if any(output == Path(p) or output.is_relative_to(p) for p in source_roots):
        raise ValueError("Conversion output must be outside source episodes")
    output.mkdir(parents=True, exist_ok=True)
    roots = []
    with directory_guard(output / "convert.owner"):
        for number, entry in enumerate(episodes):
            if not isinstance(entry.get("instruction"), str) or not entry["instruction"].strip():
                raise ValueError("Provide the exact instruction for every episode")
            plan = plan_episode(entry["path"], config)
            contract = dict(source=plan["source_sha256"], instruction=entry["instruction"],
                            settings={k: v for k, v in config.items() if k not in ("output", "episodes")},
                            converter_sha256=sha256(__file__), lerobot_version="0.4.3")
            target = output / f"episode_{number:06d}"
            receipt_path = target / "CONVERTED.json"
            if target.exists():
                receipt = read_json(receipt_path)
                if receipt["contract"] != contract or receipt["files"] != inventory(target):
                    raise ValueError("Existing conversion is incomplete or belongs to different inputs")
                roots.append(str(target))
                continue
            partial = output / f"episode_{number:06d}.partial"
            if partial.exists():
                raise FileExistsError("Inspect/remove the incomplete .partial directory before retrying")
            features = {key: dict(dtype="float32", shape=(14,), names=JOINT_NAMES)
                        for key in ("observation.state.joint", "action_abs_joint")}
            features.update({key: dict(dtype="video", shape=(3, 256, 256), names=["channels", "height", "width"])
                             for key in FEATURES})
            dataset = LeRobotDataset.create(repo_id=f"phoenix/episode_{number:06d}", root=partial,
                fps=30, robot_type="bimanual_joint14", features=features, use_videos=True,
                vcodec="h264", image_writer_threads=config["image_writer_threads"])
            readers = []
            try:
                readers = [SelectedVideoReader(p) for p in plan["videos"]]
                for frame_index, (state, action) in enumerate(zip(plan["state"], plan["action"])):
                    frame = {"observation.state.joint": state, "action_abs_joint": action,
                             "task": entry["instruction"]}
                    for key, reader, indices in zip(FEATURES, readers, plan["indices"]):
                        rgb = reader.get(int(indices[frame_index]))
                        gray = rgb.astype(np.float32).mean(axis=2)
                        if (gray.std() < config["min_image_std"] or gray.mean() < config["min_image_mean"]
                                or gray.mean() > config["max_image_mean"]):
                            raise ValueError("Blank/saturated RGB frame")
                        frame[key] = rgb
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for reader in readers:
                    reader.close()
                try:
                    dataset.finalize()
                finally:
                    dataset.stop_image_writer()
            # A receipt is published only after footer finalization and source recheck.
            if any(sha256(p) != value for p, value in plan["source_sha256"].items()):
                raise ValueError("Recording changed during conversion")
            atomic_json(partial / "CONVERTED.json",
                        dict(schema="phoenix.conversion.v1", contract=contract, files=inventory(partial),
                             frames=len(plan["state"]), action_semantics="materialized_next_observation",
                             gripper_unit=config["gripper_unit"], joint_unit="radian",
                             alignment_max_ms=plan["alignment_max_ms"]))
            partial.rename(target)
            roots.append(str(target))
        atomic_json(output / "dataset_dirs.json", roots)
    return dict(status="CONVERTED", datasets=len(roots), dataset_list=str(output / "dataset_dirs.json"))
