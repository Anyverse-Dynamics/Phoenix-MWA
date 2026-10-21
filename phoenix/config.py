from pathlib import Path
import json
import math

HORIZON = 32
VIDEO_FRAMES = 9
DENOISING_STEPS = 10
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
JOINT_SLOTS = [*range(6), 16, *range(40, 46), 56]


def read_config(path):
    from omegaconf import OmegaConf
    path = Path(path).resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg.get("schema") != "phoenix.inference.v1":
        raise ValueError("Unsupported Phoenix configuration")
    if (cfg.get("action_horizon"), cfg.get("video_frames"), cfg.get("denoising_steps")) != (32, 9, 10):
        raise ValueError("Phoenix requires H32, video_frames=9, denoising_steps=10")
    if cfg.get("action_hz") != 30:
        raise ValueError("Phoenix control configuration requires 30Hz")
    rtc = cfg["rtc"]
    s, d = rtc["execution_horizon"], rtc["inference_delay"]
    if type(s) is not int or type(d) is not int or not 0 < d < s or s+d > 32:
        raise ValueError("RTC requires 0<d<s and d+s<=32")
    if rtc["method"] != "condition" or type(rtc["max_prefix_steps"]) is not int or not d<=rtc["max_prefix_steps"]<32:
        raise ValueError("Training RTC requires condition and d<=max_prefix_steps<32")
    for key in ("checkpoint", "model_config", "normalization", "text_context"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(value if value.is_absolute() else path.parent.parent / value)
    return cfg


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
