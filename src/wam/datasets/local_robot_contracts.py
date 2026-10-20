"""Explicit local-source contracts; no FastWAM action transforms are imported."""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from wam.resources import resource_path

from wam.datasets.pretrain_lerobot_loader import (
    PretrainLeRobotDataset, _adapter_columns, _merge_mapping,
)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "pretrain"
SCHEMA = "phoenix.local_robot80.v1"
SLOTS = ("main", "left", "right")


def sha256(path):
    h = hashlib.sha256()
    with resource_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def _profile_identities():
    from wam.resources import repository_root
    payload = json.loads((repository_root() / "configs/data_profile_ids.json").read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported profile identity map")
    return payload


def profile_identity(value, prefix):
    """Preserve published sample IDs for exactly matching canonical profiles."""
    content = {key: item for key, item in value.items() if key != "id"}
    fingerprint = digest(content)
    return _profile_identities()["profiles"].get(fingerprint, prefix + fingerprint[:16])


def normalization_content_sha256(identity):
    """Resolve the exact content checksum for a preserved normalization ID."""
    return _profile_identities().get("normalization_content_sha256", {}).get(identity, identity)


def metadata_content_sha256(identity):
    """Resolve rewritten metadata bytes while preserving its dataset identity."""
    return _profile_identities().get("metadata_content_sha256", {}).get(identity, identity)


def eligible_group(group):
    source, body = group["source"], group["signature"][1]
    return (
        source == "intern_data_a1" and body in {"Franka", "ARX Lift-2", "AgileX Split Aloha", "Genie-1"}
        or source == "agibot_gripper"
        or source == "robocoin" and body in {
            "agilex_cobot_decoupled_magic", "galaxea_r1_lite", "realman_rmc_aidal"}
        or source == "robomind2" and body in {"franka", "ur"}
    )


@lru_cache(maxsize=16)
def _yaml(relative):
    return yaml.safe_load((CONFIG_ROOT / relative).read_text())


def camera_policy(group):
    source, body, policy, keys = group["signature"][:4]
    if source != group["source"]:
        raise ValueError("source/signature mismatch")
    mapping = {
        "agibot_gripper_dual": ("observation.images.head", "observation.images.hand_left", "observation.images.hand_right"),
        "intern_data_a1_single": ("images.rgb.head", "", "images.rgb.hand"),
        "intern_data_a1_dual": ("images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"),
        "robomind2_03": ("observation.images.camera_front", "observation.images.camera_left", "observation.images.camera_right"),
        "robomind2_10": ("observation.images.camera_front", "", ""),
        "robomind2_11": ("observation.images.camera_front", "", ""),
        "robomind2_12": ("observation.images.camera_front", "observation.images.camera_wrist_left", "observation.images.camera_wrist_right"),
    }
    for number in (0, 1, 2, 3, 7, 8, 9):
        suffix = "_rgb_rgb" if number == 3 else "_rgb"
        mapping[f"robocoin_{number:02d}"] = (
            "observation.images.cam_front_rgb" if number == 0 else "observation.images.cam_high_rgb",
            "observation.images.cam_left_wrist" + suffix,
            "observation.images.cam_right_wrist" + suffix)
    for number in (4, 10):
        mapping[f"robocoin_{number:02d}"] = (
            "observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist")
    if policy not in mapping:
        raise ValueError(f"unreviewed camera policy: {source}/{body}/{policy}")
    cameras = mapping[policy]
    if len(keys) != len(set(keys)) or set(keys) != set(cameras) - {""}:
        raise ValueError("camera keys differ from reviewed roles")
    roles = ("main", "external_left", "external_right") if policy == "robomind2_03" else (
        "main", "left_wrist", "right_wrist")
    return cameras, tuple(role if key else "pad" for role, key in zip(roles, cameras))


def resolve_profile(group, record):
    """Use upstream body rules, overriding only reviewed local column aliases."""
    if not eligible_group(group):
        raise ValueError("source is not admitted; ABC command/EEF timing is still gated")
    source, body = group["source"], group["signature"][1]
    if record["source_id"] != source or record["robot_type"] != body:
        raise ValueError("record/group embodiment mismatch")
    if source == "intern_data_a1":
        family, embodiment = "intern_a1", "intern"
        key = {"Franka": "franka", "ARX Lift-2": "lift2", "AgileX Split Aloha": "split_aloha", "Genie-1": "genie1"}[body]
        candidates = [s for s in _yaml("sources/intern_a1.yaml")["sources"]
                      if s.get("stats_group") == "a1/" + key]
    elif source == "agibot_gripper":
        family, embodiment = "agibotworld", "agibot"
        candidates = _yaml("sources/agibotworld.yaml")["sources"]
    elif source == "robomind2":
        family = embodiment = "robomind2"
        key = "franka" if body == "franka" else "ur5"
        candidates = [s for s in _yaml("sources/robomind2.yaml")["sources"]
                      if s.get("stats_group") == "robomind2/" + key]
    else:
        family = embodiment = "RoboCOIN"
        name = Path(record["source_root"]).name.lower()
        # Local coarse robot_type merges Cobot and Split-Aloha. Resolve by provenance.
        if "split_aloha" in name:
            key = "RoboCOIN/Split_aloha"
        elif "cobot_magic" in name:
            key = "RoboCOIN/Cobot_Magic"
        elif body == "galaxea_r1_lite":
            key = "RoboCOIN/R1_Lite"
        elif body == "realman_rmc_aidal":
            key = "RoboCOIN/RMC-AIDA-L"
        else:
            raise ValueError(f"unreviewed RoboCOIN root: {record['source_root']}")
        candidates = [s for s in _yaml("sources/RoboCOIN.yaml")["sources"] if s.get("stats_group") == key]
    bases = _yaml(f"embodiments/{embodiment}.yaml")["embodiments"]
    merged = [_merge_mapping(bases[s["adapter"]]["canonical_adapter"], s.get("canonical_adapter", {}))
              for s in candidates]
    if not merged or len({digest(m) for m in merged}) != 1:
        raise ValueError(f"ambiguous upstream body policy: {source}/{body}")
    adapter = copy.deepcopy(merged[0])
    if source == "robomind2":
        for slot in adapter["slots"].values():
            for component in slot.values():
                if not isinstance(component, dict):
                    continue
                for key, value in list(component.items()):
                    if isinstance(value, str) and key.endswith("key"):
                        if value.startswith("observation.states."):
                            component[key] = value.replace("observation.states.", "observation.states.puppet.", 1)
                        elif value.startswith("actions."):
                            component[key] = value.replace("actions.", "actions.master.", 1)
    cameras, roles = camera_policy(group)
    stats_group = candidates[0].get("stats_group", "agibot/a2d")
    value = dict(source=source, body=body, stats_group=stats_group, adapter=adapter,
                 cameras=list(cameras), camera_roles=list(roles), upstream_adapter=candidates[0]["adapter"])
    value["id"] = source + "_" + digest(value)[:16]
    return value


def normalization_policy():
    cfg = _yaml("dataset.yaml")
    return copy.deepcopy(dict(stats=cfg["stats"], post_normalize_transforms=cfg["post_normalize_transforms"]))


def native_ranges(record):
    """Inclusive state endpoints: a retained [a,b) span has b-a-32 starts."""
    if record["eligibility"] != "eligible" or not record.get("has_action"):
        raise ValueError("ineligible robot record")
    ranges = []
    previous = -1
    for span in record["sampling_spans"]:
        a, b, task = int(span["start"]), int(span["end_exclusive"]), int(span["task_index"])
        if a < previous or a < 0 or b <= a or b > int(record["length"]):
            raise ValueError("overlapping or invalid retained spans")
        previous = b
        if not any(int(r["start"]) <= a and b <= int(r["end_exclusive"]) for r in record["allowed_frame_ranges"]):
            raise ValueError("sampling span escapes cleaning boundary")
        if not any(int(r["task_index"]) == task and int(r["start"]) <= a and b <= int(r["end_exclusive"])
                   for r in record["task_spans"]):
            raise ValueError("sampling span crosses task boundary")
        if b - a >= 33:
            ranges.append([a, b - 32, task])
    return ranges


def range_start(ranges, rank):
    if rank < 0:
        raise IndexError(rank)
    for a, b, _ in ranges:
        if rank < b - a:
            return a + int(rank)
        rank -= b - a
    raise IndexError(rank)


def task_at(ranges, start):
    for a, b, task in ranges:
        if a <= start < b:
            return task
    raise IndexError(f"start {start} outside retained native windows")


class LocalRobotProjector(PretrainLeRobotDataset):
    """Reuse canonical projection without upstream discovery, I/O or augmentation."""
    def __init__(self, profile):
        self.name = profile["id"]
        self.canonical_adapter = copy.deepcopy(profile["adapter"])
        self.use_canonical_adapter = True
        self.canonical_dim = 80
        self.action_translation_frame = str(self.canonical_adapter.get("action_translation_frame", "source"))

    @property
    def columns(self):
        return _adapter_columns(self.canonical_adapter)

    def project_window(self, data):
        if any(len(data[key]) != 33 for key in self.columns):
            raise ValueError("projection requires exactly 33 native state rows")
        action, apad, amask = self._canonical_project_with_validity(data, list(range(32)), "action", list(range(1, 33)))
        state, spad, smask = self._canonical_project_with_validity(data, [0], "state")
        amask = (~apad)[None].expand_as(action).clone() if amask is None else amask & ~apad[None]
        smask = (~spad)[None].expand_as(state).clone() if smask is None else smask & ~spad[None]
        if not amask.any() or not smask.any() or not torch.isfinite(action[amask]).all() or not torch.isfinite(state[smask]).all():
            raise ValueError("empty or non-finite canonical supervision")
        action[~amask] = 0
        state[~smask] = 0
        return dict(action=action, action_dim_is_pad=apad, action_mask=amask,
                    action_is_pad=torch.zeros(32, dtype=torch.bool),
                    proprio=state, proprio_dim_is_pad=spad, proprio_mask=smask,
                    proprio_is_pad=torch.zeros(1, dtype=torch.bool))
