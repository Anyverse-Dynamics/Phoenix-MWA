"""Independent, opt-in Panda-reference B on the frozen raw B=I field recipes.

Public API: to_raw_panda_b_profile(raw_bi_profile), resolve_profile(group, record),
RawPandaBProjector(profile), make_raw_panda_b_projector(profile_or_record, group).
The source recipe is retained verbatim; only EEF orientation changes:
R = A R_raw B, delta_R = B.T delta_R_BI B. Origins, fields, offsets and masks
are inherited. Confirmed design choices are not physical/timing validation.
Unconfirmed selected sources have explicit pending status and identity B.
No discovery, statistics, publication, or I/O occurs in this module.
"""
from __future__ import annotations

import copy

import torch

from wam.datasets.local_robot_contracts import digest, profile_identity
from wam.datasets import local_robot_raw_bi_contracts as raw_bi

SCHEMA = "phoenix.local_robot80.raw_panda_b.v1"
B_POLICY = "panda_confirmed_pending_identity"
I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
RZ_PLUS90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
RZ_MINUS90 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
RY_PLUS90 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
RY_180 = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
_ROTATIONS = {"I": I, "Rz_plus90": RZ_PLUS90, "Rz_minus90": RZ_MINUS90,
              "Ry_plus90": RY_PLUS90, "Ry_180": RY_180}

# Exact raw-BI group identities keep same-body sources and pending variants
# separate. These choices implement the user's table, not name-based inference.
_DECISIONS = {
    "a1/franka": ("intern_data_a1", "Franka", "confirmed", "I", "I"),
    "a1/split_aloha": ("intern_data_a1", "AgileX Split Aloha", "confirmed", "Rz_plus90", "Rz_plus90"),
    "a1/genie1": ("intern_data_a1", "Genie-1", "confirmed", "Rz_plus90", "Rz_minus90"),
    "a1/lift2": ("intern_data_a1", "ARX Lift-2", "confirmed", "Ry_plus90", "Ry_plus90"),
    "agibot/a2d": ("agibot_gripper", "a2d", "confirmed", "Rz_plus90", "Rz_minus90"),
    "RoboCOIN/Cobot_Magic": ("robocoin", "agilex_cobot_decoupled_magic", "confirmed", "Rz_plus90", "Rz_plus90"),
    "RoboCOIN/Split_aloha": ("robocoin", "agilex_cobot_decoupled_magic", "confirmed", "Rz_plus90", "Rz_plus90"),
    "RoboCOIN/R1_Lite": ("robocoin", "galaxea_r1_lite", "confirmed", "Ry_plus90", "Ry_plus90"),
    "RoboCOIN/RMC-AIDA-L": ("robocoin", "realman_rmc_aidal", "confirmed", "I", "I"),
    "galaxea_upperbody/r1lite": ("galaxea_upperbody", "r1lite", "confirmed", "Ry_plus90", "Ry_plus90"),
    # Adopted for the selected 2025 gripper_link subset; mesh-version gap remains.
    "galaxea_upperbody/r1pro": ("galaxea_upperbody", "r1pro", "confirmed", "Ry_180", "Ry_180"),
    "abc130k/yam_bimanual": ("abc130k", "yam_bimanual", "confirmed", "I", "I"),
    "cosmos3_droid/panda": ("cosmos3_droid", "panda", "confirmed", "I", "I"),
    "robomind2/franka": ("robomind2", "franka", "confirmed", "I", "I"),
    # User-adopted identity; TCP mounting/physical axes remain unverified.
    "robomind2/ur5": ("robomind2", "ur", "confirmed", "I", "I"),
    "task_generalization/franka_fr3": ("task_generalization", "franka_fr3", "confirmed", "I", "I"),
    "openneo_arx5/arx5": ("openneo_arx5", "arx5", "confirmed", "Ry_plus90", "Ry_plus90"),
    "openneo_aloha/aloha": ("openneo_aloha", "aloha", "pending", "I", "I"),
    # Adopted after independent FK audit; collection tool-version gap remains.
    "openneo_arx5_single/arx5_single": ("openneo_arx5_single", "arx5_single", "confirmed", "Ry_plus90", "Ry_plus90"),
    # User-adopted identity; tactile mounting and action consistency remain open.
    "openneo_flexiv/flexiv": ("openneo_flexiv", "flexiv", "confirmed", "I", "I"),
    # User-adopted identity; collection TCP and gripper mounting remain unverified.
    "openneo_ur/ur": ("openneo_ur", "ur", "confirmed", "I", "I"),
}
for _source in sorted(raw_bi.OLD_ARX_SOURCES):
    for _body in ("arx4", "arx5", "arx6"):
        _DECISIONS[f"{_source}/{_body}"] = (_source, _body, "confirmed", "Ry_plus90", "Ry_plus90")
for _source in sorted(raw_bi.NEW_ARX_SOURCES):
    _DECISIONS[f"{_source}/arx_x5"] = (_source, "arx_x5", "confirmed", "Ry_plus90", "Ry_plus90")

_ADDED_KEYS = ("base_profile_id", "base_stats_group", "source_field_recipe", "panda_b", "b_decision_document")


def to_raw_panda_b_profile(raw_bi_profile):
    """Pure, deterministic conversion of one admitted raw-BI profile.

    adapter is the effective A/B contract. source_field_recipe is its frozen
    B=I predecessor, with every raw field, target slice and row offset retained.
    Pending B=I is explicitly different metadata from confirmed identity B.
    """
    raw_bi.RawBIProjector(raw_bi_profile)
    if any(key in raw_bi_profile for key in _ADDED_KEYS):
        raise ValueError("raw-BI input already contains Panda B conversion metadata")
    group = raw_bi_profile["stats_group"]
    if not isinstance(group, str) or not group.startswith("raw_bi/"):
        raise ValueError("Panda B conversion requires the raw_bi stats namespace")
    suffix = group.removeprefix("raw_bi/")
    decision = _DECISIONS.get(suffix)
    if decision is None:
        raise ValueError(f"source/group is outside the selected Panda B scope: {group}")
    source, body, status, left, right = decision
    if ((raw_bi_profile["source"], raw_bi_profile["body"]) != (source, body)
            or raw_bi_profile["field_scheme"] != raw_bi._scheme(source, body)):
        raise ValueError("raw-BI source/body/field scheme disagrees with selected stats group")
    profile = copy.deepcopy(raw_bi_profile)
    profile.update(schema=SCHEMA, b_policy=B_POLICY,
        base_profile_id=raw_bi_profile["id"], base_stats_group=group,
        stats_group="raw_panda_b/" + suffix,
        source_field_recipe=copy.deepcopy(raw_bi_profile["adapter"]),
        b_decision_document="EEF_B_PANDA_REFERENCE_20261002.md", panda_b={})
    for slot, component in profile["adapter"]["slots"].items():
        rotation = {"left": left, "right": right}[slot]
        matrix = _ROTATIONS[rotation]
        profile["panda_b"][slot] = dict(rotation=rotation, matrix=copy.deepcopy(matrix), status=status)
        component["ee"]["frame_transform"] = dict(source_ee_from_canonical_ee=copy.deepcopy(matrix))
    profile.pop("id")
    profile["id"] = profile_identity(profile, source + "_raw_panda_b_")
    return profile


def resolve_profile(group, record):
    """Resolve a selected manifest record with the unchanged raw-BI admission recipe."""
    return to_raw_panda_b_profile(raw_bi.resolve_profile(group, record))


def make_raw_panda_b_projector(profile_or_record, group=None):
    """Accept a resolved Panda/B=I profile, or a record and its original group."""
    profile = resolve_profile(group, profile_or_record) if group is not None else profile_or_record
    if profile.get("schema") == raw_bi.SCHEMA:
        profile = to_raw_panda_b_profile(profile)
    return RawPandaBProjector(profile)


class RawPandaBProjector(raw_bi.RawBIProjector):
    """The raw-BI projector with only a same-arm EEF right-multiplication by B."""

    def __init__(self, profile):
        if profile.get("schema") != SCHEMA or profile.get("b_policy") != B_POLICY:
            raise ValueError("RawPandaBProjector requires an explicitly resolved Panda B profile")
        base = copy.deepcopy(profile)
        for key in _ADDED_KEYS:
            base.pop(key)
        base.update(schema=raw_bi.SCHEMA, b_policy="identity_all_slots",
            id=profile["base_profile_id"], stats_group=profile["base_stats_group"],
            adapter=copy.deepcopy(profile["source_field_recipe"]))
        if to_raw_panda_b_profile(base) != profile:
            raise ValueError("Panda B profile differs from the selected field/A/B/status contract")
        super().__init__(base)
        self.profile = copy.deepcopy(profile)
        self.name = profile["id"]
        self.canonical_adapter = copy.deepcopy(profile["adapter"])
        self.b_by_slot = {slot: torch.tensor(rule["matrix"], dtype=torch.float32)
                          for slot, rule in profile["panda_b"].items()}
        self._pose_b = {tuple(self.canonical_adapter["slots"][slot]["ee"]["state_target_slice"]):
                        (rule["rotation"] != "I", self.b_by_slot[slot])
                        for slot, rule in profile["panda_b"].items()}

    def _pose(self, data, component, indices, kind):
        xyz, rotation, xyz_valid, rotation_valid = super()._pose(data, component, indices, kind)
        changed, b = self._pose_b[tuple(component["state_target_slice"])]
        if changed:
            rotation = rotation @ b
        return xyz, rotation, xyz_valid, rotation_valid
