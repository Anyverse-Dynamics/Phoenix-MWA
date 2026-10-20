"""Versioned raw-field contracts for the explicitly selected 36 entries."""
from __future__ import annotations

import copy

import torch

from wam.datasets.local_robot_contracts import LocalRobotProjector, digest, profile_identity
from wam.datasets import local_robot_raw_bi_contracts as bi
from wam.datasets import local_robot_raw_panda_b_contracts as panda

SCHEMA = "phoenix.local_robot36.raw_panda_b.v1"
B_POLICY = "user_adopted_panda_20261003_v1"
NEW_SOURCES = {
    "realsource": ("D31", "RS-02", bi.A_MINUS90, panda.I),
    "rwrl_g2a": ("D32", "G2A", bi.I, panda.RZ_PLUS90),
    "rwrl_r1lite": ("D33", "r1lite", bi.I, panda.RY_PLUS90),
    "rwrl_piperx": ("D34", "piperx", bi.I, None),
    "habit": ("D35", "fr3_robotiq", bi.I, panda.I),
    "generic_pick": ("D36", "franka_fr3", bi.I, panda.I),
}


def _seal(profile):
    profile = copy.deepcopy(profile)
    profile.pop("id", None)
    profile["id"] = profile_identity(profile, profile["source"] + "_raw36_")
    return profile


def from_raw_bi(profile):
    """Keep the old package untouched; update only this version's contracts."""
    out = panda.to_raw_panda_b_profile(profile)
    out.update(schema=SCHEMA, b_policy=B_POLICY,
               stats_group="raw36/" + profile["stats_group"].removeprefix("raw_bi/"),
               base_raw_profile=copy.deepcopy(profile))
    for rule in out["panda_b"].values():
        rule["status"] = "confirmed_design"
    # The selected Pro source has seven arm joints, not the Lite's six.
    if profile["field_scheme"] == "F09":
        for slot, components in out["adapter"]["slots"].items():
            base = 0 if slot == "left" else 40
            components["joint"].update(state_raw_slice=[0, 7], action_raw_slice=[0, 7],
                                        target_slice=[base, base + 7])
        out["joint_dimension_revision"] = "r1pro_seven_arm_joints_v1"
    if profile["source"] == "abc130k":
        out["timestamp_policy"] = dict(mode="reviewed_monotonic",
            review_reference="validation_runs/raw_bi_abc_clock_20261002/report.json")
    return _seal(out)


def _ee(state, action, raw_slice, base, fmt, *, offset=0, semantics="source_target"):
    out = bi._component(state, action, raw_slice, [base + 7, base + 13],
                        offset=offset, semantics=semantics)
    out.pop("target_slice")
    out.update(state_target_slice=[base + 7, base + 16],
               action_target_slice=[base + 7, base + 13],
               state_pose_input=fmt, action_pose_input=fmt)
    return out


def added_profile(source, cameras, *, variant="default"):
    if source not in NEW_SOURCES or len(cameras) != 3 or not cameras[0]:
        raise ValueError("unselected source or invalid three-slot camera contract")
    entry, body, a, b = NEW_SOURCES[source]
    adapter = dict(dim=80, action_translation_frame="source",
        state_pose_output="xyz_rot6d", action_pose_output="xyz_rotvec",
        action_pose_semantics="absolute_target", slots={},
        frame_transform=dict(world_from_source=copy.deepcopy(a),
                             source_ee_from_canonical_ee=copy.deepcopy(b or bi.I)))
    if source == "generic_pick":
        adapter = bi.field_recipe("F17")
    else:
        for arm, side in enumerate(("left", "right")):
            base = arm * 40
            if source == "realsource":
                components = dict(
                    ee=_ee("observation.state", "observation.state", [43+arm*7, 50+arm*7],
                           base, "xyz_quat_wxyz", offset=1, semantics="next_observation_target"),
                    joint=bi._component("observation.state", "action", [arm*8, arm*8+7], [base, base+7]),
                    gripper=bi._component("observation.state", "action", [arm*8+7, arm*8+8], [base+16, base+17]))
            elif source == "rwrl_g2a":
                components = dict(
                    ee=_ee("observation.state.end.position", "action.end.position", [arm*7, arm*7+7], base, "xyz_quat_xyzw"),
                    joint=bi._component("observation.state.joint.position", "action.joint.position", [arm*7, arm*7+7], [base, base+7]),
                    gripper=bi._component("observation.state.effector.position", "action.effector.position", [arm, arm+1], [base+16, base+17]))
            elif source == "rwrl_r1lite":
                components = dict(
                    ee=_ee(f"observation.state.{side}_ee_pose", f"observation.state.{side}_ee_pose", [0, 7], base,
                           "xyz_quat_xyzw", offset=1, semantics="next_observation_target"),
                    joint=bi._component(f"observation.state.{side}_arm", f"action.{side}_arm", [0, 6], [base, base+6]),
                    gripper=bi._component(f"observation.state.{side}_gripper", f"action.{side}_gripper", [0, 1], [base+16, base+17]))
            elif source == "rwrl_piperx":
                components = dict(
                    joint=bi._component("observation.state", "action", [arm*7, arm*7+6], [base, base+6]),
                    gripper=bi._component("observation.state", "action", [arm*7+6, arm*7+7], [base+16, base+17]))
            else:
                components = dict(
                    ee=_ee(f"robot{arm}.observation.state.cartesian_position", f"robot{arm}.action.target_cartesian_position",
                           [0, 6], base, "xyz_rpy"),
                    gripper=bi._component(f"robot{arm}.observation.state.gripper_position",
                        f"robot{arm}.action.target_gripper_position", [0, 1], [base+16, base+17]))
            adapter["slots"][side] = components
    decisions = {}
    for slot, components in adapter["slots"].items():
        if "ee" in components:
            components["ee"]["frame_transform"] = dict(source_ee_from_canonical_ee=copy.deepcopy(b))
            decisions[slot] = dict(matrix=copy.deepcopy(b), status="confirmed_design")
        else:
            decisions[slot] = dict(matrix=None, status="not_applicable_eef_disabled")
    profile = dict(schema=SCHEMA, b_policy=B_POLICY, source=source, body=body,
        selection_entry=entry, stats_group=f"raw36/{source}/{body}/{variant}",
        field_scheme=entry, native_stride=1, cameras=list(cameras),
        camera_roles=[role if key else "pad" for role, key in zip(("main", "left_wrist", "right_wrist"), cameras)],
        adapter=adapter, panda_b=decisions,
        timestamp_policy=dict(mode="native_uniform"))
    if source == "rwrl_piperx":
        profile["unit_contract"] = dict(joint="native_degrees_like_capture_version_unverified",
            gripper="native", physical_unit_verified=False, normalization_group_isolated=True)
    if source == "rwrl_g2a":
        profile["unit_contract"] = dict(gripper="native_per_dataset_isolated", physical_scale_verified=False)
    return _seal(profile)


class Robot36Projector(bi.RawBIProjector):
    def __init__(self, profile):
        if profile.get("schema") != SCHEMA or profile.get("b_policy") != B_POLICY or _seal(profile) != profile:
            raise ValueError("Robot36 profile schema/content identity mismatch")
        if "base_raw_profile" in profile:
            if from_raw_bi(profile["base_raw_profile"]) != profile:
                raise ValueError("Robot36 legacy field contract drift")
        elif profile != added_profile(profile["source"], profile["cameras"],
                                      variant=profile["stats_group"].rsplit("/", 1)[-1]):
            raise ValueError("Robot36 added-source field contract drift")
        LocalRobotProjector.__init__(self, profile)
        self.profile = copy.deepcopy(profile)
        self.a = torch.tensor(profile["adapter"]["frame_transform"]["world_from_source"], dtype=torch.float32)
        self._pose_b = {tuple(c["ee"]["state_target_slice"]): torch.tensor(profile["panda_b"][slot]["matrix"], dtype=torch.float32)
                        for slot, c in self.canonical_adapter["slots"].items() if "ee" in c}

    def _pose(self, data, component, indices, kind):
        xyz, rotation, xv, rv = super()._pose(data, component, indices, kind)
        rotation = rotation @ self._pose_b[tuple(component["state_target_slice"])]
        return xyz, rotation, xv, rv

    def validate_raw_schema(self, columns):
        super().validate_raw_schema(columns)
        if any("derived." in c or "eef_xforward" in c or ".x_forward" in c for c in self.columns):
            raise ValueError("derived canonical EEF must not be transformed twice")
