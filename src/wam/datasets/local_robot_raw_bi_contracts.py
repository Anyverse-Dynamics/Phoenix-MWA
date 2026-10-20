"""Opt-in raw-field Robot80 contracts from ROBOT_RAW_FIELDS_AND_A_20261002.

Public API: resolve_profile(group, record), RawBIProjector(profile), and
make_raw_bi_projector(profile_or_record, group=None). No discovery or I/O occurs.
The inherited batch API returns (values, structural_padding, exact_validity).
Missing required columns are errors. Missing/nonfinite numeric dimensions are
masked, not imputed supervision.
Callers must supply one retained native span, or supply segment_id per row to
identify cleaning boundaries. Metadata, when present, also guards transitions.
This is a projection contract, not physical/FK/timing admission evidence.
"""
from __future__ import annotations

import copy

import torch

from wam.datasets.local_robot_contracts import (
    LocalRobotProjector, camera_policy, digest, profile_identity,
    resolve_profile as _old_resolve_profile,
)
from wam.datasets.pretrain_lerobot_loader import (
    _matrix_to_rotvec, _pose_to_xyz_rotvec, _rot6d_columns_to_matrix,
    _rotvec_to_matrix,
)

SCHEMA = "phoenix.local_robot80.raw_bi.v1"
I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
A_PLUS90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
A_MINUS90 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
NEW_ARX_SOURCES = frozenset({
    "consolidated_pickup", "random_pick_and_place", "object_to_container",
    "relational_pick_and_place", "stack_blocks", "cover_blocks",
    "arrange_largest_number",
})
OLD_ARX_SOURCES = frozenset({
    "arx456_xt_motion_dedup_20260916", "arx_machine_named_20260916",
    "arx_incremental920_20260917",
})


def _decode_rot6d(raw, valid):
    first, second = raw[:, :3], raw[:, 3:6]
    norms = first.norm(dim=-1) * second.norm(dim=-1)
    good = (valid.all(-1) & torch.isfinite(norms) & (norms > 1e-12)
            & (torch.linalg.cross(first, second).norm(dim=-1) > 1e-6 * norms))
    rotation = _rot6d_columns_to_matrix(raw)
    good &= (torch.isfinite(rotation).flatten(1).all(-1)
             & ((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().flatten(1).amax(-1) < 2e-3)
             & ((torch.linalg.det(rotation) - 1).abs() < 2e-3))
    return rotation, good


def _component(state, action, raw_slice, target, *, offset=0, target_offset=None,
               action_slice=None, semantics="absolute_target_candidate"):
    return dict(state_key=state, action_key=action, state_raw_slice=list(raw_slice),
                action_raw_slice=list(action_slice or raw_slice), target_slice=list(target),
                action_source_offset=offset,
                action_target_offset=offset if target_offset is None else target_offset,
                action_target_semantics=semantics)


def field_recipe(field_scheme, *, joint_count=None):
    """Build a JSON-serializable adapter; F02 requires its embodiment joint count.

    Offsets are native raw-row offsets, not resampled indices. target_offset also
    describes already materialized next-observation labels (F15: source=0,
    target=1); they still require a legal next row at episode/crop boundaries.
    """
    scheme = str(field_scheme).upper()
    if scheme not in {f"F{i:02d}" for i in range(1, 18)}:
        raise ValueError(f"unknown raw field scheme: {scheme}")
    if scheme == "F02" and joint_count not in (6, 7):
        raise ValueError("F02 requires joint_count=6 or 7")
    single = scheme in {"F01", "F07", "F12", "F13", "F14", "F17"}
    n = (joint_count if scheme == "F02" else
         7 if scheme in {"F01", "F03", "F05", "F07", "F13", "F17"} else 6)
    a = A_PLUS90 if scheme == "F06" else A_MINUS90 if scheme == "F16" else I
    adapter = dict(dim=80, action_translation_frame="source",
                   state_pose_output="xyz_rot6d", action_pose_output="xyz_rotvec",
                   action_pose_semantics="absolute_target",
                   frame_transform=dict(world_from_source=copy.deepcopy(a),
                                        source_ee_from_canonical_ee=copy.deepcopy(I)),
                   slots={})
    for arm, slot in enumerate(("right",) if single else ("left", "right")):
        base = 40 if slot == "right" else 0
        js, ja, gs, ga = "", "", "", ""
        es, ea, fmt, afmt = "", "", "xyz_quat_wxyz", None
        eraw, araw, jraw, graw = [0, 7], None, [0, n], [0, 1]
        eo = jo = go = 0
        et = jt = gt = None
        semantics = "absolute_target_candidate"
        if scheme in {"F01", "F02"}:
            prefix = "" if single else slot + "_"
            es, ea = f"states.{prefix}tcp_to_robot_pose", f"actions.{prefix}tcp_to_robot_pose"
            js, ja = f"states.{prefix}joint.position", f"actions.{prefix}joint.position"
            gs = f"states.{prefix}gripper.position"
            ga = "actions.gripper.openness" if single else f"actions.{prefix}gripper.position"
        elif scheme == "F03":
            fmt, eo, jo = "xyz_quat_xyzw", 1, 1
            js, ja = "observation.states.joint.position", "actions.joint.position"
            gs, ga = "observation.states.effector.position", "actions.effector.position"
            jraw, graw = [arm * 7, arm * 7 + 7], [arm, arm + 1]
        elif scheme == "F04":
            es, ea, fmt = "eef_sim_pose_state", "eef_sim_pose_action", "xyz_rpy"
            eraw = [arm * 6, arm * 6 + 6]
            gs, ga, graw = "gripper_open_scale_state", "gripper_open_scale_action", [arm, arm + 1]
            eo = go = 1
        elif scheme in {"F05", "F06"}:
            es = ea = f"observation.states.puppet.end_effector_{slot}_pose"
            js, ja = f"observation.states.puppet.arm_{slot}_position", f"actions.master.arm_{slot}_position"
            gs = f"observation.states.puppet.end_effector_{slot}_position"
            ga = f"actions.master.end_effector_{slot}_position"
            fmt, eo = "xyz_quat_xyzw", 1
        elif scheme == "F07":
            es, ea, fmt, eraw = "observation.state.cartesian_position", "action.cartesian_position", "xyz_rpy", [0, 6]
            js, ja = "observation.state.joint_positions", "action.joint_position"
            gs, ga = "observation.state.gripper_position", "action.gripper_position"
        elif scheme in {"F08", "F09"}:
            es, fmt = f"observation.state.{slot}_ee_pose", "xyz_quat_xyzw"
            ea, eo = (es, 1) if scheme == "F08" else (f"action.{slot}_ee_pose", 0)
            js, ja = f"observation.state.{slot}_arm", f"action.{slot}_arm"
            gs, ga = f"observation.state.{slot}_gripper", f"action.{slot}_gripper"
        elif scheme == "F10":
            es, ea, fmt, eraw = f"observation.eef_{slot}", f"action.eef_{slot}", "matrix4x4_row_major", [0, 16]
            js = gs = "observation.state"
            ja = ga = "action_command"
            jraw, graw = [arm * 7, arm * 7 + 6], [arm * 7 + 6, arm * 7 + 7]
        elif scheme in {"F11", "F12", "F13", "F14"}:
            es, fmt = "observation.eef_pose", "xyz_rot6d_columns"
            eraw = [arm * 10, arm * 10 + 9]
            js = gs = "observation.state"
            jraw, graw = [arm * 7, arm * 7 + n], [arm * 7 + n, arm * 7 + n + 1]
            if scheme == "F14":
                ea, ja, ga = "action.eef_pose", "action", "action"
            else:
                ea, ja, ga = es, js, gs
                eo = jo = go = 1
        elif scheme == "F15":
            es, ea, afmt = "observation.state.eef", "action_abs_eef", "xyz_rpy"
            eraw, araw = [arm * 8, arm * 8 + 7], [arm * 7, arm * 7 + 6]
            js = gs = "observation.state.joint"
            ja = ga = "action_abs_joint"
            jraw, graw = [arm * 7, arm * 7 + 6], [arm * 7 + 6, arm * 7 + 7]
            et = jt = gt = 1
            semantics = "materialized_next_observation"
        elif scheme == "F16":
            es, ea, eraw = "observation.eef_pose", "action.eef_pose", [arm * 7, arm * 7 + 7]
            js = gs = "observation.state"
            ja = ga = "action"
            jraw, graw = [arm * 7, arm * 7 + 6], [arm * 7 + 6, arm * 7 + 7]
            semantics = "commanded_joint_fk_candidate"
        elif scheme == "F17":
            es, ea = "observation.eef.tcp_base_xyz_wxyz", "action.eef.tcp_base_xyz_wxyz"
            js, ja = "observation.state", "action"
            gs = ga = "source.franka.observation.states.gripper.position"
            go = 1
        ee = _component(es, ea, eraw, [base + 7, base + 13], offset=eo,
                        target_offset=et, action_slice=araw, semantics=semantics)
        ee.pop("target_slice")
        ee.update(state_target_slice=[base + 7, base + 16],
                  action_target_slice=[base + 7, base + 13],
                  state_pose_input=fmt, action_pose_input=afmt or fmt)
        if scheme == "F03":
            for kind, prefix in (("state", "observation.states"), ("action", "actions")):
                ee.pop(kind + "_key")
                ee[kind + "_position_key"] = prefix + ".end.position"
                ee[kind + "_orientation_key"] = prefix + ".end.orientation"
            ee.update(position_slice=[arm * 3, arm * 3 + 3],
                      orientation_slice=[arm * 4, arm * 4 + 4])
        components = {"ee": ee, "gripper": _component(gs, ga, graw, [base + 16, base + 17],
                      offset=go, target_offset=gt, semantics=semantics)}
        if js:
            components["joint"] = _component(js, ja, jraw, [base, base + n],
                                              offset=jo, target_offset=jt, semantics=semantics)
        for component in components.values():
            if component["action_source_offset"] == 1:
                component["action_target_semantics"] = (
                    "next_row_action_target_candidate" if scheme == "F04" else
                    "next_row_action_observation_target" if scheme == "F03" else
                    "next_observation_target")
        if scheme == "F16":
            for name in ("joint", "gripper"):
                components[name]["action_target_semantics"] = "command_target_candidate"
        elif scheme == "F17":
            ee["action_target_semantics"] = "commanded_joint_fk_target"
            components["joint"]["action_target_semantics"] = "command_or_held_target"
        adapter["slots"][slot] = components
    return adapter


def _scheme(source, body):
    if source == "intern_data_a1":
        return {"Franka": "F01", "ARX Lift-2": "F02", "AgileX Split Aloha": "F02", "Genie-1": "F02"}.get(body)
    if source == "agibot_gripper" and body == "a2d":
        return "F03"
    if source == "robocoin" and body in {"agilex_cobot_decoupled_magic", "galaxea_r1_lite", "realman_rmc_aidal"}:
        return "F04"
    if source == "robomind2":
        return {"franka": "F05", "ur": "F06"}.get(body)
    if source == "cosmos3_droid" and body == "panda":
        return "F07"
    if source == "galaxea_upperbody":
        return {"r1lite": "F08", "r1pro": "F09"}.get(body)
    if source == "abc130k" and body == "yam_bimanual":
        return "F10"
    openneo = {"openneo_aloha": ("aloha", "F11"), "openneo_arx5": ("arx5", "F11"),
               "openneo_arx5_single": ("arx5_single", "F12"), "openneo_flexiv": ("flexiv", "F13"),
               "openneo_ur": ("ur", "F14")}
    if source in openneo and body == openneo[source][0]:
        return openneo[source][1]
    if source in OLD_ARX_SOURCES and body in {"arx4", "arx5", "arx6"}:
        return "F15"
    if source in NEW_ARX_SOURCES and body == "arx_x5":
        return "F16"
    if source == "task_generalization" and body == "franka_fr3":
        return "F17"
    return None


def eligible_group(group):
    """Selection only; physical exclusions remain the package owner's gate."""
    return _scheme(group["source"], group["signature"][1]) is not None


def _new_cameras(source, keys):
    openneo = source.startswith("openneo_")
    if source == "cosmos3_droid":
        cameras = ["observation.image.exterior_image_1_left", "", "observation.image.wrist_image_left"]
    elif source == "galaxea_upperbody":
        cameras = [f"observation.images.{x}_rgb" for x in ("head", "left_wrist", "right_wrist")]
    elif openneo:
        cameras = ["observation.images.third_view", "observation.images.left_wrist_view", "observation.images.right_wrist_view"]
        if source not in {"openneo_aloha", "openneo_arx5"}:
            cameras = [cameras[0], "", cameras[1]]
    elif source == "abc130k":
        cameras = [f"observation.images.{x}" for x in ("top", "left_wrist", "right_wrist")]
    elif source in OLD_ARX_SOURCES:
        cameras = [f"observation.images.{x}_0_rgb" for x in ("base", "left_wrist", "right_wrist")]
    elif source in NEW_ARX_SOURCES:
        cameras = [f"observation.images.cam_{x}" for x in ("head", "left_wrist", "right_wrist")]
    elif source == "task_generalization":
        cameras = ["observation.images.franka_head", "", "observation.images.franka_hand"]
    else:
        # Fresh sources carry explicitly ordered roles in their new manifest.
        raise ValueError("fresh source requires explicit cameras and camera_roles")
    expected = set(cameras) - {""}
    matches = expected.issubset(keys) if source == "task_generalization" else set(keys) == expected
    if not matches or len(keys) != len(set(keys)):
        raise ValueError("camera keys differ from reviewed raw-field roles")
    return cameras, [r if k else "pad" for r, k in zip(("main", "left_wrist", "right_wrist"), cameras)]


def resolve_profile(group, record):
    """Resolve an existing-style group/episode record without modifying either.

    Fresh F16/F17 groups may provide explicit cameras/camera_roles (three slots,
    single-arm wrist in the right slot). No old gripper scaling is inherited.
    """
    source, body = group["source"], group["signature"][1]
    if (group["signature"][0] != source or record["source_id"] != source
            or record["robot_type"] != body):
        raise ValueError("record/group embodiment mismatch")
    scheme = _scheme(source, body)
    if scheme is None:
        raise ValueError(f"source/body is not selected for raw B=I: {source}/{body}")
    if "generic_pick" in str(record.get("source_root", "")):
        raise ValueError("generic_pick is not selected")
    if scheme in {"F01", "F02", "F03", "F04", "F05", "F06"}:
        old = _old_resolve_profile(group, record)
        cameras, roles = camera_policy(group)
        stats_group = "raw_bi/" + old["stats_group"]
    else:
        stats_group = f"raw_bi/{source}/{body}"
        if "cameras" in group and "camera_roles" in group:
            cameras, roles = group["cameras"], group["camera_roles"]
            if len(cameras) != 3 or len(roles) != 3 or set(cameras) - {""} != set(group["signature"][3]):
                raise ValueError("invalid explicit camera contract")
        else:
            cameras, roles = _new_cameras(source, group["signature"][3])
    value = dict(schema=SCHEMA, source=source, body=body, stats_group=stats_group,
                 field_scheme=scheme, adapter=field_recipe(scheme, joint_count=7 if body == "Genie-1" else 6),
                 cameras=list(cameras), camera_roles=list(roles),
                 raw_fields_document="ROBOT_RAW_FIELDS_AND_A_20261002.md",
                 b_policy="identity_all_slots", native_stride=1)
    if source == "task_generalization":
        value["unselected_camera_keys"] = sorted(set(group["signature"][3]) - set(cameras))
    value["id"] = profile_identity(value, source + "_raw_bi_")
    return value


def make_raw_bi_projector(profile_or_record, group=None):
    """Factory for a resolved manifest profile or a record plus its group."""
    profile = resolve_profile(group, profile_or_record) if group is not None else profile_or_record
    return RawBIProjector(profile)


class RawBIProjector(LocalRobotProjector):
    """CPU-only pure projection with exact per-row/per-dimension masks.

    Structural padding is recipe-defined. Missing data never changes that
    layout, and all invalid values are returned as zero with validity=False.
    Batch callers may project terminal rows; only targets needing a next row
    are invalid there. project_window retains the existing 33-row/32-action API.
    """

    def __init__(self, profile):
        if profile.get("schema") != SCHEMA or profile.get("b_policy") != "identity_all_slots":
            raise ValueError("RawBIProjector requires an explicitly resolved raw B=I profile")
        expected = field_recipe(profile["field_scheme"], joint_count=7 if profile.get("body") == "Genie-1" else 6)
        if profile["adapter"] != expected or profile.get("native_stride", 1) != 1:
            raise ValueError("raw B=I adapter differs from the frozen field/A/offset recipe")
        super().__init__(profile)
        self.profile = copy.deepcopy(profile)
        self.a = torch.tensor(expected["frame_transform"]["world_from_source"], dtype=torch.float32)

    @property
    def columns(self):
        return sorted({value for slot in self.canonical_adapter["slots"].values()
                       for component in slot.values() for key, value in component.items()
                       if key.endswith("_key")})

    def validate_raw_schema(self, columns):
        """Fail closed on absent required keys; numeric invalidity is separate."""
        missing = set(self.columns) - set(columns)
        if missing:
            raise KeyError("missing required raw fields: " + repr(sorted(missing)))

    def _length(self, data):
        self.validate_raw_schema(data)
        lengths = {len(data[key]) for key in self.columns if key in data}
        if not lengths or len(lengths) != 1:
            raise ValueError("raw columns require a nonempty, aligned row axis")
        length = lengths.pop()
        if length == 0:
            raise ValueError("empty raw batch")
        return length

    @staticmethod
    def _feature(data, key, indices, raw_slice):
        begin, end = raw_slice
        out = torch.zeros((len(indices), end - begin), dtype=torch.float32)
        valid = torch.zeros_like(out, dtype=torch.bool)
        if key not in data:
            return out, valid
        value = torch.as_tensor(data[key])
        if value.device.type != "cpu":
            raise ValueError("raw B=I projection is CPU-only")
        value = value.to(torch.float32).reshape(len(value), -1)
        width = max(0, min(end, value.shape[1]) - begin)
        if width:
            in_bounds = (indices >= 0) & (indices < len(value))
            selected = value[indices.clamp(0, len(value) - 1), begin:begin + width]
            mask = torch.isfinite(selected) & in_bounds[:, None]
            out[:, :width] = torch.where(mask, selected, 0)
            valid[:, :width] = mask
        return out, valid

    def _pose(self, data, component, indices, kind):
        fmt = component[kind + "_pose_input"]
        if kind + "_key" in component:
            raw, valid = self._feature(data, component[kind + "_key"], indices, component[kind + "_raw_slice"])
        else:
            xyz, xv = self._feature(data, component[kind + "_position_key"], indices, component["position_slice"])
            quat, qv = self._feature(data, component[kind + "_orientation_key"], indices, component["orientation_slice"])
            raw, valid = torch.cat((xyz, quat), -1), torch.cat((xv, qv), -1)
        if fmt == "matrix4x4_row_major":
            matrix = raw.reshape(-1, 4, 4)
            mv = valid.reshape(-1, 4, 4)
            xyz, xv, rotation = matrix[:, :3, 3], mv[:, :3, 3], matrix[:, :3, :3]
            homogeneous = mv[:, 3].all(-1) & ((matrix[:, 3] - torch.tensor([0., 0., 0., 1.])).abs().amax(-1) < 1e-4)
            rv = (mv[:, :3, :3].flatten(1).all(-1) & homogeneous
                  & ((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().flatten(1).amax(-1) < 2e-3)
                  & ((torch.linalg.det(rotation) - 1).abs() < 2e-3))
            xv = xv & homogeneous[:, None]
        elif fmt == "xyz_rot6d_columns":
            xyz, xv = raw[:, :3], valid[:, :3]
            rotation, rv = _decode_rot6d(raw[:, 3:], valid[:, 3:])
        else:
            xyz, xv = raw[:, :3], valid[:, :3]
            rv = valid[:, 3:].all(-1)
            if "quat" in fmt:
                norms = raw[:, 3:].norm(dim=-1)
                rv = rv & torch.isfinite(norms) & (norms > 1e-8)
            rotation = _rotvec_to_matrix(_pose_to_xyz_rotvec(raw, fmt)[:, 3:])
        # A is a signed permutation: preserve independent XYZ validity exactly.
        xyz = xyz @ self.a.T
        xv = xv[:, self.a.abs().argmax(-1)]
        rotation = self.a @ torch.where(rv[:, None, None], rotation, torch.eye(3))
        return xyz, rotation, xv, rv

    def rotation_diagnostics(self, data):
        """Report OpenNEO raw scale/skew before Gram-Schmidt, plus output SO(3).

        This is a read-only numeric audit of the supplied rows, not FK/physical
        frame verification. Nonfinite and degenerate rows are counted explicitly.
        """
        indices = torch.arange(self._length(data))
        result = {}
        for name, slot in self.canonical_adapter["slots"].items():
            component = slot["ee"]
            for kind in ("state", "action"):
                if component[kind + "_pose_input"] != "xyz_rot6d_columns":
                    continue
                key, section = component[kind + "_key"], component[kind + "_raw_slice"]
                raw, valid = self._feature(data, key, indices, section)
                vectors = raw[:, 3:]
                finite = valid[:, 3:].all(-1)
                rotation, good = _decode_rot6d(vectors, valid[:, 3:])
                first, second = vectors[:, :3], vectors[:, 3:]
                n0, n1 = first.norm(dim=-1), second.norm(dim=-1)
                cosine = (first * second).sum(-1) / (n0 * n1).clamp_min(1e-12)
                def maximum(value, mask):
                    chosen = value[mask]
                    return float(chosen.max()) if len(chosen) else None
                result[name + "." + kind] = dict(
                    key=key, raw_slice=section, rows=len(indices), finite_rows=int(finite.sum()),
                    valid_rotation_rows=int(good.sum()), gram_schmidt_applied=True,
                    max_raw_column_norm_error=maximum(torch.maximum((n0 - 1).abs(), (n1 - 1).abs()), finite),
                    max_raw_abs_column_cosine=maximum(cosine.abs(), finite),
                    max_output_orthogonality_error=maximum((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().flatten(1).amax(-1), good),
                    max_output_det_error=maximum((torch.linalg.det(rotation) - 1).abs(), good))
        return result

    @staticmethod
    def _transition(data, indices, length):
        future = indices + 1
        good = future < length
        safe = future.clamp(max=length - 1)
        for key in ("episode_index", "task_index", "segment_id"):
            if key in data:
                values = torch.as_tensor(data[key]).reshape(length, -1)
                good &= (values[indices] == values[safe]).all(-1)
        for key in ("frame_index", "index"):
            if key in data:
                values = torch.as_tensor(data[key]).reshape(length, -1)
                good &= (values[safe] == values[indices] + 1).all(-1)
        if "timestamp" in data:
            times = torch.as_tensor(data["timestamp"]).reshape(length, -1)
            good &= (torch.isfinite(times[indices]) & torch.isfinite(times[safe]) & (times[safe] > times[indices])).all(-1)
        return good

    def _canonical_project_with_validity(self, data, indices, kind, next_indices=None, **kwargs):
        if kwargs:
            raise TypeError("raw B=I projection has no retiming or transform overrides")
        if kind not in {"state", "action"}:
            raise ValueError(kind)
        length = self._length(data)
        index = torch.as_tensor(indices, dtype=torch.long)
        if index.ndim != 1 or ((index < 0) | (index >= length)).any():
            raise ValueError("current indices outside raw batch")
        if next_indices is not None and not torch.equal(torch.as_tensor(next_indices), index + 1):
            raise ValueError("raw B=I requires native stride=1 next_indices")
        out = torch.zeros((len(index), 80), dtype=torch.float32)
        mask = torch.zeros_like(out, dtype=torch.bool)
        pad = torch.ones(80, dtype=torch.bool)
        transition = self._transition(data, index, length) if kind == "action" else None
        for slot in self.canonical_adapter["slots"].values():
            for name, component in slot.items():
                begin, end = component.get(kind + "_target_slice", component.get("target_slice"))
                pad[begin:end] = False
                offset = component["action_source_offset"] if kind == "action" else 0
                selected = index + offset
                if name == "ee":
                    p, r, pv, rv = self._pose(data, component, selected, kind)
                    if kind == "state":
                        values = torch.cat((p, r[:, :, :2].transpose(-1, -2).reshape(-1, 6)), -1)
                        valid = torch.cat((pv, rv[:, None].expand(-1, 6)), -1)
                    else:
                        cp, cr, cpv, crv = self._pose(data, component, index, "state")
                        values = torch.cat((p - cp, _matrix_to_rotvec(cr.transpose(-1, -2) @ r)), -1)
                        valid = torch.cat((pv & cpv, (rv & crv)[:, None].expand(-1, 3)), -1)
                else:
                    values, valid = self._feature(data, component[kind + "_key"], selected, component[kind + "_raw_slice"])
                if kind == "action" and component["action_target_offset"]:
                    valid &= transition[:, None]
                valid &= torch.isfinite(values)
                out[:, begin:end] = torch.where(valid, values, 0)
                mask[:, begin:end] = valid
        return out, pad, mask

    def project_rows(self, data, indices=None, kind="state", next_indices=None):
        """Return (N x 80 float32, 80 bool padding, N x 80 bool validity)."""
        if indices is None:
            indices = list(range(self._length(data)))
        return self._canonical_project_with_validity(data, indices, kind, next_indices)

    def project_window(self, data):
        if self._length(data) != 33:
            raise ValueError("projection requires exactly 33 native state rows")
        if not self._transition(data, torch.arange(32), 33).all():
            raise ValueError("projection window crosses episode/task/cleaning/native boundary")
        # Keep the parent schema while allowing per-component numeric corruption.
        action, apad, amask = self.project_rows(data, list(range(32)), "action")
        state, spad, smask = self.project_rows(data, [0], "state")
        if not amask.any() or not smask.any():
            raise ValueError("empty canonical supervision")
        return dict(action=action, action_dim_is_pad=apad, action_mask=amask,
                    action_is_pad=~amask.any(-1), proprio=state,
                    proprio_dim_is_pad=spad, proprio_mask=smask,
                    proprio_is_pad=~smask.any(-1))


# Names used by the independently implemented package and statistics entrypoints.
LocalRobotRawBIProjector = RawBIProjector
RawBILocalRobotProjector = RawBIProjector
