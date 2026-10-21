"""Joint96 LIBERO policy with per-execution-step inverse frame conversion."""

from collections import deque

import numpy as np
import torch

from wam.datasets.libero_joint96_cache import CacheRGBTransform
from wam.datasets.lerobot.transforms.libero_ab import LiberoRight80ABTransform
from wam.datasets.lerobot.utils.rotation import quaternion_to_axis_angle
from .frame_joint import infer_frame_joint_window


def observation_fields(obs):
    quat = torch.as_tensor(np.asarray(obs["robot0_eef_quat"]).copy(), dtype=torch.float32)
    ee = torch.cat((torch.as_tensor(obs["robot0_eef_pos"], dtype=torch.float32),
                    quaternion_to_axis_angle(quat[[3, 0, 1, 2]])))
    return {"ee": ee[None],
            "joint": torch.as_tensor(obs["robot0_joint_pos"], dtype=torch.float32)[None],
            "gripper": torch.as_tensor(obs["robot0_gripper_qpos"], dtype=torch.float32)[None]}


class LiberoABActionQueue:
    """Keep canonical actions pending; never cache a whole source-frame chunk."""

    def __init__(self, processor):
        self.processor = processor
        self.transform = processor.action_state_transforms[0]
        if not isinstance(self.transform, LiberoRight80ABTransform):
            raise ValueError("This queue requires the LIBERO AB processor and its statistics")
        self.pending = deque()

    def reset(self):
        self.pending.clear()

    def push(self, model_actions, execute_steps):
        if self.pending:
            raise ValueError("Drain or reset pending actions before replanning")
        if model_actions.ndim != 2 or model_actions.shape[-1] != 80:
            raise ValueError("Expected one [T,80] prediction")
        if not 1 <= int(execute_steps) <= len(model_actions):
            raise ValueError("Invalid execution horizon")
        merger = self.processor.action_state_merger
        gathered = merger._gather(model_actions.float().cpu()[None], merger.action_target_slices)
        fields = merger._split(gathered, merger.action_meta)
        canonical = self.processor.normalizer.normalizers["action"]["default"].backward(fields["default"])[0]
        if not torch.isfinite(canonical).all():
            raise ValueError("Nonfinite action prediction")
        self.pending.extend(canonical[:int(execute_steps)].clone())

    def pop(self, source_ee):
        if not self.pending:
            raise ValueError("No pending prediction")
        canonical = self.pending[0][None]
        source_ee = torch.as_tensor(source_ee, dtype=torch.float32).reshape(1, 6)
        command = self.transform.decode_commands(canonical, source_ee)[0]
        # Dataset: 0=closed, 1=open. LIBERO OSC gripper: +1=close, -1=open.
        command[:6].clamp_(-1, 1)
        command[6] = 1 - 2 * command[6].clamp(0, 1)
        if not torch.isfinite(command).all():
            raise ValueError("Nonfinite decoded command")
        self.pending.popleft()
        return command.numpy()


class LiberoJoint96ABPolicy:
    def __init__(self, model, processor, *, context, context_mask, execute_steps=8,
                 num_inference_steps=20, seed=20261001):
        self.model, self.processor = model, processor
        self.context, self.context_mask = context, context_mask
        self.execute_steps, self.num_inference_steps = int(execute_steps), int(num_inference_steps)
        self.seed = int(seed)
        self.queue = LiberoABActionQueue(processor)
        self.replans = 0

    def reset(self):
        self.queue.reset()
        self.replans = 0

    @torch.no_grad()
    def step(self, obs):
        fields = observation_fields(obs)
        if not self.queue.pending:
            raw = {"state": {k: v.clone() for k, v in fields.items()},
                   "action": {"default": torch.zeros(1, 7)}}
            encoded = self.processor.action_state_merger.forward(
                self.processor.normalizer.forward(self.processor.action_state_transform(raw)))
            images = []
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                rgb = np.ascontiguousarray(obs[key][::-1, ::-1])
                images.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)
            rgb = CacheRGBTransform()(torch.stack(images)) * 2 - 1
            slots = rgb.new_zeros(1, 3, 3, 256, 256)
            slots[0, [0, 2]] = rgb
            pred = infer_frame_joint_window(self.model, prompt=None, input_image=slots,
                num_frames=9, action_horizon=32, proprio=encoded["state"],
                context=self.context, context_mask=self.context_mask,
                camera_is_pad=torch.tensor([[False, True, False]]),
                action_dim_is_pad=encoded["action_dim_is_pad"][None],
                proprio_dim_is_pad=encoded["state_dim_is_pad"][None],
                num_inference_steps=self.num_inference_steps, seed=self.seed + self.replans)
            self.queue.push(pred["action"], self.execute_steps)
            self.replans += 1
        return self.queue.pop(fields["ee"])
