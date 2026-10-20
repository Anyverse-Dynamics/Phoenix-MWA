from typing import Any, Optional, Sequence, Union

import torch

from wam.utils.logging_config import get_logger

from .modules.experts.action_dit import ActionDiT
from .modules.conditioning.proprio_encoder import ProprioContextEncoder
from .modules.mot.mixture_of_transformers import MoT
from .backbones.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

logger = get_logger(__name__)


class WAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        understanding: Optional[dict[str, Any]] = None,
        memory: Optional[dict[str, Any]] = None,
        future_delta: Optional[dict[str, Any]] = None,
        joint96_three_view: bool = False,
    ):
        super().__init__()
        self.mot = mot
        if (
            self.video_expert is not video_expert
            or self.action_expert is not action_expert
        ):
            raise ValueError(
                "Phoenix experts must be the same modules registered in `mot.mixtures`."
            )

        self.vae = vae
        self.joint96_three_view = bool(joint96_three_view)
        self.joint_visual_encoder = None
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = None
        self.action_proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.understanding = None
        self.understanding_interval_chunks = 1
        self.understanding_cfg = dict(understanding or {})
        self._init_understanding(self.understanding_cfg)
        if self.proprio_dim is not None:
            self.proprio_encoder = ProprioContextEncoder(
                self.proprio_dim, self.text_dim
            ).to(torch_dtype)
        self.memory_cfg = dict(memory or {})
        self._memory_enabled = False
        self.memory_video_enabled = False
        self.memory_video_anchor_frames = 0
        self.memory_video_recent_frames = 0
        self._init_memory(self.memory_cfg)
        self.future_delta_cfg = dict(future_delta or {})
        self.future_delta_enabled = False
        self.future_delta_action_video_freq_ratio = 4
        self.future_delta_num_frames = 1
        self.future_delta_num_tokens = 0
        self.loss_lambda_future_delta = 0.0
        self.semantic_future_alignment_enabled = False
        self.semantic_future_student_layer = 0
        self.semantic_future_teacher_layer = 0
        self.semantic_future_max_sigma = 0.0
        self.loss_lambda_semantic_future = 0.0
        self._init_future_delta(self.future_delta_cfg)

        self.track_bridge = None
        self.to(self.device)

    @property
    def video_expert(self):
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    def set_runtime_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        # ZeRO-3 builds the model on CPU and only owns parameters that enter its
        # optimizer. Frozen inference modules therefore need an explicit move
        # after ``accelerator.prepare``. Do not move ``mot`` or the trainable
        # understanding adapter here: DeepSpeed owns those parameters.
        self.vae.to(device=self.device, dtype=self.torch_dtype)
        if self.joint_visual_encoder is not None:
            self.joint_visual_encoder.to(device=self.device)
        if hasattr(self.vae, "mean") and torch.is_tensor(self.vae.mean):
            self.vae.mean = self.vae.mean.to(device=self.device, dtype=self.torch_dtype)
        if hasattr(self.vae, "std") and torch.is_tensor(self.vae.std):
            self.vae.std = self.vae.std.to(device=self.device, dtype=self.torch_dtype)
        if hasattr(self.vae, "mean") and hasattr(self.vae, "std"):
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        if self.text_encoder is not None:
            self.text_encoder.to(device=self.device, dtype=self.torch_dtype)
        if self.understanding is not None:
            self.understanding.vlm.device = self.device
            if not self.understanding.train_vlm:
                self.understanding.vlm.model.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )

    def _init_understanding(self, cfg):
        if cfg.get("enabled", False):
            raise ValueError("Phoenix pretraining does not include understanding")

    @property
    def understanding_enabled(self) -> bool:
        return self.understanding is not None

    def _init_future_delta(self, cfg):
        if cfg.get("enabled", False):
            raise ValueError("Phoenix pretraining does not include future_delta")

    def _init_memory(self, cfg):
        video = cfg.get("video", {})
        if (not self.joint96_three_view or cfg.get("enabled", False)
                or video.get("enabled", True)
                or video.get("num_anchor_frames", 0) or video.get("num_recent_frames", 0)):
            raise ValueError("Phoenix requires current-only Joint96 conditioning")
        self.deduplicate_identical_anchor_batch = True

    @property
    def memory_enabled(self) -> bool:
        return bool(self._memory_enabled)

    def _action_token_seq_len_for_mask(self, action_seq_len: int) -> int:
        return int(action_seq_len)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = prompt_emb.masked_fill(~mask.unsqueeze(-1), 0)
        return prompt_emb.to(device=self.device), torch.ones_like(mask)

    def set_trainable_modules(self):
        self.eval()
        self.requires_grad_(False)
        self.mot.train()
        self.mot.requires_grad_(True)
        if self.track_bridge is not None:
            self.track_bridge.train()
            self.track_bridge.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        if self.action_proprio_encoder is not None:
            self.action_proprio_encoder.train()
            self.action_proprio_encoder.requires_grad_(True)
        if self.understanding is not None:
            self.understanding.set_trainable()

    def trainable_parameters(self):
        params = list(self.mot.parameters())
        if self.track_bridge is not None:
            params.extend(self.track_bridge.parameters())
        if self.proprio_encoder is not None:
            params.extend(list(self.proprio_encoder.parameters()))
        if self.action_proprio_encoder is not None:
            params.extend(list(self.action_proprio_encoder.parameters()))
        if self.understanding is not None:
            params.extend(list(self.understanding.trainable_parameters()))
        return [param for param in params if param.requires_grad]
