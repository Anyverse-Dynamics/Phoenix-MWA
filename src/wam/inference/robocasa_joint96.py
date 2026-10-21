"""Joint video/action denoising with GR1 absolute joint-control output."""
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wam.datasets.robocasa_delta import CONTRACT, masks, project, decode, observation_state, prompt_for, RoboCasaNormalizer
from wam.datasets.robocasa_posttrain import action_space_module, package_space
from wam.datasets.robocasa_visual_cache import read_json, require
from wam.training.checkpoint import load_joint96_initialization
from wam.utils.config_resolvers import register_default_resolvers
from .frame_joint import infer_frame_joint_window
from .robocasa_text import RoboCasaInferenceText
from .robocasa_normalization import resolve_normalization_stats
from .robocasa_resources import DEFAULT_PACKAGE, ROOT, load_inference_package


def instantiate_checkpoint_model(model_config, device):
    """Construct evaluation modules before the complete strict policy restore."""
    import wam.model.factory as factory
    original_loader = factory.load_wan22_ti2v_5b_components

    def checkpoint_only_components(**kwargs):
        kwargs['skip_dit_load_from_pretrain'] = True
        return original_loader(**kwargs)

    # Keep the training architecture checks and saved config intact. The full
    # policy below replaces the temporary DiT initialization before inference.
    factory.load_wan22_ti2v_5b_components = checkpoint_only_components
    try:
        return instantiate(model_config, model_dtype=torch.bfloat16, device=device)
    finally:
        factory.load_wan22_ti2v_5b_components = original_loader


def validate_checkpoint(payload, package, root=None, *, expected_pretrain_step=None):
    saved = payload.get("joint96_metadata", {}).get("soft_prompt", {}).get("training_contract", {})
    contract = package.get("contract", CONTRACT)
    expected = {"domain": "robocasa", "frame_contract": contract, "embodiment_id": contract["embodiment_id"],
                "visual_attention_backend": "math_sdpa"}
    for key, value in expected.items():
        require(saved.get(key) == value, f"RoboCasa checkpoint provenance mismatch: {key}")


class RoboCasaJoint96Policy:
    def __init__(self, model, normalizer, text, *, execute_steps=12, num_inference_steps=10, seed=7,
                 action_space="delta80"):
        require(1 <= execute_steps <= 32 and num_inference_steps > 0, "Invalid inference horizon/steps")
        self.model, self.normalizer, self.text = model, normalizer, text
        self.execute_steps, self.num_inference_steps, self.seed = execute_steps, num_inference_steps, seed
        self.space = action_space_module(action_space)
        self.action_space = action_space
        self.action_dim = len(self.space.masks()[0])
        self.embodiment_id = self.space.CONTRACT["embodiment_id"]
        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint, config, package_root=DEFAULT_PACKAGE, *, device="cuda:0", allow_smoke=False, **kwargs):
        register_default_resolvers()
        cfg = OmegaConf.load(config)
        from wam.posttrain.guard import install_import_guard, validate_scope
        install_import_guard()
        validate_scope(cfg)
        from wam.training.robocasa_posttrain import validate_robocasa_config
        validate_robocasa_config(cfg)
        package = load_inference_package(package_root, allow_smoke=allow_smoke)
        space = package_space(package)
        require(cfg.joint96_posttrain.robocasa.frame_contract == space.CONTRACT["version"],
                "Inference config/package action space mismatch")
        stats_path = resolve_normalization_stats(cfg)
        # Validate the source statistics before constructing the expensive model.
        normalizer = space.RoboCasaNormalizer(read_json(stats_path))
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        saved_contract = payload.get('joint96_metadata', {}).get('soft_prompt', {}).get('training_contract', {})
        if saved_contract.get('smoke_selection_sha256') and not allow_smoke:
            raise ValueError('Smoke checkpoint requires explicit allow_smoke; not a formal result')
        validate_checkpoint(payload, package, package_root,
                            expected_pretrain_step=None)
        from wam.utils.checkpoint_steps import checkpoint_steps
        progress = checkpoint_steps(payload)
        step = progress['step']
        del payload
        model = instantiate_checkpoint_model(cfg.model, device)
        load_joint96_initialization(model, str(Path(checkpoint).resolve()), expected_step=step)
        model.set_runtime_device(device)
        model.eval()
        cache_dir = OmegaConf.select(cfg, 'inference.text_cache_dir', default='.cache/robocasa_t5')
        if cache_dir is not None:
            cache_dir = Path(cache_dir).expanduser()
            if not cache_dir.is_absolute():
                cache_dir = ROOT / cache_dir
        text = RoboCasaInferenceText(None, cfg.model, device, cache_dir=cache_dir)
        policy = cls(model, normalizer,
                     text, action_space=package.get("action_space", "delta80"), **kwargs)
        policy.checkpoint_progress = progress
        policy.normalization_stats_path = str(stats_path)
        print('ROBOCASA_NORMALIZATION_STATS', str(stats_path), flush=True)
        print('ROBOCASA_CHECKPOINT_PROGRESS', progress, flush=True)
        return policy

    def reset(self, task_description=None):
        self.replans = 0
        self.task_description = task_description

    @torch.inference_mode()
    def predict(self, rgb, raw_state, instruction):
        rgb = np.asarray(rgb)
        require(rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8, "Use unflipped native 256x256 RGB uint8")
        state, _ = self.space.project(raw_state)
        require(state.shape == (1, self.action_dim), "One environment per policy")
        slots = torch.full((1, 3, 3, 256, 256), -1.)
        slots[0, 0] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255*2-1
        context, context_mask = self.text.get(prompt_for(instruction))
        state_valid, action_valid = self.space.masks()
        result = infer_frame_joint_window(self.model, prompt=None, input_image=slots,
            num_frames=9, action_horizon=32, proprio=self.normalizer.forward(state, "state"),
            context=context[None], context_mask=context_mask[None],
            camera_is_pad=torch.tensor([[False, True, True]]),
            action_dim_is_pad=~action_valid[None], proprio_dim_is_pad=~state_valid[None],
            embodiment_ids=torch.tensor([self.embodiment_id], dtype=torch.long, device=self.model.device),
            num_inference_steps=self.num_inference_steps, seed=self.seed,
            video_layout="latent_horizontal", video_view_names="main|left|right", decode_video=False)
        self.replans += 1
        canonical = self.normalizer.backward(result["action"].float().cpu(), "action")
        require(canonical.shape == (32, self.action_dim) and bool(torch.isfinite(canonical).all()), "Invalid action prediction")
        # Auxiliary EEF targets stay available for diagnostics. The GR1 controller consumes
        # the independent absolute joint heads, not an invented IK conversion.
        self.last_canonical = canonical
        return {key: value[:self.execute_steps].numpy()[None] for key, value in self.space.decode(canonical).items()}

    def step(self, observations):
        key = "video.ego_view" if "video.ego_view" in observations else "video.ego_view_bg_crop_pad_res256_freq20"
        rgb = np.asarray(observations[key])
        require(rgb.shape == (1, 1, 256, 256, 3), "Expected one environment and one current main camera")
        instruction = np.asarray(observations["annotation.human.coarse_action"]).reshape(-1)
        require(instruction.size == 1, "Expected one instruction")
        return {"actions": self.predict(rgb[0, 0], observation_state(observations), str(instruction[0]))}
