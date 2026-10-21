"""Load only frozen cache encoders; no policy or action expert is constructed."""
from pathlib import Path

from omegaconf import OmegaConf

from .common import digest, sha256


def model_settings(path):
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config.model if "model" in config else config, resolve=True)


def tree_digest(root):
    root = Path(root)
    return digest({p.relative_to(root).as_posix(): sha256(p)
                   for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py", ".json")})


def visual_resources(model_definition):
    from wam.model.backbones.wan22.helpers.loader import _resolve_configs
    config = model_settings(model_definition)
    _, _, vae, _ = _resolve_configs(config["model_id"], config["tokenizer_model_id"],
                                   config.get("redirect_common_files", True))
    vae.download_if_necessary()
    if not isinstance(vae.path, str):
        raise ValueError("Expected one frozen Wan VAE file")
    codec = config["joint_visual_codec"]
    if codec["source_root"] != "bundled://dino-stvae-v1":
        raise ValueError("Public cache generation uses the bundled Phoenix encoder")
    return config, vae.path


def visual_identity(config, vae_path):
    import wam
    root = Path(wam.__file__).parent
    codec = config["joint_visual_codec"]
    semantic = dict(codec)
    semantic.pop("source_sha256", None)
    semantic.pop("expected_stvae_step", None)
    semantic["stvae_checkpoint_path"] = sha256(codec["stvae_checkpoint_path"])
    semantic["dino_kwargs"] = dict(codec["dino_kwargs"])
    semantic["dino_kwargs"].pop("dinov3_source_path", None)
    semantic["dino_kwargs"]["checkpoint_path"] = sha256(codec["dino_kwargs"]["checkpoint_path"])
    if semantic["dino_kwargs"].get("projector_state_path") or semantic["dino_kwargs"].get("feature_norm_state_path"):
        raise ValueError("Phoenix cache requires raw DINO features without auxiliary projection/statistics")
    return dict(vae_sha256=sha256(vae_path), codec=semantic,
                current_conditioning=codec.get("current_conditioning", "legacy_clip_prefix"),
                encoder_source_sha256=tree_digest(root / "model/modules/codecs"),
                dino_source_sha256=tree_digest(root / "third_party/dino_runtime"),
                student_source_sha256=tree_digest(root / "encoder"),
                loader_source_sha256=tree_digest(root / "model/backbones/wan22/helpers"),
                wan_vae_source_sha256=sha256(root / "model/backbones/wan22/wan_video_vae.py"))


def load_visual(config, vae_path, device):
    import torch
    from wam.model.backbones.wan22.helpers.loader import _load_registered_model
    from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder
    from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
    from wam.utils.torch24_compat import install_safe_serialization_compat
    install_safe_serialization_compat()
    vae = _load_registered_model(vae_path, "wan_video_vae", torch_dtype=torch.bfloat16, device=device)
    encoder = JointVisualEncoder.from_config(bundled_config(config["joint_visual_codec"]), vae=vae, device=device)
    encoder.eval().requires_grad_(False)
    # JointVisualEncoder keeps a weak VAE reference; the caller owns both objects.
    return vae, encoder


def text_identity(config):
    from . import text_cache
    from wam.model.backbones.wan22 import wan_video_text_encoder
    tokenizer = Path(config["tokenizer"])
    if not tokenizer.is_dir() or not any(tokenizer.iterdir()):
        raise ValueError("Provide a local Wan UMT5 tokenizer directory")
    return dict(encoder_sha256=sha256(config["encoder"]),
                tokenizer_sha256=digest({p.relative_to(tokenizer).as_posix(): sha256(p)
                                         for p in sorted(tokenizer.rglob("*")) if p.is_file()}),
                encoder_source_sha256=sha256(wan_video_text_encoder.__file__),
                cache_source_sha256=sha256(text_cache.__file__), max_length=128, dtype="bfloat16",
                padding="zero_embeddings_all_tokens_visible")


def load_text(config):
    import torch
    from wam.model.backbones.wan22.helpers.loader import _load_registered_model
    from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    encoder = _load_registered_model(config["encoder"], "wan_video_text_encoder",
                                     torch_dtype=torch.bfloat16, device=config["device"])
    encoder.eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=config["tokenizer"], seq_len=128, clean="whitespace")

    @torch.no_grad()
    def encode(prompt):
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(config["device"]), mask.to(config["device"], dtype=torch.bool)
        context = encoder(ids, mask).masked_fill(~mask.unsqueeze(-1), 0)
        # Match WorldActionModel.encode_prompt: zero padding remains visible to attention.
        return context[0].to("cpu", dtype=torch.bfloat16), torch.ones(128, dtype=torch.bool)
    return encode
