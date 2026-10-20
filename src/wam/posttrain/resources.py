"""Required resource and semantic checks used by production launchers."""
import hashlib
import json
from pathlib import Path
import os
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from wam.embodiment_registry import EmbodimentRegistry
from wam.training.checkpoint import _joint96_identity
from wam.training.config_validation import validate_training_config
from wam.training.soft_prompt import validate_soft_prompt_config
from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT
from wam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[3]

def inspect_checkpoint(path, expected_step=None):
    path = Path(str(path))
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Expected an absolute weights file, not a full-state directory')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    identity = _joint96_identity(payload.get('joint96_metadata'))
    attention = identity['action_video_attention']
    if attention != JOINT96_ACTION_VIDEO_ATTENTION:
        raise ValueError(f'Joint96 action_video_attention mismatch: checkpoint={attention!r}, model={JOINT96_ACTION_VIDEO_ATTENTION!r}')
    if (identity['action_dim'], identity['proprio_dim'], identity['channel_order'], identity['camera_order']) != (80, 80, ['wan_vae48', 'csvae48'], ['main', 'left', 'right']):
        raise ValueError('Checkpoint is not the canonical80 three-slot Joint96 interface')
    return {'path': str(path), 'step': payload.get('step'), 'bytes': path.stat().st_size, 'action_video_attention': attention, 'identity': identity, 'tensors': {key: len(payload[key]) for key in ('mot', 'proprio_encoder', 'action_proprio_encoder') if key in payload}}


def check_resources(cfg):
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from hydra.utils import instantiate
    train = cfg.data.train
    if cfg.joint96_posttrain.get('libero_frame_contract'):
        from wam.datasets.lerobot.transforms.libero_ab import LIBERO_AB_CONTRACT
        from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT
        contracts = {
            LIBERO_AB_CONTRACT['version']: (LIBERO_AB_CONTRACT, 'wam.datasets.lerobot.transforms.libero_ab.LiberoRight80ABTransform'),
            LIBERO_FLU_BI_CONTRACT['version']: (LIBERO_FLU_BI_CONTRACT, 'wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform'),
        }
        contract, transform = contracts[cfg.joint96_posttrain.libero_frame_contract]
        stats = json.loads(Path(train.pretrained_norm_stats).read_text())
        if stats.get('frame_contract') != json.loads(json.dumps(contract)):
            raise ValueError('LIBERO requires frame-aligned statistics')
        if str(train.processor.action_state_transforms[0]._target_) != transform:
            raise ValueError('LIBERO requires its matching coordinate transform')
    processor = instantiate(train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(train.pretrained_norm_stats))
    roots, prompts = ([], set())
    for raw_root in train.dataset_dirs:
        root = Path(raw_root)
        info = json.loads((root / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v2.1':
            raise ValueError('These posttraining adapters require the audited LeRobot v2.1 schema')
        for role in ('images', 'action', 'state'):
            for field in train.shape_meta[role]:
                key = field.get('lerobot_key') or (f'observation.images.{field.key}' if role == 'images' else ('action' if role == 'action' else 'observation.state') + ('' if field.key == 'default' else '.' + field.key))
                feature = info['features'][key]
                if role != 'images' and feature['shape'] != [int(field.raw_shape)]:
                    raise ValueError(f'Wrong native field shape: {root}:{key}')
        for line in (root / 'meta/tasks.jsonl').read_text().splitlines():
            if line.strip():
                prompts.add(DEFAULT_PROMPT.format(task=json.loads(line)['task']))
        roots.append({'root': str(root), 'episodes': info['total_episodes'], 'fps': info['fps']})
    missing = []
    for prompt in sorted(prompts):
        path = Path(train.text_embedding_cache_dir) / (hashlib.sha256(prompt.encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt')
        if not path.is_file():
            missing.append(str(path))
            continue
        text = torch.load(path, map_location='cpu', weights_only=True)
        if text['context'].shape != (128, 4096) or text['mask'].shape != (128,):
            raise ValueError(f'Wrong T5 shape: {path}')
        if not torch.isfinite(text['context']).all():
            raise ValueError(f'Nonfinite T5: {path}')
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)}/{len(prompts)} T5 caches. Run tools/text_cache.py with the same task/config. First: {missing[0]}')
    codec = cfg.model.joint_visual_codec
    from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
    for name in (codec.csvae_checkpoint_path, codec.dino_kwargs.checkpoint_path):
        if not Path(name).is_file():
            raise FileNotFoundError(name)
    result = {'datasets': roots, 't5_prompts': len(prompts), 'norm': str(train.pretrained_norm_stats)}
    if cfg.joint96_posttrain.get('cached_visuals', False):
        from wam.datasets.libero_joint96_cache import PairedLiberoCache
        cache = PairedLiberoCache(**OmegaConf.to_container(train.visual_cache, resolve=True))
        if any((str(Path(root).resolve()) not in cache._source_roots for root in train.dataset_dirs)):
            raise ValueError('Dataset root absent from visual cache')
        result['visual_cache'] = {'roots': [str(p) for p in cache.roots], 'samples_per_view': cache.length, 'channel_order': ['wan_vae48', 'csvae48'], 'rgb_augmentation': False}
    return result


def inspect_inputs(cfg=None):
    path = Path(cfg.init_checkpoint if cfg is not None else os.environ["WAM_PRETRAIN_CHECKPOINT"])
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    identity = _joint96_identity(payload["joint96_metadata"])
    step = payload.get("step")
    registry = EmbodimentRegistry(cfg.local_robot_softprompt.registry_path if cfg is not None else os.environ.get("EMBODIMENT_REGISTRY", str(ROOT / "configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json")))
    prompt = identity.get("soft_prompt", {})
    key = "mixtures.action.embodiment_prompt.embedding.weight"
    if (prompt.get('num_embodiments') != 28 or prompt.get('num_tokens') != 8 or prompt.get('context_dim') != 1024 or (tuple(payload['mot'][key].shape) != (28, 8192)) or (not torch.isfinite(payload['mot'][key]).all()) or (registry.content['embodiments'][0] != 'franka_panda')):
        raise ValueError("Pretrained Robot37/Panda soft prompt mismatch")
    if identity["action_video_attention"] != "current_and_noisy_future_v1":
        raise ValueError("Expected current/noisy-future attention")
    encoders = (cfg.model.joint_visual_codec.csvae_checkpoint_path,
                cfg.model.joint_visual_codec.dino_kwargs.checkpoint_path) if cfg is not None else (
                os.environ['CSVAE_CHECKPOINT'], os.environ['DINO_ENCODER_CHECKPOINT'])
    for path_value in encoders:
        if not Path(path_value).is_file():
            raise FileNotFoundError(path_value)
    return {"checkpoint": str(path), "pretrain_step": step, "csvae_step": identity.get("csvae_step"),
        "embodiment_id": 0, "embodiment_name": "franka_panda", "registry_sha256": registry.sha256,
        "prompt_shape": [28, 8, 1024]}


def validate_recipe(cfg):
    if cfg.resume is not None or cfg.local_robot_joint96.enabled:
        raise ValueError("Fresh LIBERO post-training only, not full-state resume")
    if not 0 <= cfg.min_learning_rate <= cfg.learning_rate or cfg.learning_rate <= 0:
        raise ValueError("Require positive learning_rate and 0 <= min_learning_rate <= learning_rate")
    if any(type(v) is not int or v <= 0 for v in (cfg.batch_size, cfg.gradient_accumulation_steps)):
        raise ValueError("Batch size and gradient accumulation must be positive integers")
    if not cfg.joint96_posttrain.cached_visuals:
        raise ValueError("Requires visual-cache-only")
    if cfg.joint96_posttrain.action_video_attention != "current_and_noisy_future_v1":
        raise ValueError("Wrong pretrained action/video attention")
    if cfg.joint96_posttrain.libero_frame_contract != LIBERO_FLU_BI_CONTRACT["version"]:
        raise ValueError("Wrong FLU/B=I frame contract")
    for split in ("train", "val"):
        ds = cfg.data[split]
        if ds._target_ != "wam.datasets.libero_joint96_cache.LiberoJoint96CachedDataset":
            raise ValueError("Both splits must bypass image decoding/preprocessing")
        stats = json.loads(Path(ds.pretrained_norm_stats).read_text())
        if stats.get("frame_contract") != json.loads(json.dumps(LIBERO_FLU_BI_CONTRACT)):
            raise ValueError("FLU/B=I normalization contract mismatch")
        if ds.processor.action_state_transforms[0]._target_ != (
                "wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform"):
            raise ValueError("Wrong action/proprio transform")


def validate_libero_resources(cfg):
    result = inspect_inputs(cfg)
    validate_training_config(cfg)
    validate_soft_prompt_config(cfg)
    validate_recipe(cfg)
    if Path(cfg.init_checkpoint).resolve() != Path(result["checkpoint"]).resolve():
        raise ValueError("Inspected checkpoint is not the training checkpoint")
    checkpoint = inspect_checkpoint(cfg.init_checkpoint, cfg.joint96_posttrain.expected_pretrain_step)
    result.update(resources=check_resources(cfg), checkpoint_identity=checkpoint["identity"],
                  tensor_counts=checkpoint["tensors"], passed=True)
    return result


def validate_robocasa_resources(cfg):
    from wam.datasets.robocasa_posttrain import load_package, RoboCasaTextCache
    from wam.datasets.robocasa_visual_cache import load_plan
    from wam.posttrain.robocasa_paths import require_migrated_cache
    from wam.posttrain.guard import validate_scope
    from wam.training.robocasa_posttrain import validate_cached_initialization
    validate_scope(cfg)
    validate_training_config(cfg)
    package = load_package(cfg.data.train.package_root)
    plan = load_plan(package["plan_path"])
    text = RoboCasaTextCache(cfg.data.train.package_root, package)
    for prompt in package["prompts"]:
        text.get(prompt)
    payload = torch.load(cfg.init_checkpoint, mmap=True, map_location="cpu", weights_only=True)
    step = payload.get("step")
    validate_cached_initialization(payload, plan, step)
    key = "mixtures.action.embodiment_prompt.embedding.weight"
    if bool(payload["mot"][key][18].count_nonzero()):
        raise ValueError("Parent slot18 is not reserved/empty")
    require_migrated_cache(cfg.data.train.cache_root, sorted({e["task"] for e in plan["episodes"]}))
    return {"passed": True, "pretrain_step": step, "parent_slot18_empty": True,
            "visual_identity_compatible": True, "text_prompts": len(package["prompts"]),
            "splits": {key: len(value) for key, value in package["membership"].items()}}


def validate_launch(task, overrides=()):
    """Compose exactly the launch recipe and check it before spawning workers."""
    validators = {"libero": validate_libero_resources, "robocasa": validate_robocasa_resources}
    if task not in validators:
        raise ValueError("Unsupported posttraining resource validator")
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}_posttrain", *overrides])
    return validators[task](cfg)
