"""Read-only schema/cache/checkpoint checks; this is not a GPU smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from wam.training.checkpoint import _joint96_identity
from wam.training.config_validation import validate_training_config
from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
from wam.utils.config_resolvers import register_default_resolvers

def compose_recipe(task, overrides=()):
    tasks = {'libero': 'libero_joint96', 'robodojo': 'robodojo_joint96', 'libero_cached': 'libero_joint96_cached', 'libero_cached_ab': 'libero_joint96_cached_ab'}
    if task not in tasks:
        raise ValueError(task)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[f'task={tasks[task]}', *overrides])
    validate_training_config(cfg)
    return cfg

def inspect_checkpoint(path, expected_step):
    path = Path(str(path))
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Expected an absolute weights file, not a full-state directory')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if type(payload.get('step')) is not int or payload['step'] != int(expected_step):
        raise ValueError(f"Checkpoint step {payload.get('step')} != {expected_step}")
    identity = _joint96_identity(payload.get('joint96_metadata'))
    attention = identity['action_video_attention']
    if attention != JOINT96_ACTION_VIDEO_ATTENTION:
        raise ValueError(f'Joint96 action_video_attention mismatch: checkpoint={attention!r}, model={JOINT96_ACTION_VIDEO_ATTENTION!r}')
    if (identity['action_dim'], identity['proprio_dim'], identity['channel_order'], identity['camera_order']) != (80, 80, ['wan_vae48', 'stvae48'], ['main', 'left', 'right']):
        raise ValueError('Checkpoint is not the canonical80 three-slot Joint96 interface')
    return {'path': str(path), 'step': payload['step'], 'bytes': path.stat().st_size, 'action_video_attention': attention, 'identity': identity, 'tensors': {key: len(payload[key]) for key in ('mot', 'proprio_encoder', 'action_proprio_encoder') if key in payload}}

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
    for name in (codec.stvae_checkpoint_path, codec.dino_kwargs.checkpoint_path):
        if not Path(name).is_file():
            raise FileNotFoundError(name)
    result = {'datasets': roots, 't5_prompts': len(prompts), 'norm': str(train.pretrained_norm_stats)}
    if cfg.joint96_posttrain.get('cached_visuals', False):
        from wam.datasets.libero_joint96_cache import PairedLiberoCache
        cache = PairedLiberoCache(**OmegaConf.to_container(train.visual_cache, resolve=True))
        if any((str(Path(root).resolve()) not in cache._source_roots for root in train.dataset_dirs)):
            raise ValueError('Dataset root absent from visual cache')
        result['visual_cache'] = {'roots': [str(p) for p in cache.roots], 'samples_per_view': cache.length, 'channel_order': ['wan_vae48', 'stvae48'], 'rgb_augmentation': False}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('libero', 'robodojo', 'libero_cached', 'libero_cached_ab'))
    parser.add_argument('--resources', action='store_true')
    parser.add_argument('--report', type=Path)
    args, overrides = parser.parse_known_args()
    cfg = compose_recipe(args.task, overrides)
    result = {'task': args.task, 'checkpoint': inspect_checkpoint(cfg.init_checkpoint, cfg.joint96_posttrain.expected_pretrain_step), 'resume': cfg.resume, 'fresh_optimizer': True, 'batch_size': cfg.batch_size, 'gradient_accumulation_steps': cfg.gradient_accumulation_steps, 'learning_rate': cfg.learning_rate, 'num_epochs': cfg.num_epochs, 'max_steps': cfg.max_steps, 'eval_every': cfg.eval_every, 'action_video_attention': JOINT96_ACTION_VIDEO_ATTENTION, 'camera_slots': list(cfg.data.train.camera_slots), 'gpu_smoke': False}
    if args.resources:
        result['resources'] = check_resources(cfg)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
