"""Create an audited runtime copy; never edit the original PPU saved config."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from omegaconf import OmegaConf
from wam.posttrain.guard import validate_scope
from wam.utils.config_resolvers import register_default_resolvers
from wam.model.modules.codecs.dino_stvae.provenance import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    register_default_resolvers()
    if args.output.exists() or args.output.with_suffix('.relocation.json').exists():
        raise FileExistsError(args.output)
    original = args.source.read_bytes()
    cfg = OmegaConf.load(args.source)
    paths = {
        'init_checkpoint': os.environ['WAM_PRETRAIN_CHECKPOINT'],
        'local_robot_softprompt.registry_path': os.environ['ROBOCASA_REGISTRY'],
        'model.action_dit_config.embodiment_registry_path': os.environ['ROBOCASA_REGISTRY'],
        'model.joint_visual_codec.source_root': 'bundled://dino-stvae-v1',
        'model.joint_visual_codec.source_sha256': verify_bundle()['source_sha256'],
        'model.joint_visual_codec.stvae_checkpoint_path': os.environ['STVAE_CHECKPOINT'],
        'model.joint_visual_codec.dino_kwargs.checkpoint_path': os.environ['DINO_ENCODER_CHECKPOINT'],
        'model.joint_visual_codec.dino_kwargs.starvla_source_path': 'bundled://dino-stvae-v1/third_party/dino_runtime',
        'model.joint_visual_codec.dino_kwargs.dinov3_source_path': 'bundled://dino-stvae-v1/third_party/dino_runtime/dinov3',
    }
    for split in ('train', 'val'):
        paths.update({f'data.{split}.package_root': os.environ['ROBOCASA_PACKAGE'],
                      f'data.{split}.cache_root': os.environ['ROBOCASA_CACHE'],
                      f'data.{split}.pretrained_norm_stats': os.environ['ROBOCASA_PACKAGE'] + '/norm.json'})
    changes = {}
    for key, value in paths.items():
        before = OmegaConf.select(cfg, key)
        OmegaConf.update(cfg, key, value)
        changes[key] = {'source': before, 'local': value}
    validate_scope(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    receipt = {'source': str(args.source.resolve()), 'output': str(args.output.resolve()),
               'source_sha256': hashlib.sha256(original).hexdigest(),
               'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'path_map_sha256': hashlib.sha256(Path(os.environ['ROBOCASA_PATH_MAP']).read_bytes()).hexdigest(),
               'changes': changes, 'algorithm_parameters_changed': False}
    args.output.with_suffix('.relocation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert args.source.read_bytes() == original
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
