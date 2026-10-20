"""Portable, offline pretraining entry. Raw datasets remain on shared storage."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['PHOENIX_ROOT'] = str(ROOT)
os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(ROOT / 'resources/weights')
os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def load_config(overrides):
    from omegaconf import OmegaConf
    from wam.pretrain_compat import canonical_pretrain_config
    cfg = OmegaConf.load(ROOT / 'configs/phoenix_pretrain.yaml')
    cfg = OmegaConf.merge(cfg, canonical_pretrain_config(OmegaConf.from_dotlist(overrides)))
    OmegaConf.resolve(cfg)
    return canonical_pretrain_config(cfg)


def validate_startup(cfg):
    from wam.model.modules.codecs.dino_csvae.provenance import BUNDLED_SOURCE, canonical_source_identity
    from wam.training.config_validation import validate_training_config
    from wam.training.checkpoint import validate_joint96_resume_directory
    from wam.resources import repository_root
    if repository_root() != ROOT:
        raise ValueError('Imported wam from another checkout')
    validate_training_config(cfg)
    if not cfg.model.joint96_three_view or cfg.model.load_text_encoder:
        raise ValueError('Only Joint96 with supplied T5 caches is packaged')
    if cfg.video_latent_cache != 'off':
        raise ValueError('Phoenix pretraining uses online visual encoding')
    for field in ('understanding', 'future_delta', 'memory', 'distillation'):
        if cfg.model.get(field, {}).get('enabled', False):
            raise ValueError('Unsupported auxiliary module: ' + field)
    if cfg.model.joint_visual_codec.source_root != BUNDLED_SOURCE:
        raise ValueError('An external encoder repository is not permitted')
    canonical_source_identity(cfg.model.joint_visual_codec)
    for path in (cfg.model.joint_visual_codec.csvae_checkpoint_path,
                 cfg.model.joint_visual_codec.dino_kwargs.checkpoint_path,
                 cfg.data.train.package_root, cfg.data.val.package_root,
                 cfg.local_robot_multiview.plan_root,
                 cfg.local_robot_multiview.rgb_selection_root,
                 cfg.local_robot_multiview.rgb_variants_root,
                 cfg.local_robot_softprompt.registry_path):
        p = Path(path).resolve()
        if not p.exists():
            raise ValueError('Missing resource: ' + str(p))
    if cfg.resume:
        p = Path(cfg.resume).resolve()
        validate_joint96_resume_directory(str(p), cfg=cfg,
            world_size=int(cfg.local_robot_joint96.expected_world_size))
    print('Phoenix startup: config, bundle, local resources and resume contract passed', flush=True)


def install_io_guard():
    """Reject an already imported wam from a different checkout."""
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if (name == 'wam' or name.startswith('wam.')) and filename:
            if not Path(filename).resolve().is_relative_to(ROOT / 'src/wam'):
                raise RuntimeError('wam module imported from another checkout: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('overrides', nargs='*', help='Configuration overrides in key=value form.')
    args = parser.parse_args()
    cfg = load_config(args.overrides)
    install_io_guard()
    validate_startup(cfg)
    from wam.runtime import run_training
    run_training(cfg)


if __name__ == '__main__':
    main()
