"""Prepare evaluation directly from a checkpoint and its saved configuration.

No release manifest, checksum, source snapshot or previous acceptance report is
required. This command prepares files only and does not run policy inference.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def existing(value):
    return Path(value).expanduser().resolve(strict=True)


def interpreter_path(value):
    """Preserve venv launcher symlinks: resolving them selects the base Python."""
    path = Path(value).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f'Interpreter is not executable: {path}')
    return path


def metadata_mapping(package, metadata_root):
    metadata_root = existing(metadata_root)
    mappings = {}
    for source in package.get('metadata', {}):
        parts = Path(source).parts
        if len(parts) < 3 or parts[-2] != 'meta' or '..' in parts[-3:]:
            raise ValueError(f'Unexpected task metadata layout: {source}')
        mappings[source] = str(existing(metadata_root.joinpath(*parts[-3:])))
    return mappings


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from wam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    checkpoint, config = existing(args.checkpoint), existing(args.config)
    dino, csvae, model_base = (
        existing(args.dino),
        existing(args.csvae),
        existing(args.model_base),
    )
    cfg = OmegaConf.load(config)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    from wam.utils.checkpoint_steps import checkpoint_steps

    progress = checkpoint_steps(payload)
    step, pretrain_step = progress['step'], progress['pretrain_step']
    del payload
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if args.benchmark in ('libero-plus', 'libero-pro'):
        if not args.stats:
            raise ValueError('--stats is required for LIBERO')
        stats = existing(args.stats)
        benchmark = existing(args.benchmark_root)
        for item in ('assets', 'bddl_files', 'init_files'):
            if not (benchmark / item).is_dir():
                raise FileNotFoundError(benchmark / item)
        if args.benchmark == 'libero-pro':
            if not args.libero_repo_root:
                raise ValueError('--libero-repo-root is required for LIBERO-Pro')
            pro_repo = existing(args.libero_repo_root)
            if not (pro_repo / 'libero/libero/benchmark/__init__.py').is_file():
                raise FileNotFoundError(
                    pro_repo / 'libero/libero/benchmark/__init__.py'
                )
    else:
        if not args.resources:
            raise ValueError('RoboCasa needs --resources for simulator sources/assets')
        from wam.inference.robocasa_resources import (
            DEFAULT_PACKAGE,
            DEFAULT_SEEDS,
            load_inference_package,
        )

        package = existing(args.package or DEFAULT_PACKAGE)
        resources, seeds = existing(args.resources), existing(
            args.scene_seeds or DEFAULT_SEEDS
        )
        load_inference_package(package)
        from wam.robocasa_seeds import load_scene_seed_list

        load_scene_seed_list(seeds, 1)
        from wam.inference.robocasa_normalization import resolve_normalization_stats

        stats = resolve_normalization_stats(cfg, override=getattr(args, 'stats', None))
    registry = (
        ROOT
        / 'configs/embodiment_registry'
        / (
            'local_robot37_arm18_capacity28_v1.json'
            if args.benchmark in ('libero-plus', 'libero-pro')
            else 'robocasa_gr1_delta80_slot18_v1.json'
        )
    )
    updates = {
        'init_checkpoint': None,
        'resume': None,
        'local_robot_softprompt.init_checkpoint': None,
        'local_robot_softprompt.registry_path': str(registry),
        'model.action_dit_config.embodiment_registry_path': str(registry),
        'model.joint_visual_codec.source_root': 'bundled://dino-csvae-v2',
        'model.joint_visual_codec.source_sha256': None,
        'model.joint_visual_codec.csvae_checkpoint_path': str(csvae),
        'model.joint_visual_codec.dino_kwargs.checkpoint_path': str(dino),
        'model.joint_visual_codec.dino_kwargs.starvla_source_path': 'bundled://dino-csvae-v2/third_party/dino_runtime',
        'model.joint_visual_codec.dino_kwargs.dinov3_source_path': 'bundled://dino-csvae-v2/third_party/dino_runtime/dinov3',
        'model.joint_visual_codec.dino_kwargs.verify_source_commit': False,
    }
    env = dict(
        PYTHON=sys.executable,
        POLICY_PYTHON=sys.executable,
        DIFFSYNTH_MODEL_BASE_PATH=str(model_base),
        DINO_ENCODER_CHECKPOINT=str(dino),
        CSVAE_CHECKPOINT=str(csvae),
        EVAL_RENDER_BACKEND=args.render_backend,
        MUJOCO_GL=args.render_backend,
        PYOPENGL_PLATFORM=args.render_backend,
    )
    output.mkdir(parents=True)
    if args.benchmark in ('libero-plus', 'libero-pro'):
        shutil.copyfile(stats, output / 'dataset_stats.json')
        (output / 'policy.pt').symlink_to(checkpoint)
        (output / 'initialization.json').write_text(
            json.dumps({'pretrain_step': pretrain_step}) + '\n'
        )
        config_target = output / 'train_config.yaml'
        env.update(
            JOINT96_EVAL_RESOURCE_DIR=str(output),
            JOINT96_EVAL_CHECKPOINT_FILENAME='policy.pt',
            LIBERO_BENCHMARK_ROOT=str(benchmark),
            EVAL_OUTPUT=str(output / 'evaluation'),
            EVAL_TEXT_CACHE=str(output / 'text_cache'),
            EVAL_ENV_BACKEND='inprocess',
        )
        if args.benchmark == 'libero-pro':
            env.update(WAM_LIBERO_REPO='LIBERO-PRO', LIBERO_PRO_REPO_ROOT=str(pro_repo))
    else:
        config_target = output / 'config.yaml'
        updates['inference.normalization_stats'] = str(stats)
        updates['inference.text_cache_dir'] = str(output / 'text_cache')
        for split in ('train', 'val'):
            updates.update(
                {
                    f'data.{split}.package_root': str(package),
                    f'data.{split}.pretrained_norm_stats': str(stats),
                }
            )
        env.update(
            CHECKPOINT=str(checkpoint),
            TRAIN_CONFIG=str(config_target),
            ROBOCASA_PACKAGE=str(package),
            ROBOCASA_RESOURCES=str(resources),
            ROBOCASA_SCENE_SEEDS=str(seeds),
            OUTROOT=str(output / 'evaluation'),
            ROBOCASA_REGISTRY=str(registry),
        )
        if args.sim_python:
            env['SIM_PYTHON'] = str(interpreter_path(args.sim_python))
        if args.metadata_root:
            package_info = json.loads((package / 'package.json').read_text())
            mappings = metadata_mapping(package_info, args.metadata_root)
            path_map = output / 'paths.json'
            path_map.write_text(
                json.dumps(
                    {'schema': 'phoenix.robocasa.path_map.v1', 'mappings': mappings},
                    indent=2,
                )
                + '\n'
            )
            env['ROBOCASA_PATH_MAP'] = str(path_map)
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    # Preserve unresolved training-only paths; inference does not instantiate data.
    OmegaConf.save(cfg, config_target, resolve=False)
    target = output / 'eval.env.sh'
    target.write_text(
        ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in env.items())
    )
    (output / 'prepare.json').write_text(
        json.dumps(
            dict(
                benchmark=args.benchmark,
                checkpoint=str(checkpoint),
                step=step,
                normalization_stats=str(stats),
                hash_validation=False,
                inference_launched=False,
            ),
            indent=2,
        )
        + '\n'
    )
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--benchmark',
        choices=('libero-plus', 'libero-pro', 'robocasa-gr1'),
        required=True,
    )
    p.add_argument(
        '--libero-repo-root',
        help='External LIBERO-PRO checkout (not installed over LIBERO-Plus)',
    )
    for name in ('checkpoint', 'config', 'dino', 'csvae', 'model-base', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument(
        '--stats',
        help='LIBERO statistics (required), or an explicit RoboCasa inference statistics override',
    )
    p.add_argument('--benchmark-root', default='third_party/LIBERO-plus/libero/libero')
    p.add_argument(
        '--package',
        help='Optional inference manifest directory; defaults to assets/robocasa',
    )
    p.add_argument(
        '--scene-seeds',
        help='Optional saved scene seed list; defaults to assets/robocasa/scene_seed_list.json',
    )
    for name in ('resources', 'sim-python'):
        p.add_argument('--' + name)
    p.add_argument(
        '--metadata-root',
        help='Legacy training metadata relocation; not needed for inference',
    )
    p.add_argument('--render-backend', choices=('egl', 'osmesa'), default='egl')
    print(prepare(p.parse_args()))


if __name__ == '__main__':
    main()
