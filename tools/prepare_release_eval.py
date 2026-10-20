"""Prepare local evaluation paths from a verified release; never launch a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.prepare_eval import interpreter_path

def sha256(path):
    """File existence only; runtime content hashing is disabled."""
    Path(path).resolve(strict=True)
    return None

class Release:

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.manifest_path = self.root / 'release_manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != 'phoenix.eval_release.v1':
            raise ValueError('Unknown evaluation release schema')
        if self.manifest.get('complete') is not True or any(((self.root / name).exists() or (self.root / name).is_symlink() for name in ('INCOMPLETE.json', 'INCOMPLETE'))):
            raise ValueError('Evaluation release is incomplete')
        self.verified = {}

    def file(self, relative):
        part = PurePosixPath(relative)
        if not relative or part.is_absolute() or '..' in part.parts or ('\\' in relative) or (part.as_posix() != relative):
            raise ValueError(f'Invalid release path: {relative!r}')
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'Release reference escapes the bundle: {relative}')
        if relative not in self.verified:
            record = self.manifest.get('files', {}).get(relative, {})
            if 'bytes' in record and path.stat().st_size != record['bytes']:
                raise ValueError(f'Release size mismatch: {relative}')
            self.verified[relative] = {'exists': True, 'hash_validation': False}
        return path

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def relocate_metadata(release, model_directory, package, output):
    """Restore source ns mtimes only after checking each copied metadata payload."""
    mappings = {}
    for original, record in package['metadata'].items():
        source = PurePosixPath(original)
        if not source.is_absolute() or len(source.parts) < 4 or source.parts[-2] != 'meta':
            raise ValueError(f'Unexpected task metadata location: {original}')
        relative = PurePosixPath(*source.parts[-3:])
        bundled = release.file(f'{model_directory}/metadata/{relative}')
        if bundled.stat().st_size != record['size']:
            raise ValueError(f'Task metadata differs from the original package: {original}')
        destination = output / 'metadata' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled, destination)
        os.utime(destination, ns=(record['mtime_ns'], record['mtime_ns']))
        if destination.stat().st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'Filesystem cannot preserve task metadata ns mtime: {destination}')
        mappings[original] = str(destination)
    return {'schema': 'phoenix.robocasa.path_map.v1', 'mappings': mappings}

def prepare(args):
    from wam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    release = Release(args.release_root)
    model = release.manifest['models'][args.model]
    checkpoint = release.file(model['checkpoint'])
    model_directory = PurePosixPath(model['checkpoint']).parent.as_posix()
    for relative in release.manifest['files']:
        if args.model == 'robocasa' and relative not in (model['checkpoint'], model['config']):
            continue  # Historical training packages/caches are not inference inputs.
        if relative.startswith(model_directory + '/'):
            release.file(relative)
    encoders = {}
    for name in ('dino', 'csvae'):
        record = release.manifest['encoders'][name]
        encoders[name] = release.file(record['path'])
    cfg = OmegaConf.load(release.file(model['config']))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new runtime directory: {output}')
    model_base = Path(args.model_base).resolve(strict=True)
    if args.model == 'libero':
        if not args.libero_benchmark_root:
            raise ValueError('--libero-benchmark-root is required for LIBERO')
        benchmark = Path(args.libero_benchmark_root).resolve(strict=True)
        for name in ('assets', 'bddl_files', 'init_files'):
            if not (benchmark / name).is_dir():
                raise FileNotFoundError(benchmark / name)
    else:
        if not args.robocasa_resources:
            raise ValueError('--robocasa-resources is required for RoboCasa simulator forks/assets')
        resources = Path(args.robocasa_resources).resolve(strict=True)
        from wam.inference.robocasa_resources import DEFAULT_PACKAGE, DEFAULT_SEEDS, load_inference_package
        package_root = DEFAULT_PACKAGE
        load_inference_package(package_root)
        # Preserve a supplied release's original evaluation schedule if present.
        seed_relative = f'{model_directory}/scene_seed_list.json'
        seeds = release.file(seed_relative) if (release.root/seed_relative).is_file() else DEFAULT_SEEDS
        from wam.robocasa_seeds import load_scene_seed_list
        load_scene_seed_list(seeds, 1)
        from wam.inference.robocasa_normalization import resolve_normalization_stats
        stats = resolve_normalization_stats(cfg, override=getattr(args, 'stats', None))
    output.mkdir(parents=True)
    from wam.model.modules.codecs.dino_csvae.provenance import verify_bundle
    bundled_source_sha256 = verify_bundle()['source_sha256']
    registry = ROOT / 'configs/embodiment_registry' / ('local_robot37_arm18_capacity28_v1.json' if args.model == 'libero' else 'robocasa_gr1_delta80_slot18_v1.json')
    updates = {'init_checkpoint': None, 'resume': None, 'local_robot_softprompt.init_checkpoint': None, 'local_robot_softprompt.registry_path': str(registry), 'model.action_dit_config.embodiment_registry_path': str(registry), 'model.joint_visual_codec.source_root': 'bundled://dino-csvae-v2', 'model.joint_visual_codec.source_sha256': bundled_source_sha256, 'model.joint_visual_codec.csvae_checkpoint_path': str(encoders['csvae']), 'model.joint_visual_codec.dino_kwargs.checkpoint_path': str(encoders['dino']), 'model.joint_visual_codec.dino_kwargs.starvla_source_path': 'bundled://dino-csvae-v2/third_party/dino_runtime', 'model.joint_visual_codec.dino_kwargs.dinov3_source_path': 'bundled://dino-csvae-v2/third_party/dino_runtime/dinov3'}
    env = {'DINO_ENCODER_CHECKPOINT': str(encoders['dino']), 'CSVAE_CHECKPOINT': str(encoders['csvae']), 'DIFFSYNTH_MODEL_BASE_PATH': str(model_base)}
    if args.model == 'libero':
        for name in ('dataset_stats.json', 'initialization.json'):
            shutil.copyfile(release.file(f'{model_directory}/{name}'), output / name)
        (output / 'policy.pt').symlink_to(checkpoint)
        config_path = output / 'train_config.yaml'
        env.update(PYTHON=args.policy_python, JOINT96_EVAL_RESOURCE_DIR=str(output), JOINT96_EVAL_CHECKPOINT_FILENAME='policy.pt', LIBERO_BENCHMARK_ROOT=str(benchmark), LIBERO_CONFIG_PATH=str(output / 'libero_paths'), EVAL_TEXT_CACHE=str(output / 'text_cache'), EVAL_OUTPUT=str(output / 'evaluation'), EVAL_ENV_BACKEND='inprocess', EVAL_RENDER_BACKEND=args.render_backend)
    else:
        updates['inference.normalization_stats'] = str(stats)
        updates['inference.text_cache_dir'] = str(output/'text_cache')
        for split in ('train', 'val'):
            updates.update({f'data.{split}.package_root': str(package_root), f'data.{split}.cache_root': str(output / 'unused_training_cache'), f'data.{split}.pretrained_norm_stats': str(stats)})
        config_path = output / 'config.yaml'
        env.update(POLICY_PYTHON=args.policy_python, CHECKPOINT=str(checkpoint), TRAIN_CONFIG=str(config_path), ROBOCASA_PACKAGE=str(package_root), ROBOCASA_REGISTRY=str(registry), ROBOCASA_RESOURCES=str(resources), ROBOCASA_SCENE_SEEDS=str(seeds), EVAL_ROOT=str(output / 'evaluation'), OUTROOT=str(output / 'evaluation'), MUJOCO_GL=args.render_backend, PYOPENGL_PLATFORM=args.render_backend)
        if args.sim_python:
            env['SIM_PYTHON'] = str(interpreter_path(args.sim_python))
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    env_path = output / 'eval.env.sh'
    env_path.write_text('# Generated evaluation paths; source this file before the documented launcher.\n' + ''.join((f'export {key}={shlex.quote(str(value))}\n' for key, value in env.items())))
    receipt = {'schema': 'phoenix.eval_runtime.v1', 'model': args.model, 'release_manifest_sha256': sha256(release.manifest_path), 'verified_files': release.verified, 'checkpoint': str(checkpoint), 'config': str(config_path), 'env': str(env_path), 'private_initialization_required': False, 'evaluation_launched': False, 'configuration_path_updates': updates}
    write_json(output / 'prepare.json', receipt)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--model', choices=('libero', 'robocasa'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-base', type=Path, required=True)
    parser.add_argument('--policy-python', default=sys.executable)
    parser.add_argument('--libero-benchmark-root', type=Path)
    parser.add_argument('--robocasa-resources', type=Path)
    parser.add_argument('--stats', type=Path, help='Override RoboCasa inference normalization statistics')
    parser.add_argument('--sim-python', type=Path)
    parser.add_argument('--render-backend', choices=('egl', 'osmesa'), default='egl')
    print(json.dumps(prepare(parser.parse_args()), indent=2))
if __name__ == '__main__':
    main()
