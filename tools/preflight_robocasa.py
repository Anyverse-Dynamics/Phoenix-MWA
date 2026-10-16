"""CPU resource/provenance checks; optionally select an explicitly bounded smoke."""
import argparse
import json
import os
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from wam.datasets.robocasa_posttrain import load_package, RoboCasaTextCache, numeric_episode
from wam.datasets.robocasa_visual_cache import load_plan, validate_complete, file_hash, atomic_json
from wam.posttrain.robocasa_paths import require_migrated_cache
from wam.posttrain.guard import validate_scope
from wam.training.robocasa_posttrain import validate_cached_initialization
from wam.training.config_validation import validate_training_config
from wam.utils.config_resolvers import register_default_resolvers
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-selection', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task=robocasa_posttrain', *args.overrides])
    validate_scope(cfg)
    validate_training_config(cfg)
    package = load_package(cfg.data.train.package_root)
    plan = load_plan(package['plan_path'])
    text = RoboCasaTextCache(cfg.data.train.package_root, package)
    for prompt in package['prompts']:
        text.get(prompt)
    payload = torch.load(cfg.init_checkpoint, mmap=True, map_location='cpu', weights_only=True)
    expected_step = int(cfg.joint96_posttrain.expected_pretrain_step)
    identity = validate_cached_initialization(payload, plan, expected_step)
    key = 'mixtures.action.embodiment_prompt.embedding.weight'
    if bool(payload['mot'][key][18].count_nonzero()):
        raise ValueError('Parent slot18 is not reserved/empty')
    del payload
    selection = {'package_sha256': package['sha256'], 'plan_sha256': plan['sha256'], 'train': [], 'val': []}
    samples = []
    if args.smoke_selection:
        if args.smoke_selection.exists():
            raise FileExistsError(args.smoke_selection)
        for ep in package['episodes']:
            split = ep['split']
            if len(selection[split]) >= 2:
                continue
            path = Path(cfg.data.train.cache_root) / 'episodes' / ep['task'] / f"episode_{ep['episode']:06d}"
            if not (path / 'COMPLETE.json').exists() or not (path / 'features.bf16.bin').exists():
                continue
            validate_complete(cfg.data.train.cache_root, ep if 'split' not in ep else next((x for x in plan['episodes'] if (x['task'], x['episode']) == (ep['task'], ep['episode']))), plan['sha256'])
            state, action, mask, _ = numeric_episode(ep, 'delta80_slot18')
            selection[split].append([ep['task'], ep['episode']])
            samples.append({'task': ep['task'], 'episode': ep['episode'], 'split': split, 'state_shape': list(state.shape), 'action_shape': list(action.shape), 'valid_action_dimensions': int(mask[0].sum())})
            if all((len(selection[s]) == 2 for s in ('train', 'val'))):
                break
        if not all((len(selection[s]) == 2 for s in ('train', 'val'))):
            raise ValueError('Not enough fully copied episodes for a 2 train + 2 val smoke')
        atomic_json(args.smoke_selection, selection)
    else:
        require_migrated_cache(cfg.data.train.cache_root, sorted({e['task'] for e in plan['episodes']}))
        parity = read_acceptance = os.environ.get('ROBOCASA_ACCEPTANCE_REPORT')
        if not parity:
            raise ValueError('Formal training requires ROBOCASA_ACCEPTANCE_REPORT, not just copied caches')
        accepted = json.loads(Path(read_acceptance).read_text())
        if accepted.get('passed') is not True or accepted.get('inference_passed') is not True:
            raise ValueError('Cache/online numerical acceptance has not passed')
    report = {'status': 'PASS', 'scope': 'bounded_smoke_resources' if args.smoke_selection else 'formal_resource_preflight', 'hash_validation': False, 'splits': {k: len(v) for k, v in package['membership'].items()}, 'text_prompts_verified': len(package['prompts']), 'pretrain_step': expected_step, 'parent_slot18_empty': True, 'visual_identity_compatible': True, 'samples': samples, 'full_local_cache_verified': not bool(args.smoke_selection)}
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
