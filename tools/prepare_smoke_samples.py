"""Read a bounded set of real cached LIBERO samples into a standalone smoke artifact."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT/'.cache/huggingface/datasets'))
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wam.posttrain.guard import install_import_guard, validate_scope
from wam.utils.config_resolvers import register_default_resolvers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    install_import_guard()
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=libero_posttrain'])
    validate_scope(cfg)
    dataset = instantiate(cfg.data.train)
    indices = [int((i+.5)*len(dataset)/4) for i in range(4)]
    samples = []
    for index in indices:
        sample = dataset[index]
        sample['embodiment_id'] = torch.tensor(0, dtype=torch.long)
        samples.append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'scope': 'four real cached training windows, not a dataset copy',
        'indices': indices, 'samples': samples}, args.output)
    report = {'dataset_length': len(dataset), 'indices': indices,
        'shapes': {k: list(v.shape) for k,v in samples[0].items() if torch.is_tensor(v)},
        'prompts': [sample['prompt'] for sample in samples]}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
