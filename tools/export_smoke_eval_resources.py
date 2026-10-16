"""Export the completed three-step smoke for a separate inference process.

Checkpoint data stays on the same filesystem via a hard link. Requires all
four training receipts; never exports a still-running or mutable formal run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from omegaconf import OmegaConf
from wam.posttrain.eval_contract import validate_evaluation_contract


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(strict=True), args.output.resolve()
    receipts = [json.loads((run/f'smoke_rank{rank}.json').read_text()) for rank in range(4)]
    assert all(r['passed'] and r['rank'] == rank and r['step'] == 3
               and r['skipped_steps'] == 0 for rank, r in enumerate(receipts))
    assert receipts[0]['save_readback_exact'] == receipts[0]['exact_initialization']
    source = Path(receipts[0]['saved_checkpoint']).resolve(strict=True)
    assert source.is_relative_to(run)
    cfg = OmegaConf.load(run/'config.yaml')
    assert cfg.max_steps == 3 and cfg.resume is None and not cfg.save_training_state
    payload = torch.load(source, mmap=True, weights_only=True, map_location='cpu')
    assert payload['step'] == 3
    pretrain_step = int(cfg.joint96_posttrain.expected_pretrain_step)
    assert pretrain_step > 0 and all(r['pretrain_step'] == pretrain_step for r in receipts)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('dataset_stats.json', 'initialization.json'):
        shutil.copyfile(run/name, output/name)
    shutil.copyfile(run/'config.yaml', output/'train_config.yaml')
    shutil.copyfile(cfg.data.train.pretrained_norm_stats, output/'normalization_source.json')
    validate_evaluation_contract(cfg, payload, output/'dataset_stats.json', expected_pretrain_step=pretrain_step)
    destination = output/'step_000003.pt'
    os.link(source, destination)
    report = dict(passed=True, pretrain_step=pretrain_step, posttrain_step=3,
        checkpoint_sha256=sha(destination), source_run=str(run),
        source_checkpoint=str(source), checkpoint=str(destination),
        scope='completed smoke only; not a model-quality checkpoint')
    (output/'export.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
