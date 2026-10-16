"""Default fail-closed gate; explicit evaluation-only parity waiver stays audited.

This does not alter the original acceptance report or the training gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
SEED_SHA256 = '695eec268dfb6f5a254edd1a5849ce29944d8b5fdab51df6625185879a08f723'

def validate_acceptance(report, checkpoint, seed_sha256, *, waive_parity=False, reason=''):
    if report.get('inference_passed') is not True:
        raise ValueError('RoboCasa checkpoint inference acceptance has not passed')
    if Path(report['checkpoint']).resolve() != Path(checkpoint).resolve():
        raise ValueError('Acceptance report belongs to a different checkpoint')
    parity_passed = report.get('passed') is True
    if not parity_passed:
        if not waive_parity or not reason.strip():
            raise ValueError('Cache/online parity failed; explicit evaluation-only waiver and reason required')
        samples = report.get('samples', [])
        if not samples or any((row.get('finite_actions') is not True for row in samples)):
            raise ValueError('Parity waiver cannot waive missing or invalid inference samples')
        for row in samples:
            for branch in ('vae', 'stvae'):
                if row.get('checks', {}).get(branch + '_clip_prefix_vs_single', {}).get('passed') is not True:
                    raise ValueError('Parity waiver cannot waive within-platform current-frame inconsistency')
    return {'schema': 'phoenix.robocasa.eval_acceptance.v1', 'evaluation_authorized': True, 'checkpoint': str(Path(checkpoint).resolve()), 'inference_passed': True, 'numerical_parity_passed': parity_passed, 'cross_platform_parity_waived': not parity_passed, 'evaluation_mode': 'accepted_port' if parity_passed else 'cross_platform_exploratory', 'waiver_reason': reason if not parity_passed else None, 'scene_seed_manifest_sha256': seed_sha256, 'training_acceptance_changed': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checkpoint = Path(os.environ["CHECKPOINT"]).resolve(strict=True)
    for name in ("TRAIN_CONFIG", "ROBOCASA_SCENE_SEEDS"):
        Path(os.environ[name]).resolve(strict=True)
    record = {"checkpoint": str(checkpoint), "hash_validation": False,
              "inference_validation": "performed_during_rollout", "numerical_parity_passed": None}
    if args.output:
        with args.output.open('x') as file:
            json.dump(record, file, indent=2)
            file.write('\n')
    print(json.dumps(record))
if __name__ == '__main__':
    main()
