"""Read-only installation audit of the exact RoboTwin evaluation snapshot.

This never downloads assets, edits the simulator or launches a GPU process.
It is not an evaluation-time gate or a claim of numerical equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'assets/robotwin/simulator_manifest.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_tree(root, spec):
    rows = []
    for directory in spec['include_dirs']:
        for path in (root / directory).rglob('*'):
            relative = path.relative_to(root)
            if not path.is_file():
                continue
            if any(d in relative.parts or relative.as_posix().startswith(d + '/')
                   for d in spec.get('exclude_dirs', [])):
                continue
            if relative.as_posix() in spec.get('exclude_files', []):
                continue
            if path.suffix in spec.get('exclude_suffixes', []):
                continue
            if spec.get('include_suffixes') and path.suffix not in spec['include_suffixes']:
                continue
            rows.append([relative.as_posix(), file_sha256(path)])
    encoded = json.dumps(sorted(rows), separators=(',', ':')).encode()
    return len(rows), hashlib.sha256(encoded).hexdigest()


def check(root, manifest, *, source_only=False, archives=False):
    if source_only and archives:
        raise ValueError('--source-only and --archives cannot be combined')
    results = {}
    for name, directory in (('robotwin', root), ('curobo', root / 'envs/curobo')):
        spec = manifest[name]
        count, digest = source_tree(directory, spec)
        results[name] = dict(files_checked=count, expected_files=spec['file_count'],
                             tree_sha256=digest,
                             passed=count == spec['file_count'] and digest == spec['tree_sha256'])
    if not source_only:
        checks = manifest['assets']['key_file_sha256'].copy()
        if archives:
            checks.update({'assets/' + name: item['sha256']
                           for name, item in manifest['assets']['archives'].items()})
        for relative, expected in checks.items():
            path = root / relative
            actual = file_sha256(path) if path.is_file() else None
            results[relative] = dict(passed=actual == expected, sha256=actual)
    return dict(passed=all(item['passed'] for item in results.values()), checks=results,
                source_only=source_only, archives_checked=archives,
                all_extracted_assets_checked=False, simulator_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=os.environ.get('ROBOTWIN_ROOT'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source-only', action='store_true')
    group.add_argument('--archives', action='store_true')
    args = parser.parse_args()
    if args.root is None or not args.root.is_dir():
        parser.error('Set --root or ROBOTWIN_ROOT to an existing simulator directory')
    report = check(args.root, json.loads(args.manifest.read_text()),
                   source_only=args.source_only, archives=args.archives)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
