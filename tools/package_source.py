"""Produce a source-only deployment archive with an explicit allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from audit_public_source import inspect

ROOT = Path(__file__).resolve().parents[1]
DIRS = ('src', 'configs', 'scripts', 'eval', 'tools', 'tests', 'assets',
        'third_party', 'LICENSES', 'docs')
FILES = ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE', '.gitignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [p for directory in DIRS for p in (ROOT/directory).rglob('*') if p.is_file()]
    paths += [ROOT/name for name in FILES if (ROOT/name).is_file()]
    records = {}
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT)
        if any(part in ('__pycache__', '.pytest_cache', '.git') or part.endswith('.egg-info') for part in rel.parts) or path.suffix in ('.pyc','.pyo'):
            continue
        if path.is_symlink() or path.suffix in ('.pt','.safetensors','.bin','.pem','.key') or any(
                part in ('.secrets','.env','runs','evaluate_results','validation_runs') for part in rel.parts):
            raise ValueError(f'Forbidden deployment file: {rel}')
        if path.stat().st_size > 20*1024*1024:
            raise ValueError(f'Unexpected large source file: {rel}')
        findings = inspect(path)
        if findings:
            raise ValueError(f'Public source audit failed: {rel}: {findings}')
        records[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'x:gz') as archive:
        for relative in records:
            archive.add(ROOT/relative, arcname=relative, recursive=False)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps({'files': len(records), 'bytes': args.output.stat().st_size,
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
