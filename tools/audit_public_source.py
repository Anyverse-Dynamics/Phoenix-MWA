"""Audit Phoenix public source for private deployment data and credentials.

Read-only release check; this tool does not rename fields, migrate checkpoints,
or participate in training/evaluation startup.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private_path': re.compile(r'/(?:mnt|home|root)/[A-Za-z0-9_.-]+/'),
    'windows_user_path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]', re.I),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'access_token': re.compile(r'\b(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|wandb_v1_[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16})\b'),
    'credential_url': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
}
IP = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')


def inspect(path, root=ROOT):
    rel = path.relative_to(root)
    if path.is_symlink(): return [('symlink', 0)]
    forbidden = any(part.startswith(('.env', '.secrets')) or part in
                    ('.aws', '.ssh', 'credentials', 'id_rsa', 'id_ed25519') for part in rel.parts)
    if forbidden or path.suffix.lower() in ('.pem', '.key', '.pt', '.pth', '.safetensors'):
        return [('private_file', 0)]
    # Explicitly reviewed public architecture figure used by the README.
    if rel.as_posix() in ('docs/assets/main.png', 'docs/assets/phoenix-mwa-title.png'):
        with path.open('rb') as image:
            return [] if image.read(8) == b'\x89PNG\r\n\x1a\n' else [('invalid_image', 0)]
    try: content = path.read_text(encoding='utf-8')
    except UnicodeDecodeError: return [('unreviewed_binary', 0)]
    findings = []
    for i, line in enumerate(content.splitlines(), 1):
        for rule, pattern in RULES.items():
            if pattern.search(line): findings.append((rule, i))
        for match in IP.finditer(line):
            ip = match.group()
            # Exclude loopback/bind addresses and dotted package versions.
            if ip not in ('127.0.0.1', '0.0.0.0') and all(int(v) <= 255 for v in ip.split('.')):
                prefix = line[:match.start()]
                if not prefix.endswith(('==', '>=', '<=', 'version=', 'version: ')):
                    findings.append(('host_address_review', i))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    findings, count = [], 0
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(part in ('.git', '__pycache__', '.pytest_cache') for part in p.relative_to(root).parts): continue
        count += 1
        findings.extend({'file': p.relative_to(root).as_posix(), 'rule': rule, 'line': line}
                        for rule, line in inspect(p, root))
    print(json.dumps({'files_checked': count, 'findings': findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__': main()
