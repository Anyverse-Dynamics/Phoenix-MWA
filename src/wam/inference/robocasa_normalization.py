"""Resolve evaluation statistics independently of the training/text package."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATS = 'assets/stats/phoenix_robocasa.json'


def resolve_normalization_stats(cfg, *, override=None):
    """Config paths are repository-relative; an explicit CLI path is cwd-relative.

    Older configs without an inference section use the released source asset,
    never data.train.pretrained_norm_stats or PACKAGE/norm.json. A missing
    explicitly configured file is an error, not permission to use a fallback.
    """
    value = override if override is not None else OmegaConf.select(
        cfg, 'inference.normalization_stats', default=DEFAULT_STATS)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError('Set inference.normalization_stats to a matching statistics file')
    path = Path(value).expanduser()
    if not path.is_absolute() and override is None:
        path = ROOT / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'Normalization statistics must be a file: {path}')
    return path
