"""Bootstrap local LIBERO imports for evaluation scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def setup_libero_paths() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    project_root = Path(__file__).resolve().parents[2]
    repo_name = os.environ.get("WAM_LIBERO_REPO", "LIBERO-plus")
    if repo_name not in {"LIBERO", "LIBERO-plus", "LIBERO-PRO"}:
        raise ValueError(f"Unsupported WAM_LIBERO_REPO={repo_name!r}")

    override_root = str(project_root / 'third_party/LIBERO-plus')
    if repo_name == 'LIBERO-PRO':
        override_root = os.environ['LIBERO_PRO_REPO_ROOT']
    libero_repo_root = (
        Path(os.path.expanduser(os.path.expandvars(override_root)))
        if override_root
        else project_root / "third_party" / repo_name
    )
    libero_package_root = libero_repo_root / "libero"
    if not (libero_package_root / 'libero' / '__init__.py').is_file():
        raise FileNotFoundError(libero_package_root / 'libero' / '__init__.py')
    loaded = sys.modules.get('libero.libero')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(libero_repo_root.resolve()):
        raise RuntimeError('A different LIBERO package is already imported; use a fresh process')
    libero_benchmark_root = Path(os.environ['LIBERO_BENCHMARK_ROOT']).resolve(strict=True)
    for resource in ('bddl_files', 'init_files', 'assets'):
        if not (libero_benchmark_root / resource).is_dir():
            raise FileNotFoundError(libero_benchmark_root / resource)

    if libero_repo_root.exists():
        for candidate in (
            project_root / "third_party" / "LIBERO",
            project_root / "third_party" / "LIBERO-plus",
            libero_repo_root,
        ):
            candidate_path = str(candidate)
            while candidate_path in sys.path:
                sys.path.remove(candidate_path)
        package_path = str(libero_repo_root)
        sys.path.insert(0, package_path)

    config_dir = Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH",
            str(project_root / ".cache" / ("libero_plus" if repo_name == "LIBERO-plus" else "libero")),
        )
    )
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {libero_benchmark_root}",
                f"bddl_files: {libero_benchmark_root / 'bddl_files'}",
                f"init_states: {libero_benchmark_root / 'init_files'}",
                f"datasets: {libero_benchmark_root.parent / 'datasets'}",
                f"assets: {libero_benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
