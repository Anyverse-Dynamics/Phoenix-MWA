from __future__ import annotations

from collections.abc import Sequence
from typing import Any


CONFIG_MODULE = "prometheus.configs"


def compose_config(config_name: str, *, overrides: Sequence[str] = ()) -> Any:
    """Compose packaged Hydra config from tests, notebooks, or small scripts.

    Workflow entrypoints should still use @hydra.main so CLI overrides and
    Hydra output handling stay standard.
    """

    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module=CONFIG_MODULE, version_base=None):
        return compose(config_name=str(config_name), overrides=list(overrides))
