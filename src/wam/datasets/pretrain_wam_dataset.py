"""Original grouped normalization used by the Phoenix robot reader."""
from typing import Any
import torch

class _PretrainGroupedNormalizer:
    def __init__(
        self,
        stats: dict[str, Any],
        *,
        mode: str = "q01/q99",
        minimum_widths: dict[str, Any] | None = None,
        passthrough_dims: dict[str, Any] | None = None,
    ) -> None:
        self.mode = str(stats.get("norm_mode") or mode)
        self.minimum_widths = minimum_widths or {}
        self.passthrough_dims: dict[str, tuple[int, ...]] = {}
        for kind in ("action", "state"):
            raw_dims = (passthrough_dims or {}).get(kind, ())
            if isinstance(raw_dims, int):
                raw_dims = [raw_dims]
            if not isinstance(raw_dims, (list, tuple)):
                raise TypeError(
                    f"stats.passthrough_dims.{kind} must be an int/list"
                )
            dims = tuple(sorted({int(dim) for dim in raw_dims}))
            if any(dim < 0 for dim in dims):
                raise ValueError(
                    f"stats.passthrough_dims.{kind} contains a negative dim"
                )
            self.passthrough_dims[kind] = dims
        raw_groups = stats.get("groups") if isinstance(stats, dict) else None
        if not isinstance(raw_groups, dict):
            raw_groups = {"default": stats}
        self.groups: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        for group, group_stats in raw_groups.items():
            if not isinstance(group_stats, dict):
                continue
            self.groups[str(group)] = {}
            for kind in ("action", "state"):
                params = self._params(group_stats, kind, str(group))
                self.groups[str(group)][kind] = self._apply_passthrough(
                    params, kind=kind
                )
        if not self.groups:
            raise ValueError("Grouped pretrain stats contains no usable groups.")

    def _apply_passthrough(
        self,
        params: tuple[torch.Tensor, torch.Tensor],
        *,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dims = self.passthrough_dims.get(kind, ())
        if not dims:
            return params
        scale, offset = (tensor.clone() for tensor in params)
        if max(dims) >= int(scale.numel()):
            raise IndexError(
                f"stats.passthrough_dims.{kind}={list(dims)} exceeds "
                f"feature dim {scale.numel()}"
            )
        index = torch.as_tensor(dims, dtype=torch.long)
        scale[index] = 1.0
        offset[index] = 0.0
        return scale, offset

    def _feature_stats(self, group_stats: dict[str, Any], kind: str) -> dict[str, Any]:
        value = group_stats.get(kind)
        if isinstance(value, dict) and "default" in value:
            return value["default"]
        if isinstance(value, dict):
            return value
        raise KeyError(f"Stats group is missing {kind!r} stats.")

    def _params(self, group_stats: dict[str, Any], kind: str, group: str) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self._feature_stats(group_stats, kind)
        prefix = "global_"
        if self.mode == "z-score":
            mean = torch.as_tensor(stats[prefix + "mean"], dtype=torch.float32)
            std = torch.as_tensor(stats[prefix + "std"], dtype=torch.float32).clamp_min(1e-8)
            return 1.0 / std, -mean / std
        if self.mode == "min/max":
            lo_key, hi_key = prefix + "min", prefix + "max"
        elif self.mode == "q01/q99":
            lo_key, hi_key = prefix + "q01", prefix + "q99"
        else:
            lo_text, hi_text = self.mode.split("/", 1)
            ref = torch.as_tensor(stats[prefix + "min"], dtype=torch.float32)
            lo = torch.full_like(ref, float(lo_text))
            hi = torch.full_like(ref, float(hi_text))
            return self._range_params(lo, hi, group=group, kind=kind)
        lo = torch.as_tensor(stats[lo_key], dtype=torch.float32)
        hi = torch.as_tensor(stats[hi_key], dtype=torch.float32)
        return self._range_params(lo, hi, group=group, kind=kind)

    def _range_params(
        self,
        lo: torch.Tensor,
        hi: torch.Tensor,
        *,
        group: str,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = hi - lo
        ignore = width.abs() < 1e-4
        raw_rules = self.minimum_widths.get(kind, {}).get(group)
        rules = raw_rules if isinstance(raw_rules, list) else [raw_rules]
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            requested = torch.zeros_like(width)
            requested[torch.as_tensor(rule["dims"], dtype=torch.long)] = float(rule["width"])
            expanded_width = torch.maximum(width, requested)
            expand = (expanded_width > width) & ~ignore
            center = (lo + hi) * 0.5
            lo = torch.where(expand, center - expanded_width * 0.5, lo)
            width = expanded_width
        width = torch.where(ignore, torch.ones_like(width) * 2.0, width)
        scale = 2.0 / width
        offset = -1.0 - scale * lo
        offset = torch.where(ignore, -lo, offset)
        return scale, offset

    def _get_params(self, *, group: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.groups.get(group) or self.groups.get("default")
        if params is None:
            raise KeyError(f"No pretrain stats group {group!r}; available={sorted(self.groups)}")
        return params[kind]

    def forward(self, value: torch.Tensor, *, group: str, kind: str) -> torch.Tensor:
        scale, offset = self._get_params(group=group, kind=kind)
        scale = scale.to(device=value.device, dtype=value.dtype)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return torch.clamp(value * scale + offset, -5.0, 5.0)

    def inverse(self, value: torch.Tensor, *, group: str, kind: str) -> torch.Tensor:
        scale, offset = self._get_params(group=group, kind=kind)
        scale = scale.to(device=value.device, dtype=value.dtype).clamp_min(1e-8)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return (value - offset) / scale
