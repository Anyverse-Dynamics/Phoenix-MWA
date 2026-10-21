#!/usr/bin/env python
"""All source rows, paired action_t/state_t; parquet only, never video."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pyarrow.parquet as pq
import torch
from wam.datasets.lerobot.transforms.libero_flu_bi import LiberoRight80FLUBIdentityTransform, LIBERO_FLU_BI_CONTRACT


def fields(x):
    x = torch.cat(x).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite source data")
    values = {"min": x.amin(0), "max": x.amax(0), "q01": torch.quantile(x, .01, dim=0),
              "q99": torch.quantile(x, .99, dim=0), "mean": x.mean(0), "std": x.std(0, unbiased=False)}
    return {**{"global_" + k: v.tolist() for k, v in values.items()},
            **{"stepwise_" + k: v[None].tolist() for k, v in values.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing statistics")
    names = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    roots = [args.data_root / (name + "_no_noops_lerobot") for name in names]
    columns = {"action": "action", "joint": "observation.states.joint_state", "ee": "observation.states.ee_state", "gripper": "observation.states.gripper_state"}
    accum = {"action": {"default": []}, "state": {k: [] for k in ("joint", "ee", "gripper")}}
    count = episodes = 0
    inputs = []
    for root in roots:
        info = json.loads((root / "meta/info.json").read_text())
        files = sorted((root / "data").rglob("*.parquet"))
        rows = 0
        for path in files:
            table = pq.read_table(path, columns=list(columns.values()))
            data = {key: torch.from_numpy(np.asarray(table[col].to_pylist(), dtype=np.float32)) for key, col in columns.items()}
            batch = {"action": {"default": data.pop("action")}, "state": data}
            batch = LiberoRight80FLUBIdentityTransform().forward(batch)
            for kind in accum:
                for key in accum[kind]:
                    accum[kind][key].append(batch[kind][key])
            rows += len(table)
            inputs.append({"root": str(root), "file": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(table)})
        if rows != info["total_frames"] or len(files) != info["total_episodes"]:
            raise ValueError(f"Incomplete parquet source: {root}, rows={rows}, files={len(files)}")
        count += rows
        episodes += len(files)
        print(f"PARQUET_ONLY {root.name}: {rows} rows", flush=True)
    stats = {kind: {key: fields(values) for key, values in group.items()} for kind, group in accum.items()}
    stats.update(num_episodes=episodes, num_transition=count, frame_contract=LIBERO_FLU_BI_CONTRACT,
                 stats_scope="all_source_rows_global_minmax_no_stepwise", source_roots=list(map(str, roots)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    args.output.with_suffix(".provenance.json").write_text(json.dumps({"files": inputs, "video_decoding": False, "stats_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(f"STATS_OK episodes={episodes} rows={count} output={args.output}")


if __name__ == "__main__":
    main()
