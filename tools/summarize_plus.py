"""Read-only, identity-checked summary of one or more distributed Plus outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(directories):
    rows, launches, planned, exits = {}, [], set(), []
    for directory in directories:
        launch = json.loads((directory/'launch.json').read_text())
        # Older receipts did not record this. Never mix unknown and explicit
        # backends, or the failed diagnostic subprocess path with inprocess.
        launch.setdefault('env_backend', 'legacy_unspecified')
        launch.setdefault('render_backend', 'legacy_unspecified')
        launches.append(launch)
        exit_path = directory/'exits.json'
        exits.append(json.loads(exit_path.read_text()) if exit_path.is_file() else None)
        plan = json.loads((directory/'plan.json').read_text())
        keys = {(r['suite'], r['task_id']) for r in plan}
        if len(keys) != len(plan) or keys & planned:
            raise ValueError('Duplicate task assignments across evaluation shards')
        planned |= keys
        for path in (directory/'results').glob('*/*.json'):
            row = json.loads(path.read_text())
            key = row['suite'], row['task_id']
            if key not in keys or key in rows:
                raise ValueError('Foreign or duplicate result')
            if row.get('env_backend', 'legacy_unspecified') != launch['env_backend']:
                raise ValueError('Result env_backend differs from its launch receipt')
            if row.get('render_backend', 'legacy_unspecified') != launch['render_backend']:
                raise ValueError('Result render_backend differs from its launch receipt')
            rows[key] = row
    for name in ('checkpoint_sha256', 'selection_sha256', 'global_expected', 'shards', 'env_backend', 'render_backend'):
        if len({row[name] for row in launches}) != 1:
            raise ValueError(f'Cannot mix evaluations with different {name}')
    groups = {}
    for name in ('suite', 'category'):
        grouped = defaultdict(list)
        for row in rows.values():
            grouped[row[name]].append(row)
        groups[name] = {k: metrics(v) for k,v in sorted(grouped.items())}
    process_failures = sum(code != 0 for codes in exits if codes is not None for code in codes)
    complete = all((p/'COMPLETE.json').is_file() for p in directories)
    return {**metrics(list(rows.values())), 'expected': launches[0]['global_expected'],
        'assigned': len(planned), 'all_tasks_completed': len(rows) == launches[0]['global_expected'],
        'completion_markers_present': complete,
        'worker_returncodes': exits, 'process_failures': process_failures,
        'env_backend': launches[0]['env_backend'],
        'render_backend': launches[0]['render_backend'],
        'technically_complete': complete and all(codes is not None for codes in exits)
            and process_failures == 0 and not any(r.get('error') for r in rows.values())
            and len(rows) == launches[0]['global_expected'],
        **groups}


def metrics(rows):
    success = sum(row['successes'] for row in rows)
    return {'completed': len(rows), 'successes': success,
        'errors': sum(bool(row.get('error')) for row in rows),
        'success_rate': success/len(rows) if rows else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outputs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps(summarize(args.outputs), indent=2, ensure_ascii=False))
