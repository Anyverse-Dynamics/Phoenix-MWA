"""Read-only inventory of the registered frozen episodes, not directory totals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


def inspect(directory):
    spec = json.loads((directory/'SPEC.json').read_text())
    tasks = spec['tasks']
    if len(tasks) != len(set(tasks)): raise ValueError('Duplicate tasks')
    def check(task):
        ledger = json.loads((directory/task/'CANDIDATES.json').read_text())
        if ledger['task'] != task: raise ValueError(task)
        for index in range(spec['episodes_per_task']):
            selected = ledger['selected'][str(index)]
            episode = directory/task/f'episode_{index:02d}'
            scene = json.loads((episode/'scene.json').read_text())
            for key, expected in dict(task=task, index=index, seed=selected['seed'],
                                      instruction=selected['instruction']).items():
                if scene[key] != expected: raise ValueError(f'{task}/{index}: {key}')
            if not (episode/'observation.npz').is_file(): raise FileNotFoundError(episode)
        return task
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(check, tasks))
    return dict(task_config=spec['task_config'], tasks=len(tasks),
        episodes_per_task=spec['episodes_per_task'], registered_episodes=len(tasks)*spec['episodes_per_task'],
        instructions=spec['instruction_type'], file_hash_validation=False, passed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = inspect(args.directory)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
