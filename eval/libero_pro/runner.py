"""LIBERO-Pro adapter using the shared Phoenix inference and worker lifecycle.

Only registry, plan, instruction and initialization selection differ from Plus.
All adaptation is process-local. Policy, normalization, EGL rendering, seed 42,
30 settling steps, chunk 32/replan 10 and per-step action decoding are shared.
"""
from collections import Counter
from functools import lru_cache
import contextlib
import io
import json
import os
from pathlib import Path

from eval.libero_plus import runner as base
from eval.libero_pro.init_states import safe_init_states

BASES = ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10')
CATEGORIES = {'object': 'Object', 'swap': 'Position', 'lan': 'Semantic', 'task': 'Task'}
SETTINGS = tuple(f'{suite}_{suffix}' for suite in BASES for suffix in CATEGORIES)


def resource_root():
    return Path(os.environ['LIBERO_BENCHMARK_ROOT'])


@lru_cache(maxsize=160)
def load_states(path):
    return safe_init_states(path)


class ProSuite:
    def __init__(self, suite):
        self.suite = suite

    def get_task(self, index):
        if not 0 <= index < 500:
            raise IndexError(index)
        return self.suite.get_task(index // 50)

    def get_task_init_states(self, index):
        task = self.get_task(index)
        path = resource_root() / 'init_files' / task.problem_folder / task.init_states_file
        return [load_states(path)[index % 50].copy()]


def benchmark():
    from eval.libero.bootstrap import setup_libero_paths
    setup_libero_paths()
    from libero.libero import benchmark as registry
    with contextlib.redirect_stdout(io.StringIO()):
        return {name: ProSuite(registry.get_benchmark(name)(task_order_index=0)) for name in SETTINGS}


def instruction(task):
    from libero.libero.envs.bddl_utils import get_problem_info
    path = resource_root() / 'bddl_files' / task.problem_folder / task.bddl_file
    text = ' '.join(get_problem_info(str(path))['language_instruction'].split())
    if not text or '.bddl' in text or '_view_' in text:
        raise ValueError(f'Invalid BDDL semantic instruction: {path}')
    return text, path


def plan_path():
    return Path(os.environ['EVAL_OUTPUT']) / 'audited_pro_plan.json'


def audit(output, *, smoke=False):
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    suites, rows = benchmark(), []
    for suite in BASES:
        for suffix, category in CATEGORIES.items():
            name = f'{suite}_{suffix}'
            if suites[name].suite.n_tasks != 10:
                raise ValueError(f'Expected 10 official tasks in {name}')
            for task_id in range(1 if smoke else 10):
                task = suites[name].get_task(task_id * 50)
                text, bddl = instruction(task)
                for trial in range(1 if smoke else 50):
                    index = task_id * 50 + trial
                    suites[name].get_task_init_states(index)
                    rows.append(dict(suite=name, task_id=index, official_task_id=task_id,
                        trial=trial, task_name=task.name, category=category, base_suite=suite,
                        instruction=text, prompt=DEFAULT_PROMPT.format(task=text), bddl_file=str(bddl)))
    expected = 16 if smoke else 8000
    if len(rows) != expected or len({(r['suite'], r['task_id']) for r in rows}) != expected:
        raise ValueError('Invalid Pro panel identity/count')
    base.dump(plan_path(), rows)
    base.dump(output / 'instruction_audit.json', dict(passed=True, panel='LIBERO-Pro four-case',
        phase='smoke' if smoke else 'full', total_episodes=len(rows),
        official_tasks=16 if smoke else 160, trials_per_task=1 if smoke else 50,
        categories=dict(Counter(r['category'] for r in rows)),
        instruction_source='BDDL :language', initialization='official task state[trial]',
        excluded_category='Environment', seed=42, settle_steps=30))
    print(f'PRO_PANEL_AUDIT_OK episodes={len(rows)} smoke={smoke}', flush=True)
    return rows


original_summarize = base.summarize


def summarize(output):
    result = original_summarize(output)
    rows = [json.loads(p.read_text()) for p in (output / 'results').glob('*/*.json')]
    identities = {(r['suite'], r['task_id']) for r in rows}
    full = (identities == {(s, i) for s in SETTINGS for i in range(500)}
            and not result['errors'])
    base.dump(output / 'summary_gam.json', dict(panel='LIBERO-Pro four-case',
        full_panel_complete=full, expected_full_episodes=8000,
        avg_16_settings=result['success_rate'] if full else None,
        **result))
    return result


def main():
    # Configure only the benchmark-facing hooks; reuse shared tested inference.
    os.environ['WAM_LIBERO_REPO'] = 'LIBERO-PRO'
    os.environ['EVAL_SELECTION'] = 'full'
    if os.environ.get('EVAL_ENV_BACKEND', 'inprocess') != 'inprocess':
        raise ValueError('LIBERO-Pro currently requires inprocess rendering')
    base.RUNNER_MODULE = 'eval.libero_pro.runner'
    base.SMOKE_SUITES = SETTINGS
    base.benchmark, base.read_instruction = benchmark, instruction
    base.plan_path, base.audit_full, base.audit = plan_path, audit, audit
    base.summarize = summarize
    base.main()


if __name__ == '__main__':
    main()
