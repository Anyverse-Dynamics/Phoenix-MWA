import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile
import numpy as np
import pytest
from eval.libero_pro import runner
from eval.libero_pro.init_states import safe_init_states


def test_safe_numeric_initial_states(tmp_path):
    values = np.arange(50 * 80, dtype=np.float64).reshape(50, 80)
    path = tmp_path / 'states.pruned_init'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', pickle.dumps(values, protocol=4))
        archive.writestr('archive/version', '3\n')
    np.testing.assert_array_equal(safe_init_states(path), values)


def test_rejects_non_numeric_pickle(tmp_path):
    path = tmp_path / 'states.pruned_init'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', pickle.dumps({'not': 'numeric'}, protocol=4))
        archive.writestr('archive/version', '3\n')
    with pytest.raises(ValueError):
        safe_init_states(path)


def test_trial_index_is_not_repeated_first_state(monkeypatch):
    monkeypatch.setenv('LIBERO_BENCHMARK_ROOT', '/resources')
    values = np.arange(50 * 3).reshape(50, 3)
    monkeypatch.setattr(runner, 'load_states', lambda path: values)
    task_ids = []
    def task(index):
        task_ids.append(index)
        return SimpleNamespace(problem_folder='suite', init_states_file='init')
    suite = runner.ProSuite(SimpleNamespace(get_task=task))
    np.testing.assert_array_equal(suite.get_task_init_states(149)[0], values[49])
    np.testing.assert_array_equal(suite.get_task_init_states(150)[0], values[0])
    assert task_ids == [2, 3]
    with pytest.raises(IndexError):
        suite.get_task(500)


@pytest.mark.parametrize('smoke,expected', [(True, 16), (False, 8000)])
def test_official_panel_plan(tmp_path, monkeypatch, smoke, expected):
    suites = {name: runner.ProSuite(SimpleNamespace(n_tasks=10,
        get_task=lambda i: SimpleNamespace(name=f'task_{i}'))) for name in runner.SETTINGS}
    monkeypatch.setattr(runner.ProSuite, 'get_task_init_states', lambda self, i: [i % 50])
    monkeypatch.setattr(runner, 'benchmark', lambda: suites)
    monkeypatch.setattr(runner, 'instruction', lambda task: ('semantic BDDL text', Path('task.bddl')))
    monkeypatch.setenv('EVAL_OUTPUT', str(tmp_path))
    rows = runner.audit(tmp_path, smoke=smoke)
    assert len(rows) == expected
    assert {r['suite'] for r in rows} == set(runner.SETTINGS)
    assert all(r['task_id'] == r['official_task_id'] * 50 + r['trial'] for r in rows)
    assert len({(r['suite'], r['task_id']) for r in rows}) == expected
    assert json.loads((tmp_path / 'instruction_audit.json').read_text())['excluded_category'] == 'Environment'


def test_incomplete_panel_has_no_full_average(tmp_path):
    (tmp_path / 'results' / runner.SETTINGS[0]).mkdir(parents=True)
    row = dict(suite=runner.SETTINGS[0], task_id=0, category='Object', successes=1)
    (tmp_path / 'results' / runner.SETTINGS[0] / 'task_00000.json').write_text(json.dumps(row))
    (tmp_path / 'plan.json').write_text(json.dumps([row]))
    runner.summarize(tmp_path)
    report = json.loads((tmp_path / 'summary_gam.json').read_text())
    assert report['avg_16_settings'] is None
    assert not report['full_panel_complete']
