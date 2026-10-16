import json
from types import SimpleNamespace

from eval.libero_plus import runner


def test_smoke_audits_only_first_task_per_suite(tmp_path, monkeypatch):
    suites = ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10')
    classification = {s: [dict(id=i + 1, name=f'{s}_{i}', category='test',
                                difficulty_level=0) for i in range(2)] for s in suites}
    path = tmp_path / 'libero/libero/benchmark/task_classification.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(classification))
    monkeypatch.setattr(runner, 'PLUS', tmp_path)
    monkeypatch.setattr(runner, 'benchmark', lambda: {
        s: SimpleNamespace(n_tasks=2, get_task=lambda i, s=s: SimpleNamespace(name=f'{s}_{i}'))
        for s in suites})
    checked = []
    monkeypatch.setattr(runner, 'initial_states', lambda suite, i: checked.append(i) or [0])
    monkeypatch.setattr(runner, 'read_instruction', lambda task: ('pick object', 'task.bddl'))
    monkeypatch.setattr(runner, 'plan_path', lambda: tmp_path / 'plan.json')
    rows = runner.audit_full(tmp_path, smoke=True)
    assert [row['suite'] for row in rows] == list(suites)
    assert [row['task_id'] for row in rows] == [0] * 4
    assert checked == [0] * 4
    assert json.loads((tmp_path / 'instruction_audit.json').read_text())['total'] == 4
