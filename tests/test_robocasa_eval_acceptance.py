from copy import deepcopy
import pytest
from tools.require_robocasa_acceptance import validate_acceptance, SEED_SHA256


def report():
    return {'checkpoint': '/test/step_045000.pt', 'passed': False, 'inference_passed': True,
            'samples': [{'finite_actions': True, 'checks': {
                'vae_clip_prefix_vs_single': {'passed': True},
                'stvae_clip_prefix_vs_single': {'passed': True},
                'stvae_cache_vs_online_clip': {'passed': False}}}]}


def test_default_rejects_parity_failure():
    with pytest.raises(ValueError):
        validate_acceptance(report(), '/test/step_045000.pt', SEED_SHA256)


def test_explicit_eval_waiver_is_audited_without_changing_report():
    original = report()
    before = deepcopy(original)
    audit = validate_acceptance(original, '/test/step_045000.pt', SEED_SHA256,
                                waive_parity=True, reason='User explicitly requested this evaluation')
    assert audit['evaluation_authorized']
    assert audit['cross_platform_parity_waived']
    assert audit['numerical_parity_passed'] is False
    assert audit['training_acceptance_changed'] is False
    assert original == before


@pytest.mark.parametrize('issue', ['inference', 'checkpoint', 'seed', 'reason', 'sample', 'current'])
def test_waiver_is_limited_to_cross_platform_cache_parity(issue):
    r, ck, seed, reason = report(), '/test/step_045000.pt', SEED_SHA256, 'approved'
    if issue == 'inference': r['inference_passed'] = False
    if issue == 'checkpoint': ck = '/test/step_060000.pt'
    if issue == 'seed': seed = '0' * 64
    if issue == 'reason': reason = ''
    if issue == 'sample': r['samples'] = []
    if issue == 'current': r['samples'][0]['checks']['stvae_clip_prefix_vs_single']['passed'] = False
    with pytest.raises(ValueError):
        validate_acceptance(r, ck, seed, waive_parity=True, reason=reason)


def test_passed_report_does_not_need_waiver():
    r = report()
    r['passed'] = True
    audit = validate_acceptance(r, '/test/step_045000.pt', SEED_SHA256)
    assert audit['evaluation_mode'] == 'accepted_port'
    assert not audit['cross_platform_parity_waived']
