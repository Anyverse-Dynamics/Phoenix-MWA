from types import SimpleNamespace
import sys

import pytest
import torch
from omegaconf import OmegaConf

from wam.inference import robocasa_joint96 as inference


@pytest.mark.parametrize('fail', [False, True])
def test_checkpoint_construction_skips_only_dit_resource_and_restores_loader(monkeypatch, fail):
    calls = []

    def original_loader(**kwargs):
        calls.append(kwargs)
        return 'components'

    factory = SimpleNamespace(load_wan22_ti2v_5b_components=original_loader)
    monkeypatch.setitem(sys.modules, 'wam.model.factory', factory)
    import wam.model
    monkeypatch.setattr(wam.model, 'factory', factory, raising=False)
    config = {'skip_dit_load_from_pretrain': False, 'preserved_parameter': 6}
    model = object()

    def instantiate(config_arg, **kwargs):
        assert config_arg is config and config_arg['skip_dit_load_from_pretrain'] is False
        assert kwargs == {'model_dtype': torch.bfloat16, 'device': 'cpu'}
        assert factory.load_wan22_ti2v_5b_components(skip_dit_load_from_pretrain=False,
            load_text_encoder=False, expand_wan48_to_joint96=True) == 'components'
        if fail:
            raise RuntimeError('constructor failed')
        return model

    monkeypatch.setattr(inference, 'instantiate', instantiate)
    if fail:
        with pytest.raises(RuntimeError, match='constructor failed'):
            inference.instantiate_checkpoint_model(config, 'cpu')
    else:
        assert inference.instantiate_checkpoint_model(config, 'cpu') is model
    assert calls == [{'skip_dit_load_from_pretrain': True, 'load_text_encoder': False,
                      'expand_wan48_to_joint96': True}]
    assert factory.load_wan22_ti2v_5b_components is original_loader


@pytest.mark.parametrize('fail_restore', [False, True])
def test_policy_restores_strict_checkpoint_before_becoming_usable(monkeypatch, tmp_path, fail_restore):
    from wam.posttrain import guard
    from wam.training import robocasa_posttrain
    cfg = OmegaConf.create({'joint96_posttrain': {'expected_pretrain_step': 110000,
        'robocasa': {'frame_contract': 'test'}}, 'model': {'unchanged': True}})
    monkeypatch.setattr(inference.OmegaConf, 'load', lambda path: cfg)
    monkeypatch.setattr(guard, 'install_import_guard', lambda: None)
    monkeypatch.setattr(guard, 'validate_scope', lambda config: None)
    monkeypatch.setattr(robocasa_posttrain, 'validate_robocasa_config', lambda config: None)
    monkeypatch.setattr(inference, 'load_package', lambda *a, **kw: {'action_space': 'delta80'})
    space = SimpleNamespace(CONTRACT={'version': 'test'}, RoboCasaNormalizer=lambda value: 'normalizer')
    monkeypatch.setattr(inference, 'package_space', lambda package: space)
    monkeypatch.setattr(inference.torch, 'load', lambda *a, **kw: {'step': 45000})
    monkeypatch.setattr(inference, 'validate_checkpoint', lambda *a, **kw: None)
    events = []
    model = SimpleNamespace(set_runtime_device=lambda device: events.append(('device', device)),
                            eval=lambda: events.append(('eval',)))
    monkeypatch.setattr(inference, 'instantiate_checkpoint_model', lambda config, device: model)
    checkpoint = tmp_path / 'policy.pt'

    def strict_restore(actual_model, path, *, expected_step):
        assert actual_model is model and path == str(checkpoint) and expected_step == 45000
        assert events == []
        if fail_restore:
            raise ValueError('strict restore rejected mismatched weights')
        events.append(('strict_restore', expected_step))

    monkeypatch.setattr(inference, 'load_joint96_initialization', strict_restore)
    monkeypatch.setattr(inference, 'RoboCasaTextCache', lambda *args: 'text cache')
    monkeypatch.setattr(inference, 'RoboCasaInferenceText', lambda *args: 'text')
    monkeypatch.setattr(inference, 'read_json', lambda path: {})

    class CapturedPolicy(inference.RoboCasaJoint96Policy):
        def __init__(self, loaded_model, normalizer, text, **kwargs):
            assert loaded_model is model
            events.append(('usable',))

    if fail_restore:
        with pytest.raises(ValueError, match='strict restore rejected'):
            CapturedPolicy.from_checkpoint(checkpoint, 'config.yaml', tmp_path, device='cpu')
        assert events == []
    else:
        CapturedPolicy.from_checkpoint(checkpoint, 'config.yaml', tmp_path, device='cpu')
        assert events == [('strict_restore', 45000), ('device', 'cpu'), ('eval',), ('usable',)]
