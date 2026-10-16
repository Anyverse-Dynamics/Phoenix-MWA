from pathlib import Path
import sys

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preflight_libero import validate_recipe
from wam.posttrain.guard import validate_scope
from wam.utils.config_resolvers import register_default_resolvers


def recipe(monkeypatch, task, step):
    for key, value in {
        'EXPECTED_PRETRAIN_STEP': str(step), 'WAM_PRETRAIN_CHECKPOINT': '/test/resources/pretrain.pt',
        'STVAE_CHECKPOINT': '/test/resources/stvae.pt', 'DINO_ENCODER_CHECKPOINT': '/test/resources/dino.pt',
        'LIBERO_DATA_ROOT': '/test/resources/libero', 'LIBERO_TEXT_CACHE': '/test/resources/text',
        'LIBERO_WAN_VAE_CACHE': '/test/resources/wan', 'LIBERO_STVAE_CACHE': '/test/resources/stvae',
        'LIBERO_PAIR_BINDING': '/test/resources/pair.json',
        'LIBERO_FLU_BI_NORM_STATS': str(ROOT / 'assets/stats/libero_joint96_flu_b_identity_v1.json'),
        'ROBOCASA_PACKAGE': '/test/resources/package', 'ROBOCASA_CACHE': '/test/resources/cache',
        'ROBOCASA_REGISTRY': str(ROOT / 'configs/embodiment_registry/robocasa_gr1_delta80_slot18_v1.json'),
    }.items():
        monkeypatch.setenv(key, value)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name='train', overrides=[f'task={task}'])


@pytest.mark.parametrize('step', [110000, 125000, 155000, 170000])
def test_public_recipes_accept_explicit_initialization_step(monkeypatch, step):
    libero = recipe(monkeypatch, 'libero_posttrain', step)
    validate_recipe(libero)
    validate_scope(libero)
    robo = recipe(monkeypatch, 'robocasa_posttrain', step)
    validate_scope(robo)
    assert robo.joint96_posttrain.expected_pretrain_step == step


def test_libero_hyperparameters_are_configurable(monkeypatch):
    cfg = recipe(monkeypatch, 'libero_posttrain', 125000)
    cfg.batch_size = 4
    cfg.gradient_accumulation_steps = 2
    cfg.learning_rate = 1e-4
    cfg.max_steps = 25000
    cfg.save_every = 1000
    cfg.num_epochs = 100
    validate_recipe(cfg)


@pytest.mark.parametrize('bad', ['step', 'frame', 'batch', 'transform'])
def test_compatibility_checks_still_reject_invalid_inputs(monkeypatch, bad):
    cfg = recipe(monkeypatch, 'libero_posttrain', 125000)
    if bad == 'step': cfg.joint96_posttrain.expected_pretrain_step = -1
    if bad == 'frame': cfg.joint96_posttrain.libero_frame_contract = 'wrong'
    if bad == 'batch': cfg.batch_size = 0
    if bad == 'transform': cfg.data.train.processor.action_state_transforms[0]._target_ = 'wrong'
    with pytest.raises(ValueError):
        validate_recipe(cfg)


def test_initialization_step_is_not_silently_defaulted(monkeypatch):
    cfg = recipe(monkeypatch, 'libero_posttrain', 125000)
    monkeypatch.delenv('EXPECTED_PRETRAIN_STEP')
    with pytest.raises(Exception, match='EXPECTED_PRETRAIN_STEP'):
        _ = cfg.joint96_posttrain.expected_pretrain_step
