import hashlib
import json
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from wam.posttrain.robocasa_paths import resolve_source, require_migrated_cache
from wam.datasets.robocasa_visual_cache import file_identity, file_hash
from wam.posttrain.guard import validate_scope
from wam.training.config_validation import validate_training_config
from wam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]


def recipe(monkeypatch):
    for key, val in {'WAM_PRETRAIN_CHECKPOINT': '/test/resources/init110k.pt',
                     'EXPECTED_PRETRAIN_STEP': '110000',
                     'STVAE_CHECKPOINT': '/test/resources/stvae.pt',
                     'DINO_ENCODER_CHECKPOINT': '/test/resources/dino.pt',
                     'ROBOCASA_PACKAGE': '/test/resources/package',
                     'ROBOCASA_CACHE': '/test/resources/cache',
                     'ROBOCASA_REGISTRY': str(ROOT / 'configs/embodiment_registry/robocasa_gr1_delta80_slot18_v1.json')}.items():
        monkeypatch.setenv(key, val)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        return compose(config_name='train', overrides=['task=robocasa_posttrain'])


def test_delta80_recipe_is_opt_in_and_exact(monkeypatch):
    cfg = recipe(monkeypatch)
    validate_scope(cfg)
    validate_training_config(cfg)
    assert not cfg.cache_locality_sampling.enabled
    assert (cfg.model.action_dit_config.action_dim, cfg.model.proprio_dim) == (80,80)
    assert cfg.local_robot_softprompt.registry_path.endswith('robocasa_gr1_delta80_slot18_v1.json')
    assert cfg.model.joint_visual_codec.expected_stvae_step == 150000


@pytest.mark.parametrize('bad', ['dimension', 'step', 'domain', 'source', 'smoke', 'sdpa'])
def test_rejects_scope_or_algorithm_drift(monkeypatch, bad):
    cfg = recipe(monkeypatch)
    if bad == 'dimension': cfg.model.action_dit_config.action_dim = 47
    if bad == 'step': cfg.joint96_posttrain.expected_pretrain_step = -1
    if bad == 'domain': cfg.data.val.domain = 'libero'
    if bad == 'source': cfg.model.joint_visual_codec.source_root = '/some/external/checkout'
    if bad == 'smoke': cfg.data.train.allow_smoke = True
    if bad == 'sdpa': cfg.model.mot_attention_backend = 'flex'
    with pytest.raises(ValueError):
        validate_scope(cfg)
        validate_training_config(cfg)


def test_relocation_preserves_payload_identity_and_original_manifest(tmp_path, monkeypatch):
    destination = tmp_path/'data'
    destination.mkdir()
    file = destination/'one.bin'
    file.write_bytes(b'untouched source payload')
    original = Path('/old/ppu/dataset')
    mapping = tmp_path/'paths.json'
    mapping.write_text(json.dumps({'schema': 'phoenix.robocasa.path_map.v1',
                                   'mappings': {str(original): str(destination)}}))
    before = mapping.read_bytes()
    monkeypatch.setenv('ROBOCASA_PATH_MAP', str(mapping))
    assert resolve_source(original/'one.bin') == file
    assert file_identity(original/'one.bin') == file_identity(file)
    assert file_hash(original/'one.bin') is None
    with pytest.raises(FileNotFoundError): file_hash(original/'missing.bin')
    assert mapping.read_bytes() == before
    assert resolve_source(Path('/old/ppu/dataset_other/one.bin')) == Path('/old/ppu/dataset_other/one.bin')
    with pytest.raises(ValueError): resolve_source(original/'..'/'outside')
    with pytest.raises(FileNotFoundError): file_identity(original/'missing.bin')


def test_formal_cache_checks_files_without_transfer_receipts(tmp_path, monkeypatch):
    monkeypatch.delenv('ROBOCASA_TRANSFER_RECEIPT', raising=False)
    with pytest.raises(FileNotFoundError): require_migrated_cache(tmp_path/'cache', ['task'])
    (tmp_path/'cache/episodes/task').mkdir(parents=True)
    require_migrated_cache(tmp_path/'cache', ['task'])
    with pytest.raises(FileNotFoundError): require_migrated_cache(tmp_path/'foreign', ['task'])
