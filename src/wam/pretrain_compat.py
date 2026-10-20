"""Robot37 presentation aliases; CSVAE configuration fields are never remapped."""
import copy
import os

from omegaconf import OmegaConf


READER_ALIASES = {
    'wam.datasets.local_robot37.Robot37Dataset':
        'wam.datasets.phoenix_robot.PhoenixRobotDataset',
    'wam.datasets.local_robot37_multiview.Robot37MultiViewDataset':
        'wam.datasets.phoenix_pretrain.PhoenixPretrainDataset',
}


def _rename_config_names(value, *, legacy=False):
    source, target = (('phoenix_pretrain', 'robot37') if legacy
                      else ('robot37', 'phoenix_pretrain'))
    if source in value:
        if target in value and value[target] != value[source]:
            raise ValueError('Conflicting Phoenix and legacy pretraining config blocks')
        value[target] = value.pop(source)
    aliases = {v: k for k, v in READER_ALIASES.items()} if legacy else READER_ALIASES
    for split in ('train', 'val'):
        dataset = value.get('data', {}).get(split)
        if isinstance(dataset, dict) and dataset.get('_target_') in aliases:
            dataset['_target_'] = aliases[dataset['_target_']]
    return value


def canonical_pretrain_config(cfg):
    value = (OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg)
             else copy.deepcopy(cfg))
    return OmegaConf.create(_rename_config_names(value))


def checkpoint_recipe_names(recipe):
    # Preserve the Robot37 reader wire names and every other configuration field.
    return _rename_config_names(copy.deepcopy(recipe), legacy=True)


def bounded_smoke_enabled():
    new, old = os.getenv('PHOENIX_BOUNDED_SMOKE'), os.getenv('ROBOT37_BOUNDED_SMOKE')
    if new is not None and old is not None and new != old:
        raise ValueError('Conflicting Phoenix and legacy bounded-smoke flags')
    return (new if new is not None else old) == '1'
