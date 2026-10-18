"""Small public inference assets, independent of training episode/cache files."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACKAGE = ROOT / 'assets/robocasa'
DEFAULT_SEEDS = DEFAULT_PACKAGE / 'scene_seed_list.json'
DEFAULT_TASKS = DEFAULT_PACKAGE / 'tasks.json'
TEXT_CONTRACT = {
    'mode': 'online_t5',
    'instruction_source': 'observation:annotation.human.coarse_action',
    'prompt_transform': 'robocasa_delta.prompt_for',
    'context_len': 128, 'context_dim': 4096, 'dtype': 'bfloat16',
    'padding': 'zero_context_full_attention',
}


def load_inference_package(root=DEFAULT_PACKAGE, *, allow_smoke=False):
    """Validate deployment semantics, never open training metadata or embeddings.

    Old training manifests remain readable for compatibility. Their episode
    bookkeeping is not an inference dependency; training's load_package is
    deliberately separate and still validates all dataset references.
    """
    from wam.datasets.robocasa_posttrain import package_space
    package = json.loads((Path(root) / 'package.json').read_text())
    schema = package.get('schema')
    if schema == 'phoenix.robocasa.inference.v1':
        if package.get('purpose') != 'inference' or package.get('text') != TEXT_CONTRACT:
            raise ValueError('Invalid RoboCasa inference/text contract')
    elif schema in ('wam.robocasa.posttrain.v1', 'wam.robocasa.posttrain.v2'):
        if package.get('purpose') != 'training' and not (allow_smoke and package.get('purpose') == 'smoke'):
            raise ValueError('Smoke package requires explicit allow_smoke')
        if schema.endswith('.v2') and package.get('prompt_source') != 'episodes.jsonl:remarks':
            raise ValueError('Missing episode prompt provenance')
    else:
        raise ValueError('Unknown RoboCasa inference package schema')
    package_space(package)  # Exact action/frame/controller/embodiment contract.
    return package


def load_tasks(path=DEFAULT_TASKS):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get('schema') != 'phoenix.robocasa.tasks.v1'
            or manifest.get('environment_template') != 'gr1_unified/{task}_GR1ArmsAndWaistFourierHands_Env'
            or manifest.get('instruction_source') != TEXT_CONTRACT['instruction_source']):
        raise ValueError('Invalid RoboCasa task manifest')
    tasks = manifest.get('tasks')
    if (not isinstance(tasks, list) or len(tasks) != 24
            or any(not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_]+', task) for task in tasks)
            or len(set(tasks)) != 24):
        raise ValueError('Expected 24 unique RoboCasa task names')
    return tasks
