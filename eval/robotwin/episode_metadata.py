"""Read official scene descriptions without invoking planners or expert motion."""
import ast
import copy
import inspect
import sys
from pathlib import Path


READ_CALLS = {'ArmTag', 'str', 'get_face_prod', 'get_pose', 'get_functional_point',
              'to_transformation_matrix'}


def _readonly(expression):
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
            if name not in READ_CALLS:
                raise ValueError('Non-readonly instruction metadata expression: ' + name)


def scene_metadata(task, env):
    module = sys.modules[type(env).__module__]
    tree = ast.parse(Path(inspect.getfile(type(env))).read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == task)
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'play_once')
    assignments = [n for n in method.body if isinstance(n, ast.Assign)]
    info_nodes = [n for n in assignments if any(ast.unparse(t) in ("self.info['info']", 'self.info["info"]') for t in n.targets)]
    if not info_nodes:
        assert task == 'handover_block'
        return {}
    assert len(info_nodes) == 1 and isinstance(info_nodes[0].value, ast.Dict)
    definitions = {t.id: n.value for n in assignments for t in n.targets if isinstance(t, ast.Name)}
    local = {'self': env}
    ArmTag = module.ArmTag
    if task in ('blocks_ranking_rgb', 'blocks_ranking_size', 'stack_blocks_three', 'stack_blocks_two'):
        count = 2 if task == 'stack_blocks_two' else 3
        for i in range(1, count + 1):
            actor = getattr(env, f'block{i}')
            local[f'arm_tag{i}'] = str(ArmTag('left' if actor.get_pose().p[0] < 0 else 'right'))
    if task == 'stack_bowls_two':
        for i in (1, 2):
            local[f'arm_tag{i}'] = ArmTag('left' if getattr(env, f'bowl{i}').get_pose().p[0] < 0 else 'right')
    if task == 'place_bread_basket':
        assert len(env.bread) in (1, 2), 'Unsupported official bread count'
        x = [actor.get_pose().p[0] for actor in env.bread]
        local['arm_info'] = ('left' if x[0] < 0 else 'right') if len(x) == 1 or x[0] * x[1] > 0 else 'dual'

    visiting = set()
    def evaluate(expression):
        _readonly(expression)
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name not in local and name in definitions:
                    assert name not in visiting, 'Metadata dependency cycle'
                    visiting.add(name)
                    local[name] = evaluate(definitions[name])
                    visiting.remove(name)
        return eval(compile(ast.Expression(copy.deepcopy(expression)), '<official-readonly-metadata>', 'eval'), vars(module), local)

    # Official check_success relies on these attributes normally established by
    # expert play_once and retained across the second setup_demo. Set only the
    # same read-only initial-scene fields, without calling any motion method.
    if task in ('open_laptop', 'put_object_cabinet'):
        env.arm_tag = evaluate(definitions['arm_tag'])
    if task == 'put_object_cabinet':
        env.origin_z = float(env.object.get_pose().p[2])
    if task == 'place_object_scale':
        env.arm_tag = ArmTag('right' if env.object.get_pose().p[0] > 0 else 'left')
    result = evaluate(info_nodes[0].value)
    if task == 'place_bread_basket' and len(env.bread) == 2:
        result['{C}'] = f'075_bread/base{env.bread_id[1]}'
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.items())
    return result


def validate_sources(robotwin, tasks):
    """Fail before launch if metadata requires a new unreviewed motion call."""
    special = {'blocks_ranking_rgb', 'blocks_ranking_size', 'stack_blocks_three',
               'stack_blocks_two', 'stack_bowls_two', 'place_bread_basket'}
    for task in tasks:
        tree = ast.parse((Path(robotwin) / 'envs' / f'{task}.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'play_once')
        nodes = [n for n in method.body if isinstance(n, ast.Assign)]
        info = [n for n in nodes if any(ast.unparse(t) == "self.info['info']" for t in n.targets)]
        if not info:
            assert task == 'handover_block'
            continue
        definitions = {t.id: n.value for n in nodes for t in n.targets if isinstance(t, ast.Name)}
        visited = set()
        def walk(expression):
            _readonly(expression)
            for node in ast.walk(expression):
                if isinstance(node, ast.Name) and node.id in definitions and node.id not in visited:
                    if task in special and node.id in ('arm_info', 'arm_tag1', 'arm_tag2', 'arm_tag3'):
                        continue
                    visited.add(node.id)
                    walk(definitions[node.id])
        walk(info[0].value)
