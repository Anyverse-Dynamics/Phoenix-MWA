"""Phoenix-MWA Joint96 FLU/B=I evaluation on LIBERO-Plus."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

RUNNER_MODULE = 'eval.libero_plus.runner'
SMOKE_SUITES = ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10')

def evaluation_checkpoint_path(resource_dir, step=None, filename=None):
    """Allow a release filename while keeping the checkpoint inside its bundle."""
    if filename is None:
        root = Path(resource_dir)
        if (root / 'policy.pt').is_file():
            filename = 'policy.pt'
        else:
            candidates = sorted(root.glob('*.pt'))
            if len(candidates) == 1:
                filename = candidates[0].name
            else:
                raise ValueError('Set JOINT96_EVAL_CHECKPOINT_FILENAME when no unique policy file exists')
    if not filename or filename in ('.', '..') or '\x00' in filename or ('/' in filename) or ('\\' in filename) or (Path(filename).name != filename):
        raise ValueError('JOINT96_EVAL_CHECKPOINT_FILENAME must be a single filename')
    return Path(resource_dir) / filename
ROOT = Path(__file__).resolve().parents[2]
INCOMING = Path(os.environ.get('JOINT96_EVAL_RESOURCE_DIR', '/unset'))
EVAL_STEP = None
PRETRAIN_STEP = None
EVAL_CKPT = None


def read_checkpoint_progress():
    global EVAL_CKPT, EVAL_STEP, PRETRAIN_STEP
    import torch
    from wam.utils.checkpoint_steps import checkpoint_steps
    EVAL_CKPT = evaluation_checkpoint_path(INCOMING, filename=os.environ.get('JOINT96_EVAL_CHECKPOINT_FILENAME'))
    payload = torch.load(EVAL_CKPT, map_location='cpu', mmap=True, weights_only=True)
    progress = checkpoint_steps(payload)
    EVAL_STEP, PRETRAIN_STEP = progress['step'], progress['pretrain_step']
    return progress
PLUS = ROOT / 'third_party/LIBERO-plus'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(PLUS)]
os.environ.setdefault('WAM_LIBERO_REPO_ROOT', str(PLUS))
os.environ.setdefault('MUJOCO_GL', os.environ.get('EVAL_RENDER_BACKEND', 'egl'))
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
MODEL_BASE = Path(os.environ.get('DIFFSYNTH_MODEL_BASE_PATH', '/unset'))
os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'true')

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)

def sha(path):
    Path(path).resolve(strict=True)
    return None

def legacy_simulator_action(native_action):
    """Dataset: 0=closed, 1=open. LIBERO: +1=closed, -1=open.

    Match configs/sim_libero_plus.yaml's binarize_gripper=true. Do not
    transform the six continuous EEF delta dimensions or model history.
    """
    import numpy as np
    action = np.asarray(native_action, dtype=np.float32).copy()
    if action.shape != (32, 7) or not np.isfinite(action).all():
        raise ValueError('Invalid native LIBERO action')
    action[..., -1] = np.sign(1.0 - 2.0 * action[..., -1])
    return action

def simulator_action(canonical):
    """Keep canonical physical actions until each real environment step."""
    import numpy as np
    canonical = np.asarray(canonical, dtype=np.float32)
    if canonical.shape != (32, 7) or not np.isfinite(canonical).all():
        raise ValueError('Invalid AB canonical action')
    return canonical.copy()

def decode_at_execution(canonical, obs):
    from wam.posttrain.eval_contract import decode_at_execution as decode
    return decode(canonical, obs)

def train_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(INCOMING / 'train_config.yaml')

def benchmark():
    from eval.libero.bootstrap import setup_libero_paths
    setup_libero_paths()
    from libero.libero import benchmark as b
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return {s: b.get_benchmark_dict()[s](task_order_index=0) for s in ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10')}

def read_instruction(task):
    """BDDL semantic language is authoritative; NEVER use the raw task stem.

    The benchmark's task.language derives filenames for most non-language
    perturbations and can contain view/texture/light/noise/initstate suffixes.
    BDDL :language retains language perturbations while omitting metadata.
    """
    from libero.libero.envs.bddl_utils import get_problem_info
    from libero.libero import get_libero_path
    path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    semantic_path = path
    if '_view_' in path.name and '_initstate_' in path.name:
        semantic_path = path.with_name(path.name.split('_view_', 1)[0] + '.bddl')
    value = get_problem_info(str(semantic_path))['language_instruction']
    text = ' '.join(value.split())
    if not text or any((x in text.lower() for x in ('initstate', '.bddl', 'sample', 'noise_', '_view_', '_language_'))):
        raise ValueError(f'Invalid semantic instruction in {path}: {text!r}')
    return (text, path)

def plan_path():
    return Path(os.environ['EVAL_OUTPUT']) / 'audited_balanced10_plan.json'

def initial_states(suite, task_id):
    """Keep weights_only=True; allow only NumPy array serialization primitives."""
    import numpy as np
    import torch
    allowed = [np.core.multiarray._reconstruct, np.core.multiarray.scalar, np.ndarray, np.dtype, type(np.dtype('float64')), type(np.dtype('float32')), type(np.dtype('int64'))]
    with torch.serialization.safe_globals(allowed):
        return suite.get_task_init_states(task_id)

def audit(output):
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    source = json.loads((ROOT / 'assets/libero_plus_balanced10_manifest.json').read_text())
    assert source['total_selected'] == len(source['tasks']) == 1003
    assert len({(i['suite'], i['task_id']) for i in source['tasks']}) == 1003
    suites = benchmark()
    rows, changed = ([], 0)
    for item in source['tasks']:
        task = suites[item['suite']].get_task(item['task_id'])
        if task.name != item['task_name']:
            raise ValueError(f'Manifest/registry identity mismatch: {item} -> {task.name}')
        instruction, bddl = read_instruction(task)
        states = initial_states(suites[item['suite']], item['task_id'])
        assert len(states) > 0
        changed += instruction != task.language
        rows.append({**item, 'instruction': instruction, 'registry_language': task.language, 'prompt': DEFAULT_PROMPT.format(task=instruction), 'bddl_file': str(bddl)})
    dump(plan_path(), rows)
    dump(output / 'instruction_audit.json', {'passed': True, 'total': len(rows), 'source': 'actual BDDL :language', 'not_raw_filename': True, 'registry_language_differences': changed, 'language_perturbations_preserved': True, 'categories': dict(Counter((i['category'] for i in rows))), 'suites': dict(Counter((i['suite'] for i in rows))), 'manifest_sha256': sha(ROOT / 'assets/libero_plus_balanced10_manifest.json'), 'examples': rows[:7]})
    print(f'INSTRUCTION_AUDIT_OK tasks={len(rows)} registry_differences={changed}', flush=True)
    return rows

def audit_full(output, *, smoke=False):
    """Full task classification from the same pinned, locally patched Plus benchmark."""
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    source = json.loads((PLUS / 'libero/libero/benchmark/task_classification.json').read_text())
    rows = []
    for suite, tasks in benchmark().items():
        assert len(source[suite]) == tasks.n_tasks
        for item in (source[suite][:1] if smoke else source[suite]):
            index = item['id'] - 1
            task = tasks.get_task(index)
            assert task.name == item['name']
            instruction, bddl = read_instruction(task)
            assert len(initial_states(tasks, index)) > 0
            rows.append({'suite': suite, 'task_id': index, 'task_name': task.name, 'category': item['category'], 'difficulty_level': item['difficulty_level'], 'instruction': instruction, 'prompt': DEFAULT_PROMPT.format(task=instruction), 'bddl_file': str(bddl)})
    assert len(rows) == len({(r['suite'], r['task_id']) for r in rows}) == (4 if smoke else 10030)
    dump(plan_path(), rows)
    dump(output / 'instruction_audit.json', {'passed': True, 'total': len(rows), 'classification_sha256': sha(PLUS / 'libero/libero/benchmark/task_classification.json'), 'suites': dict(Counter((r['suite'] for r in rows))), 'categories': dict(Counter((r['category'] for r in rows)))})
    return rows

def preflight(output, *, smoke=False):
    import torch
    from wam.training.checkpoint import _joint96_identity
    from wam.model.modules.codecs.dino_stvae.provenance import verify_bundle, canonical_source_identity, BUNDLED_SOURCE
    from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
    cfg = train_cfg()
    read_checkpoint_progress()
    import mujoco
    assert mujoco.__version__ == '3.3.2'
    from wand.api import library
    assert library.MagickMotionBlurImage is not None
    expected_sha = None
    from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT
    assert cfg.joint96_posttrain.libero_frame_contract == LIBERO_FLU_BI_CONTRACT['version']
    for split in ('train', 'val'):
        assert cfg.data[split].processor.action_state_transforms[0]._target_.endswith('LiberoRight80FLUBIdentityTransform')
    stats_document = json.loads((INCOMING / 'dataset_stats.json').read_text())
    print('PREFLIGHT_RESOURCES_OK hash_validation=disabled', flush=True)
    p = torch.load(EVAL_CKPT, mmap=True, weights_only=True, map_location='cpu')
    from wam.posttrain.eval_contract import validate_evaluation_contract
    validate_evaluation_contract(cfg, p, INCOMING / 'dataset_stats.json', expected_pretrain_step=PRETRAIN_STEP)
    assert p['mot']['mixtures.action.action_encoder.weight'].shape == (1024, 80)
    assert p['mot']['mixtures.video.patch_embedding.weight'].shape == (3072, 96, 1, 2, 2)
    assert p['joint96_metadata']['action_video_attention'] == JOINT96_ACTION_VIDEO_ATTENTION == 'current_and_noisy_future_v1'
    identity = _joint96_identity(p['joint96_metadata'])
    assert 'current_conditioning' not in identity, 'This release preserves the legacy single-current-frame evaluation'
    assert cfg.joint96_posttrain.libero_softprompt.embodiment_id == 0
    assert cfg.local_robot_softprompt.enabled
    assert p['mot']['mixtures.action.embodiment_prompt.embedding.weight'].shape == (28, 8192)
    contract = p['joint96_metadata']['soft_prompt']['training_contract']
    assert contract['frame_contract'] == LIBERO_FLU_BI_CONTRACT['version']
    assert contract['embodiment_id'] == 0 and contract['domain'] == 'libero'
    assert cfg.model.video_scheduler.infer_shift == 6 and cfg.model.action_scheduler.infer_shift == 1
    assert cfg.data.train.camera_slots == [0, 2] and cfg.data.train.context_len == 128
    assert not cfg.model.memory.enabled and (not cfg.model.future_delta.enabled)
    stats = json.loads((INCOMING / 'dataset_stats.json').read_text())['action']['default']
    assert stats['global_min'][-1] == 0.0 and stats['global_max'][-1] == 1.0
    import numpy as np
    probe = np.zeros((32, 7), dtype=np.float32)
    probe[:3, -1] = [0.0, 1.0, 0.5]
    from wam.datasets.lerobot.transforms.libero_flu_bi import LiberoRight80FLUBIdentityTransform
    converted = LiberoRight80FLUBIdentityTransform().decode_simulator_commands(torch.from_numpy(probe), torch.zeros(32, 6)).numpy()
    assert np.array_equal(converted[:3, -1], [1.0, -1.0, 0.0])
    assert np.array_equal(converted[:, :6], probe[:, :6])
    dump(output / 'preflight.json', {'passed': True, 'checkpoint': str(EVAL_CKPT), 'step': p['step'], 'pretrain_step': PRETRAIN_STEP, 'checkpoint_sha256': expected_sha, 'identity': identity, 'action_video_attention': JOINT96_ACTION_VIDEO_ATTENTION, 'text_template': "A video recorded from a robot's point of view executing the following instruction: {task}", 'text_mask': 'zero padded embeddings; all 128 positions valid', 'camera_slots': [0, 2], 'camera_is_pad': [False, True, False], 'action_valid_dims': [*range(47, 53), 56], 'proprio_valid_dims': list(range(40, 57)), 'video_shift': 6, 'action_shift': 1, 'chunk': 32, 'replan': 10, 'denoise': 10, 'environment': f"MuJoCo {mujoco.__version__} / {os.environ['MUJOCO_GL']}", 'video_write': False, 'gripper': 'native 0=close/1=open -> env 1-2*clip(g,0,1); continuous', 'action_frame': LIBERO_FLU_BI_CONTRACT, 'execution': 'canonical queue -> each-step current source EE -> inverse FLU/B=I -> clipped OSC command'})
    print('JOINT96_PREFLIGHT_OK', flush=True)
    return audit_full(output, smoke=smoke) if os.environ.get('EVAL_SELECTION') == 'full' else audit(output)

def text_cache(output):
    import torch
    from wam.model.backbones.wan22.helpers.loader import _load_registered_model
    from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    rows = json.loads(plan_path().read_text())
    target = Path(os.environ.get('EVAL_TEXT_CACHE', str(output / 'text_cache')))
    readonly = os.environ.get('EVAL_TEXT_CACHE_READONLY') == '1' or target.is_symlink()
    if not readonly:
        target.mkdir(parents=True, exist_ok=True)
    prompts = sorted({i['prompt'] for i in rows})
    missing = [p for p in prompts if not (target / (hashlib.sha256(p.encode()).hexdigest() + '.pt')).is_file()]
    if not missing:
        dump(output / 'text_cache.json', {'unique_prompts': len(prompts), 'template_aligned': True, 'encoded': 0, 'reused': True})
        return
    if readonly:
        raise FileNotFoundError(f'Missing {len(missing)} prompts in read-only text cache: {target}')
    encoder = _load_registered_model(str(MODEL_BASE / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors'), 'wan_video_text_encoder', torch.bfloat16, 'cuda').eval()
    tokenizer = HuggingfaceTokenizer(str(MODEL_BASE / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'), seq_len=128, clean='whitespace')
    with torch.no_grad():
        for start in range(0, len(missing), 4):
            batch = missing[start:start + 4]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            ids, mask = (ids.cuda(), mask.cuda().bool())
            context = encoder(ids, mask).masked_fill(~mask[..., None], 0).cpu()
            for prompt, ctx in zip(batch, context):
                path = target / (hashlib.sha256(prompt.encode()).hexdigest() + '.pt')
                tmp = path.with_suffix('.partial')
                torch.save({'prompt': prompt, 'context': ctx, 'mask': torch.ones(128, dtype=torch.bool)}, tmp)
                os.replace(tmp, path)
            print(f'TEXT_CACHE {min(start + 4, len(missing))}/{len(missing)}', flush=True)
    dump(output / 'text_cache.json', {'unique_prompts': len(prompts), 'template_aligned': True})
    del encoder
    torch.cuda.empty_cache()

class Policy:

    def __init__(self, output):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        cfg = train_cfg()
        from wam.posttrain.eval_contract import validate_evaluation_contract
        payload = torch.load(EVAL_CKPT, mmap=True, weights_only=True, map_location='cpu')
        validate_evaluation_contract(cfg, payload, INCOMING / 'dataset_stats.json', expected_pretrain_step=PRETRAIN_STEP)
        del payload
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.skip_dit_load_from_pretrain = False
        model_cfg.mot_checkpoint_mixed_attn = False
        model_cfg.video_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.embodiment_registry_path = str(ROOT / 'configs/embodiment_registry/local_robot_arm18_capacity28_v1.json')
        from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
        model_cfg.joint_visual_codec = bundled_config(OmegaConf.to_container(model_cfg.joint_visual_codec, resolve=True))
        codec = model_cfg.joint_visual_codec
        codec.stvae_checkpoint_path = os.environ['STVAE_CHECKPOINT']
        codec.dino_kwargs.checkpoint_path = os.environ['DINO_ENCODER_CHECKPOINT']
        import wam.model.factory as factory
        original_loader = factory.load_wan22_ti2v_5b_components

        def checkpoint_only_components(**kwargs):
            kwargs['skip_dit_load_from_pretrain'] = True
            return original_loader(**kwargs)
        factory.load_wan22_ti2v_5b_components = checkpoint_only_components
        try:
            self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device='cuda').eval()
        finally:
            factory.load_wan22_ti2v_5b_components = original_loader
        from wam.training.checkpoint import load_joint96_initialization
        report = load_joint96_initialization(self.model, EVAL_CKPT, expected_step=EVAL_STEP)
        self.model.requires_grad_(False)
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(INCOMING / 'dataset_stats.json'))
        self.cfg = cfg
        report['evaluation_step'] = report.pop('pretrain_step')
        report.pop('new_global_step', None)
        dump(output / 'strict_load.json', report)
        print(f'MODEL_STRICT_LOAD_OK step={EVAL_STEP}', flush=True)

    def act(self, obs, row):
        import torch
        from eval.libero.eval_libero_single import _obs_to_model_input, _static_dimension_is_pad, _denormalize_action
        from wam.inference.frame_joint import infer_frame_joint_window
        image, proprio, _ = _obs_to_model_input(obs, self.cfg, self.processor, 256, 256, 'cuda', torch.bfloat16)
        packed = image.new_zeros((1, 3, 3, 256, 256))
        packed[:, [0, 2]] = image
        ctx = torch.load(Path(os.environ['EVAL_TEXT_CACHE']) / (hashlib.sha256(row['prompt'].encode()).hexdigest() + '.pt'), weights_only=True, map_location='cpu')
        assert ctx['prompt'] == row['prompt'] and ctx['mask'].all()
        with torch.inference_mode():
            ret = infer_frame_joint_window(self.model, None, packed, 9, 32, proprio=proprio, context=ctx['context'], context_mask=ctx['mask'], num_inference_steps=10, video_layout='latent_horizontal', video_view_names='main|left|right', camera_is_pad=torch.tensor([[False, True, False]], device='cuda'), action_dim_is_pad=_static_dimension_is_pad(self.processor, 'action')[None], proprio_dim_is_pad=_static_dimension_is_pad(self.processor, 'state')[None], decode_video=False, embodiment_ids=torch.zeros(1, dtype=torch.long, device='cuda'))
        action = _denormalize_action(ret['action'], self.processor)[0]
        return simulator_action(action)

def egl_probe(output):
    from eval.libero_plus.rendering import probe
    return probe(output)

def render_check(output):
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    from eval.libero_plus.rendering import install_render_fix
    install_render_fix()
    report = egl_probe(output)
    rows = json.loads(plan_path().read_text())
    suites = benchmark()
    results = []
    for suite in ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10'):
        row = next((r for r in rows if r['suite'] == suite))
        env = OffScreenRenderEnv(bddl_file_name=row['bddl_file'], camera_heights=256, camera_widths=256, render_gpu_device_id=-1)
        try:
            env.seed(42)
            env.reset()
            states = initial_states(suites[suite], row['task_id'])
            env.set_init_state(states[0])
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            images = {}
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                image = np.asarray(obs[key])
                if image.shape != (256, 256, 3) or not np.isfinite(image).all() or image.std() == 0:
                    raise RuntimeError(f'Invalid {suite} / {key}: {image.shape}')
                images[key] = {'shape': list(image.shape), 'dtype': str(image.dtype), 'min': int(image.min()), 'max': int(image.max()), 'std': float(image.std()), 'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
            results.append({'suite': suite, 'task_id': row['task_id'], 'images': images, 'reset_and_step': True})
            print(f"RENDER_OK {suite} backend={report['backend']} dual_camera=256x256", flush=True)
        finally:
            env.close()
    dump(output / 'render_check.json', {'passed': True, **report, 'tasks': results})

def evaluation_backend():
    backend = os.environ.get('EVAL_ENV_BACKEND', 'inprocess')
    if backend not in ('inprocess', 'subprocess'):
        raise ValueError('EVAL_ENV_BACKEND must be inprocess or subprocess')
    return backend

def worker(args):
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    output = args.output / f'worker_{args.worker:02d}'
    output.mkdir(parents=True, exist_ok=True)
    from eval.libero_plus.rendering import install_render_fix, render_backend
    env_backend = evaluation_backend()
    if env_backend == 'inprocess':
        install_render_fix()
        egl_probe(output)
    rows = json.loads((args.output / 'plan.json').read_text())
    assigned = rows[args.worker::args.workers]
    suites = benchmark()
    policy = Policy(output)
    completed, successes, errors = (0, 0, 0)
    for row in assigned:
        task = suites[row['suite']].get_task(row['task_id'])
        assert read_instruction(task)[0] == row['instruction']
        path = args.output / 'results' / row['suite'] / f"task_{row['task_id']:05d}.json"
        if path.is_file():
            old = json.loads(path.read_text())
            if not old.get('error'):
                completed += 1
                successes += old['successes']
                continue
        result = {**row, 'successes': 0, 'total_episodes': 1, 'worker': args.worker, 'step': EVAL_STEP, 'replan': 10, 'denoise': 10, 'started': time.time(), 'env_backend': env_backend, 'gripper_interface': 'native01_to_env_continuous_1minus2clipg', 'frame_contract': 'libero_osc_flu_b_identity_v1', 'render_backend': render_backend()}
        env = None
        try:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            if env_backend == 'subprocess':
                from eval.libero.eval_libero_single import SubprocessLiberoEnv
                env = SubprocessLiberoEnv(task_suite_name=row['suite'], task_id=row['task_id'], seed=42, render_gpu_device_id=-1, env_worker_cuda_visible_devices='', report_dir=str(output / f"renderer_{row['suite']}_{row['task_id']}"))
            else:
                env = OffScreenRenderEnv(bddl_file_name=row['bddl_file'], camera_heights=256, camera_widths=256, render_gpu_device_id=-1)
                env.seed(42)
            env.reset()
            states = initial_states(suites[row['suite']], row['task_id'])
            obs = env.set_init_state(states[0])
            for _ in range(30):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            pending = []
            limit = 700 if row.get('base_suite', row['suite']) == 'libero_10' else 400
            for t in range(limit):
                if not pending:
                    prediction = policy.act(obs, row)
                    if not (output / 'first_action.json').exists():
                        assert prediction.shape == (32, 7) and np.isfinite(prediction).all()
                        dump(output / 'first_action.json', {'step': EVAL_STEP, 'shape': [32, 7], 'finite': True})
                        print(f'FIRST_ACTION_OK step={EVAL_STEP}', flush=True)
                    pending = prediction[:10].tolist()
                obs, _, done, _ = env.step(decode_at_execution(pending.pop(0), obs))
                if done:
                    result['successes'] = 1
                    break
            result['env_steps'] = t + 1
            successes += result['successes']
        except Exception as exc:
            errors += 1
            result.update(error=repr(exc), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            if env is not None:
                env.close()
            result['duration'] = time.time() - result['started']
            dump(path, result)
            completed += 1
            dump(output / 'status.json', {'completed': completed, 'assigned': len(assigned), 'successes': successes, 'errors': errors, 'current': row['task_name'], 'pid': os.getpid()})
            print(f"RESULT {row['suite']}/{row['task_id']} success={result['successes']} error={bool(result.get('error'))} completed={completed}/{len(assigned)}", flush=True)
        if result.get('error'):
            raise RuntimeError(f"Worker stopped after technical failure: {result['error']}")

def summarize(output):
    rows = [json.loads(p.read_text()) for p in (output / 'results').glob('*/*.json')]
    if len(rows) != len({(r['suite'], r['task_id']) for r in rows}):
        raise ValueError('Duplicate result identities')
    plan = json.loads((output / 'plan.json').read_text()) if (output / 'plan.json').is_file() else []
    if plan and (not {(r['suite'], r['task_id']) for r in rows} <= {(r['suite'], r['task_id']) for r in plan}):
        raise ValueError('Results contain tasks outside this run plan')
    groups = {}
    for field in ('suite', 'category'):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        groups[field] = {key: {'completed': len(values), 'successes': sum((r['successes'] for r in values)), 'errors': sum((bool(r.get('error')) for r in values))} for key, values in grouped.items()}
    data = {'completed': len(rows), 'successes': sum((r['successes'] for r in rows)), 'expected': len(plan), 'success_rate': sum((r['successes'] for r in rows)) / len(rows) if rows else None, 'errors': sum((bool(r.get('error')) for r in rows)), **groups}
    exit_path = output / 'exits.json'
    exits = json.loads(exit_path.read_text()) if exit_path.is_file() else None
    data['worker_returncodes'] = exits
    data['process_failures'] = sum((code != 0 for code in exits)) if exits is not None else None
    dump(output / 'summary.json', data)
    print(json.dumps(data), flush=True)
    return data

def check_gpus(gpu_text):
    gpus = [int(gpu) for gpu in gpu_text.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or any((g < 0 for g in gpus)):
        raise ValueError('Use unique explicit nonnegative GPU indices')
    gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    usage = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu_usage.strip().splitlines()}
    if not set(gpus) <= usage.keys():
        raise ValueError('A requested GPU does not exist')
    for gpu in gpus:
        if usage[gpu] >= 1024:
            raise RuntimeError(f'Refusing occupied GPU {gpu}: {usage[gpu]} MiB')
    return gpus

def launch(args):
    from eval.libero_plus.rendering import render_backend
    rendering = render_backend()
    env_backend = evaluation_backend()
    if env_backend == 'subprocess':
        print('WARNING: subprocess rendering is diagnostic only; observation parity and policy smoke require target-platform validation. See docs/EVALUATION.md.', flush=True)
    rows = json.loads(plan_path().read_text())
    if args.phase == 'smoke':
        rows = [next((r for r in rows if r['suite'] == suite)) for suite in SMOKE_SUITES]
    global_expected = len(rows)
    selection_hash = hashlib.sha256(json.dumps(sorted(((r['suite'], r['task_id']) for r in rows))).encode()).hexdigest()
    if args.shards < 1 or not 0 <= args.shard_rank < args.shards:
        raise ValueError('Invalid evaluation shard rank/count')
    rows = rows[args.shard_rank::args.shards]
    if not rows:
        raise ValueError('Empty evaluation shard')
    gpus = check_gpus(args.gpus)
    if args.workers_per_gpu < 1:
        raise ValueError('Use unique explicit GPU indices and a positive worker count')
    workers = min(len(rows), len(gpus) * args.workers_per_gpu)
    if (args.output / 'launch.json').exists():
        raise FileExistsError('Use a new output directory; previous results cannot be mixed')
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / 'plan.json', rows)
    procs = []
    for i in range(workers):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpus[i % len(gpus)])
        if rendering == 'egl':
            env['MUJOCO_EGL_DEVICE_ID'] = env['CUDA_VISIBLE_DEVICES']
        else:
            env.pop('MUJOCO_EGL_DEVICE_ID', None)
        stream = (args.output / f'worker_{i:02d}.log').open('w')
        p = subprocess.Popen([sys.executable, '-m', RUNNER_MODULE, '--phase', 'worker', '--worker', str(i), '--workers', str(workers), '--output', str(args.output)], env=env, stdout=stream, stderr=subprocess.STDOUT)
        stream.close()
        procs.append(p)
    dump(args.output / 'launch.json', {'pids': [p.pid for p in procs], 'workers': workers, 'phase': args.phase, 'checkpoint': str(EVAL_CKPT), 'expected': len(rows), 'global_expected': global_expected, 'selection_sha256': selection_hash, 'shards': args.shards, 'shard_rank': args.shard_rank, 'checkpoint_sha256': None, 'hash_validation': False, 'started': time.time(), 'render_backend': rendering, 'env_backend': env_backend})
    import signal
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            exits = [p.poll() for p in procs]
            dump(args.output / 'process_status.json', {'pids': [p.pid for p in procs], 'returncodes': exits, 'updated': time.time()})
            if any((code is not None and code != 0 for code in exits)):
                break
            if all((code is not None for code in exits)):
                break
            time.sleep(2)
    finally:
        for child in procs:
            if child.poll() is None:
                child.terminate()
        for child in procs:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        signal.signal(signal.SIGTERM, previous_handler)
        exits = [p.returncode for p in procs]
        dump(args.output / 'exits.json', exits)
    result = summarize(args.output)
    if any(exits) or result['errors'] or result['completed'] != len(rows):
        raise RuntimeError('Evaluation not technically complete; inspect worker logs')
    dump(args.output / 'COMPLETE.json', {**result, 'passed': True, 'expected': len(rows), 'phase': args.phase, 'checkpoint': str(EVAL_CKPT), 'posttrain_step': EVAL_STEP, 'pretrain_step': PRETRAIN_STEP, 'env_backend': env_backend, 'render_backend': rendering, 'finished_at': time.time()})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('audit', 'audit-full', 'preflight', 'text', 'render-check', 'smoke', 'balanced10', 'full', 'worker', 'summary'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-rank', type=int, default=0)
    args = p.parse_args()
    if sys.flags.optimize:
        raise RuntimeError('Python -O disables checkpoint guards and is not supported')
    from eval.libero_plus.rendering import render_backend
    render_backend()
    from wam.posttrain.guard import install_import_guard
    install_import_guard()
    args.output = args.output.resolve()
    os.environ['EVAL_OUTPUT'] = str(args.output)
    os.environ['LIBERO_CONFIG_PATH'] = str(args.output / 'libero_config')
    os.environ.setdefault('EVAL_TEXT_CACHE', str(args.output / 'text_cache'))
    if args.phase != 'summary':
        from eval.libero.bootstrap import setup_libero_paths
        setup_libero_paths()
    if args.phase != 'summary':
        read_checkpoint_progress()
    if args.phase in ('smoke', 'balanced10', 'full'):
        if (args.output / 'launch.json').exists():
            raise FileExistsError('Use a new output directory')
        os.environ.setdefault('EVAL_SELECTION', 'full' if args.phase == 'full' else 'balanced10')
        gpus = check_gpus(args.gpus)
        preflight(args.output, smoke=args.phase == 'smoke')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpus[0])
        subprocess.run([sys.executable, '-m', RUNNER_MODULE, '--phase', 'text', '--output', str(args.output)], env=env, check=True)
    if args.phase == 'audit':
        audit(args.output)
    elif args.phase == 'audit-full':
        audit_full(args.output)
    elif args.phase == 'preflight':
        preflight(args.output)
    elif args.phase == 'text':
        text_cache(args.output)
    elif args.phase == 'render-check':
        render_check(args.output)
    elif args.phase == 'worker':
        worker(args)
    elif args.phase == 'summary':
        summarize(args.output)
    else:
        launch(args)
if __name__ == '__main__':
    main()
