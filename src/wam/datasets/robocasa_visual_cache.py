"""Versioned, episode-sharded RoboCasa Wan48/CSVAE48 sliding-window cache.

Only real main-camera latents are stored. The training-facing reader restores
the two absent camera slots without storing them on disk.
"""
from __future__ import annotations
from collections import OrderedDict
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from wam.posttrain.robocasa_paths import resolve_source
SCHEMA = 'wam.robocasa.dual48.window_cache.v1'
SHAPE = (96, 3, 16, 16)
OFFSETS = np.arange(0, 33, 4)
ROW_BYTES = int(np.prod(SHAPE)) * 2
CAMERA = 'observation.images.ego_view'
DEFAULT_DATA = 'data/robocasa'
DEFAULT_WEIGHTS = 'checkpoints'
DEFAULT_VAE = 'checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
TIMESTAMP_ATOL_SECONDS = 0.001
TIMESTAMP_POLICY = 'native20hz_numeric_drift_max1ms_v1'
LEGACY_CACHE_IMPLEMENTATION = '57b34b9c118c1a19975dd3efc0d6885033c8fcad2409dbaba8f9b35b509ed82f'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(resolve_source(path).read_text())

def file_identity(path):
    stat = resolve_source(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

def file_hash(path):
    """File existence only; runtime content hashing is disabled."""
    resolve_source(path).resolve(strict=True)
    return None

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

@contextlib.contextmanager
def locked(path, *, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        yield

def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / 'model/modules/codecs/joint_visual_encoder.py', root / 'model/backbones/wan22/wan_video_vae.py', root / 'model/backbones/wan22/helpers/loader.py', root / 'model/backbones/wan22/helpers/state_dict_converters.py', root / 'utils/torch24_compat.py']
    return {str(path.relative_to(root)): file_hash(path) for path in files}

def validate_implementation(plan, compatibility_path=None):
    """Accept the reviewed validation-only repair without rewriting old receipts."""
    current = implementation_identity()
    if plan['implementation'] == current:
        return {'mode': 'exact', 'implementation': current}
    require(compatibility_path is not None, 'Cache implementation changed; supply a reviewed --resume-compat record')
    compatibility = read_json(compatibility_path)
    require(compatibility.get('schema') == 'wam.robocasa.timestamp_resume.v1', 'Unknown resume compatibility schema')
    require(compatibility.get('old_implementation') == plan['implementation'], 'Resume compatibility legacy implementation mismatch')
    require(compatibility.get('new_implementation') == current, 'Resume compatibility current implementation mismatch')
    key = 'datasets/robocasa_visual_cache.py'
    changed = {name for name in set(current) | set(plan['implementation']) if current.get(name) != plan['implementation'].get(name)}
    require(changed == {key} and plan['implementation'][key] == LEGACY_CACHE_IMPLEMENTATION, 'Resume compatibility may only change the reviewed cache implementation')
    require(compatibility.get('timestamp_policy') == TIMESTAMP_POLICY, 'Resume timestamp policy mismatch')
    return {'mode': 'reviewed_timestamp_repair', 'implementation': current, 'compatibility_sha256': file_hash(compatibility_path), 'timestamp_policy': TIMESTAMP_POLICY}

def check_resume(args):
    plan = load_plan(args.plan)
    audit = validate_implementation(plan, args.resume_compat)
    print(json.dumps({'plan_sha256': plan['sha256'], **audit}), flush=True)

def validate_video_timestamps(pts, stamps, episode):
    pts = np.asarray(pts, dtype=np.float64)
    require(pts.shape == stamps.shape and len(pts) == episode['length'], 'Video frame count differs from numeric rows')
    require(np.isfinite(pts).all() and np.allclose(np.diff(pts), 0.05, atol=1e-06, rtol=0), 'RoboCasa video requires native 20Hz timestamps')
    delta = np.abs(pts - stamps)
    require(np.all(delta <= TIMESTAMP_ATOL_SECONDS), f"Video and numeric timestamps differ: task={episode['task']} episode={episode['episode']} max_abs_s={float(delta.max()):.9g} allowed_s={TIMESTAMP_ATOL_SECONDS}")

def window_mapping(length, start):
    require(0 <= start < max(0, length - 4), 'Window needs one real future sampled image')
    indices = start + OFFSETS
    return {'start': start, 'frame_indices': np.minimum(indices, length - 1).tolist(), 'image_is_pad': (indices >= length).tolist(), 'action_is_pad': (start + np.arange(32) >= length).tolist()}

def starts_for(episode):
    return episode['starts'] if 'starts' in episode else range(episode['windows'])

def episode_output(root, episode):
    return Path(root) / 'episodes' / episode['task'] / f"episode_{episode['episode']:06d}"

def assigned_episodes(plan, rank, world):
    require(world > 0 and 0 <= rank < world, 'Invalid extraction rank/world')
    return [entry for i, entry in enumerate(plan['episodes']) if i % world == rank]

def check_episode_source(episode):
    for kind in ('video', 'parquet'):
        require(resolve_source(episode[kind]).is_file(), f'Missing source {kind}: {episode[kind]}')

def load_plan(path):
    plan = read_json(path)
    expected = plan.pop('sha256', None)
    require(plan['schema'] == SCHEMA, 'Plan identity mismatch')
    plan['sha256'] = expected
    require(sum((e['windows'] for e in plan['episodes'])) == plan['windows'], 'Plan count mismatch')
    return plan

def prepare(args):
    from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
    from wam.training.checkpoint import _joint96_identity
    paths = {'policy': args.policy, 'vae': args.vae, 'csvae': args.csvae, 'dino': args.dino}
    resources = {}
    for name, path in paths.items():
        path = str(Path(path).resolve(strict=True))
        print(f'[prepare] hashing {name}: {path}', flush=True)
        resources[name] = {'path': path, **file_identity(path), 'sha256': file_hash(path)}
    payload = torch.load(resources['policy']['path'], map_location='cpu', mmap=True, weights_only=True)
    identity = _joint96_identity(payload['joint96_metadata'])
    require(identity['action_dim'] == identity['proprio_dim'] == 80, 'Expected canonical80 policy')
    require(identity['channel_order'] == ['wan_vae48', 'csvae48'], 'Unexpected visual channel order')
    source_root = str(args.source_root)
    encoder_source_sha256(source_root)
    data_root = Path(args.data_root).resolve(strict=True)
    tasks = sorted(data_root.glob('gr1_unified.*'))
    require(len(tasks) == 24, f'Expected 24 RoboCasa tasks, got {len(tasks)}')
    episodes, metadata = ([], [])
    for root in tasks:
        info = read_json(root / 'meta/info.json')
        require(info['robot_type'] == 'GR1ArmsAndWaistFourierHands', 'Unexpected robot')
        require(float(info['fps']) == 20 and info['features'][CAMERA]['shape'] == [256, 256, 3], f'Unexpected camera/fps: {root}')
        for name in ('info.json', 'episodes.jsonl', 'modality.json'):
            path = root / 'meta' / name
            metadata.append({'path': str(path), **file_identity(path), 'sha256': file_hash(path)})
        rows = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        require(len(rows) == info['total_episodes'], 'Episode count mismatch')
        require(sum((int(row['length']) for row in rows)) == info['total_frames'], 'Frame count mismatch')
        require(len({r['episode_index'] for r in rows}) == len(rows), 'Duplicate episode IDs')
        for row in sorted(rows, key=lambda r: r['episode_index']):
            ep, length = (int(row['episode_index']), int(row['length']))
            if length <= 4:
                continue
            fmt = {'episode_index': ep, 'episode_chunk': ep // info['chunks_size'], 'video_key': CAMERA}
            entry = {'task': root.name, 'episode': ep, 'length': length, 'fps': 20.0, 'windows': length - 4}
            for kind in ('video', 'parquet'):
                path = root / info['video_path' if kind == 'video' else 'data_path'].format(**fmt)
                entry[kind] = str(path)
                entry[kind + '_identity'] = file_identity(path)
            episodes.append(entry)
    plan = {'schema': SCHEMA, 'scope': 'full24_all_episodes', 'resources': resources, 'source_root': source_root, 'source_sha256': identity['source_sha256'], 'encoder_identity': {k: v for k, v in identity.items() if k != 'soft_prompt'}, 'metadata': metadata, 'data_root': str(data_root), 'episodes': episodes, 'windows': sum((e['windows'] for e in episodes)), 'feature_shape': list(SHAPE), 'dtype': 'bfloat16', 'storage_dtype': 'uint16', 'camera': CAMERA, 'camera_is_pad': [False, True, True], 'rgb_offsets': OFFSETS.tolist(), 'stride': 1, 'minimum_real_future_offset': 4, 'tail_policy': 'repeat_last_rgb_with_explicit_padding_masks', 'rgb_preprocessing': 'pyav_rgb24_256_uint8_div255_float32_times2_minus1_then_bf16', 'augmentation': 'none', 'posterior': 'mean', 'torch_version': str(torch.__version__), 'visual_attention_backend': 'math_sdpa', 'implementation': implementation_identity()}
    plan['sha256'] = digest(plan)
    target = Path(args.plan)
    with locked(target.with_suffix('.lock')):
        if target.exists():
            require(load_plan(target) == plan, 'Existing plan differs; use a new output directory')
        else:
            atomic_json(target, plan)
    print(json.dumps({'plan': str(target), 'episodes': len(episodes), 'windows': plan['windows'], 'feature_GiB': plan['windows'] * ROW_BYTES / 2 ** 30, 'sha256': plan['sha256']}), flush=True)

def decode_episode(episode):
    import av
    import pyarrow.parquet as pq
    check_episode_source(episode)
    table = pq.read_table(resolve_source(episode['parquet']), columns=['timestamp', 'episode_index'])
    stamps = np.asarray(table['timestamp'].to_pylist(), dtype=np.float64).reshape(-1)
    eps = np.asarray(table['episode_index'].to_pylist()).reshape(-1)
    require(len(stamps) == episode['length'] and np.all(eps == episode['episode']), 'Parquet episode identity mismatch')
    require(np.isfinite(stamps).all() and np.allclose(np.diff(stamps), 0.05, atol=1e-06, rtol=0), 'RoboCasa requires native 20Hz timestamps')
    images, pts = ([], [])
    with av.open(str(resolve_source(episode['video']))) as container:
        container.streams.video[0].thread_count = 1
        for frame in container.decode(video=0):
            require(frame.time is not None, 'Missing video timestamp')
            image = frame.to_ndarray(format='rgb24')
            require(image.shape == (256, 256, 3), 'Unexpected image dimensions')
            images.append(image)
            pts.append(frame.time)
    require(len(images) == episode['length'], 'Video frame count differs from numeric rows')
    validate_video_timestamps(pts, stamps, episode)
    return (np.stack(images), stamps)

def pack_rgb(images, mapping):
    selected = torch.from_numpy(images[mapping['frame_indices']].copy()).permute(3, 0, 1, 2)
    main = selected.float().div(255).mul(2).sub(1)
    result = torch.full((1, 3, 3, len(mapping['frame_indices']), 256, 256), -1.0)
    result[0, 0] = main
    return result

class CacheEncoder:

    def __init__(self, plan, device):
        from wam.utils.torch24_compat import install_safe_serialization_compat
        from wam.model.backbones.wan22.helpers.loader import _load_registered_model
        from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder
        install_safe_serialization_compat()
        require(plan['visual_attention_backend'] == 'math_sdpa', 'Unvalidated visual attention backend')
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        r = plan['resources']
        for resource in r.values():
            require(file_identity(resource['path']) == {k: resource[k] for k in ('size', 'mtime_ns')}, f"Checkpoint changed since prepare: {resource['path']}")
        self.vae = _load_registered_model(r['vae']['path'], 'wan_video_vae', torch_dtype=torch.bfloat16, device=str(self.device))
        source = plan['source_root']
        cfg = {'source_root': source, 'source_sha256': plan['source_sha256'], 'csvae_checkpoint_path': r['csvae']['path'], 'expected_csvae_step': None, 'dino_kwargs': {'checkpoint_path': r['dino']['path'], 'checkpoint_state': 'ema_student', 'output_source': 'backbone', 'projection_mode': 'raw', 'projector_out_dim': 1024, 'dinov3_source_commit': plan['encoder_identity']['dinov3_source_commit'], 'verify_source_commit': False, 'strict_load': True, 'image_size': 256, 'patch_size': 16}}
        self.encoder = JointVisualEncoder.from_config(cfg, vae=self.vae, device=self.device)
        self.pad = torch.tensor([[False, True, True]], device=self.device)
        self.runtime = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(self.device), 'encoder': 'JointVisualEncoder.encode', 'camera_slots': 3, 'window_batch_size': 1, 'visual_attention_backend': 'math_sdpa'}

    @torch.inference_mode()
    def encode(self, rgb):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            result = self.encoder.encode(rgb.to(self.device), camera_is_pad=self.pad)
        require(tuple(result.shape) == (1, 96, (rgb.shape[3] - 1) // 4 + 1, 16, 48), 'Unexpected latent shape')
        require(result.dtype == torch.bfloat16 and bool(torch.isfinite(result).all()), 'Invalid latent dtype/values')
        require(not bool(result[..., 16:].count_nonzero()), 'Padded cameras are not zero')
        return result[0, ..., :16].contiguous().cpu()

def validate_complete(root, episode, plan_hash, *, verify_hash=True):
    folder = episode_output(root, episode)
    receipt = read_json(folder / 'COMPLETE.json')
    require(receipt['rows'] == episode['windows'] and receipt['feature_shape'] == list(SHAPE), 'Receipt shape mismatch')
    features = folder / 'features.bf16.bin'
    require(features.stat().st_size == episode['windows'] * ROW_BYTES, 'Truncated feature file')
    return receipt

class RoboCasaVisualCache:
    """Lookup by stable (task, episode, start), independent of extraction world size."""

    def __init__(self, root, plan_path, max_open_episodes=8):
        self.root, self.plan = (Path(root), load_plan(plan_path))
        self.episodes = {(e['task'], e['episode']): e for e in self.plan['episodes']}
        self.max_open = max_open_episodes
        require(max_open_episodes > 0, 'max_open_episodes must be positive')
        self._open = OrderedDict()

    def __getstate__(self):
        return {**self.__dict__, '_open': OrderedDict()}

    def get(self, task, episode, start):
        key = (task, episode)
        entry = self.episodes[key]
        mapping = window_mapping(entry['length'], start)
        if key not in self._open:
            check_episode_source(entry)
            validate_complete(self.root, entry, self.plan['sha256'])
            self._open[key] = np.memmap(episode_output(self.root, entry) / 'features.bf16.bin', mode='r', dtype=np.uint16, shape=(entry['windows'], *SHAPE))
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)
        self._open.move_to_end(key)
        row = entry['starts'].index(start) if 'starts' in entry else start
        value = torch.from_numpy(np.array(self._open[key][row], copy=True)).view(torch.bfloat16)
        require(bool(torch.isfinite(value).all()), 'Nonfinite cached feature')
        return {'video_latents': torch.cat((value, torch.zeros_like(value), torch.zeros_like(value)), dim=-1), 'video_shape': torch.tensor([3, 3, 9, 256, 256]), 'camera_is_pad': torch.tensor([False, True, True]), 'image_is_pad': torch.tensor(mapping['image_is_pad']), 'mapping': mapping}

def extract(args):
    plan = load_plan(args.plan)
    implementation_audit = validate_implementation(plan, args.resume_compat)
    require(plan['torch_version'] == str(torch.__version__), 'Torch runtime differs from prepared plan')
    root = Path(args.output)
    episodes = assigned_episodes(plan, args.rank, args.world_size)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / f'rank_{args.rank:05d}.lock'):
        with locked(root / 'binding.lock', wait=True):
            binding = root / 'BINDING.json'
            if binding.exists():
                pass
            else:
                atomic_json(binding, {'schema': SCHEMA, 'plan_sha256': plan['sha256'], 'plan_path': str(Path(args.plan).resolve())})
        for meta in plan['metadata']:
            require(file_identity(meta['path']) == {k: meta[k] for k in ('size', 'mtime_ns')}, 'Source metadata changed')
        pending, skipped = ([], 0)
        for episode in episodes:
            check_episode_source(episode)
            if (episode_output(root, episode) / 'COMPLETE.json').exists():
                validate_complete(root, episode, plan['sha256'])
                skipped += 1
            else:
                pending.append(episode)
        print(f'[rank={args.rank}] assigned={len(episodes)} skipped={skipped} pending={len(pending)}', flush=True)
        start_time = time.monotonic()
        runtime = CacheEncoder(plan, args.device) if pending else None
        total = 0
        for episode in pending:
            folder = episode_output(root, episode)
            with locked(folder / 'episode.lock'):
                if (folder / 'COMPLETE.json').exists():
                    validate_complete(root, episode, plan['sha256'])
                    continue
                images, timestamps = decode_episode(episode)
                tmp = folder / 'features.partial.bin'
                array = np.memmap(tmp, mode='w+', dtype=np.uint16, shape=(episode['windows'], *SHAPE))
                checks = []
                t0 = time.monotonic()
                for row, frame in enumerate(starts_for(episode)):
                    mapping = window_mapping(episode['length'], frame)
                    rgb = pack_rgb(images, mapping)
                    value = runtime.encode(rgb)
                    array[row] = value.view(torch.uint16).numpy()
                    if row == 0 or (row + 1) % args.log_every == 0:
                        print(f"[rank={args.rank}] {episode['task']} ep={episode['episode']} {row + 1}/{episode['windows']} elapsed={time.monotonic() - t0:.1f}s", flush=True)
                array.flush()
                del array
                with tmp.open('rb') as stream:
                    os.fsync(stream.fileno())
                check_episode_source(episode)
                target = folder / 'features.bf16.bin'
                os.replace(tmp, target)
                receipt = {'schema': SCHEMA, 'plan_sha256': plan['sha256'], 'episode_sha256': digest(episode), 'rows': episode['windows'], 'feature_shape': list(SHAPE), 'features_sha256': file_hash(target), 'rank': args.rank, 'runtime': runtime.runtime, 'elapsed_s': time.monotonic() - t0, 'checks': checks, 'extraction_implementation': implementation_audit}
                atomic_json(folder / 'COMPLETE.json', receipt)
                total += episode['windows']
                atomic_json(root / f'progress_rank_{args.rank:05d}.json', {'plan_sha256': plan['sha256'], 'extracted_windows': total, 'last_task': episode['task'], 'last_episode': episode['episode'], 'elapsed_s': time.monotonic() - start_time})
        atomic_json(root / f'rank_{args.rank:05d}.json', {'plan_sha256': plan['sha256'], 'world_size': args.world_size, 'rank': args.rank, 'episodes': len(episodes), 'skipped_episodes': skipped, 'new_windows': total, 'elapsed_s': time.monotonic() - start_time, 'status': 'complete'})

def verify(args):
    plan = load_plan(args.plan)
    count = 0
    runtimes = set()
    missing = []
    for episode in plan['episodes']:
        folder = episode_output(args.output, episode)
        if not (folder / 'COMPLETE.json').exists():
            missing.append([episode['task'], episode['episode']])
            continue
        receipt = validate_complete(args.output, episode, plan['sha256'], verify_hash=args.hash_all)
        check_episode_source(episode)
        count += episode['windows']
        runtimes.add(digest(receipt['runtime']))
    require(len(runtimes) <= 1, 'Mixed hardware/software encoding runtimes in cache')
    summary = {'schema': SCHEMA, 'plan_sha256': plan['sha256'], 'status': 'complete' if not missing else 'incomplete', 'windows': count, 'expected_windows': plan['windows'], 'missing_episodes': len(missing), 'missing_preview': missing[:20], 'hash_all': args.hash_all}
    print(json.dumps(summary, indent=2), flush=True)
    if not missing:
        atomic_json(Path(args.output) / 'COMPLETE.json', summary)
    require(not missing or args.allow_incomplete, 'Cache incomplete')

