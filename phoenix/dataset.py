"""Phoenix H32 LeRobot-v3 windows with aligned RGB, state and action inputs."""
import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, default_collate

from phoenix.training_data import dimension_padding, joint14_to_80, joint80_to_14

PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Joint14Processor:
    """The inverse contract used by trainer diagnostics and real inference."""
    def __init__(self, stats):
        self.stats = stats
    def normalize(self, value, kind):
        mean = torch.tensor(self.stats[kind]['mean'], dtype=value.dtype, device=value.device)
        std = torch.tensor(self.stats[kind]['std'], dtype=value.dtype, device=value.device)
        return (value - mean) / std
    def decode(self, value, kind='action'):
        native = joint80_to_14(value)
        mean = torch.tensor(self.stats[kind]['mean'], dtype=native.dtype, device=native.device)
        std = torch.tensor(self.stats[kind]['std'], dtype=native.dtype, device=native.device)
        return native * std + mean


class PhoenixDataset(Dataset):
    def __init__(self, manifest_path, normalization, text_cache_index, *, split="train",
                 embodiment_id=2, gripper_unit="m", visual_cache_index=None, **kwargs):
        if kwargs or split not in ("train", "val"):
            raise ValueError("Unexpected Phoenix reader arguments/split")
        self.num_frames, self.action_horizon, self.image_frames = 33, 32, 9
        self.embodiment_id = int(embodiment_id)
        self.manifest_sha256 = file_hash(manifest_path)
        self.norm_sha256 = file_hash(normalization)
        self.text_sha256 = file_hash(text_cache_index)
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("gripper_unit", gripper_unit) != gripper_unit:
            raise ValueError("Manifest and normalization units differ")
        self.visual_cache = None
        if visual_cache_index:
            from phoenix.data.visual_cache import VisualCache
            self.visual_cache = VisualCache(visual_cache_index, manifest_path)
        if self.manifest.get("schema") not in ("phoenix.windows.v1", "internw0.arx_x5.v3_index.v1"):
            raise ValueError("Unexpected frozen Phoenix manifest schema")
        if self.manifest.get("fps", 30) != 30:
            raise ValueError("Phoenix H32 reader requires 30Hz")
        stats = json.loads(Path(normalization).read_text())
        from phoenix.model import Normalizer
        Normalizer(stats, gripper_unit)
        self.processor = Joint14Processor(stats)
        self.text_index_root = Path(text_cache_index).parent
        self.text = json.loads(Path(text_cache_index).read_text())
        if self.text.get('schema') not in ('phoenix.text_index.v1', 'internw0.arx_x5.t5_index.v1') or self.text.get('prompt_template') != PROMPT:
            raise ValueError('Explicit Wan T5 cache index required')
        if self.text.get('manifest_sha256') != self.manifest_sha256:
            raise ValueError('T5 cache belongs to a different frozen manifest')
        self.is_training_set = split == "train"
        self.episodes = [e for e in self.manifest['episodes'] if e['split'] == split]
        self.ends, self.ranges = [], []
        count = 0
        for episode in self.episodes:
            if episode['fps'] != 30 or len(episode['views']) != 3:
                raise ValueError('Phoenix requires 30Hz and main/left/right views')
            size = max(0, int(episode['length']) - 4)
            if not size:
                raise ValueError('Frozen episode has no real future image')
            self.ranges.append((count, count + size));count += size;self.ends.append(count)
        if not count:
            raise ValueError('No windows in the requested Phoenix split')
        self.sample_episode_ranges = tuple(self.ranges)
        self.fixed_validation_panel = [{'index': int((i + .5) * count / min(128, count)), 'seed': 20261007 + i,
                                        'source': 'phoenix', 'embodiment_id': self.embodiment_id}
                                       for i in range(min(128, count))] if not self.is_training_set else []
        self.cache = OrderedDict();self.cache_bytes = 0;self.limit = 128 << 20
        self.text_cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]
    def denormalize_action(self, action, *, stats_group=None, dim_is_pad=None):
        """Return physical canonical80 values for the trainer's masked diagnostics."""
        result = joint14_to_80(self.processor.decode(action))
        if dim_is_pad is not None:
            pad = torch.as_tensor(dim_is_pad, device=result.device, dtype=torch.bool)
            while pad.ndim < result.ndim:
                pad = pad.unsqueeze(-2)
            result = result.masked_fill(pad, 0)
        return result
    def _table(self, episode, start):
        begin = int(episode['dataset_from_index']) + start
        stop = int(episode['dataset_from_index']) + min(start + self.num_frames, episode['length'])
        columns = ['index', 'frame_index', 'episode_index', 'timestamp', 'task_index',
                   self.manifest['state_key'], self.manifest['action_key']]
        pieces = []
        for part in episode['row_groups']:
            if part['end'] <= begin or part['start'] >= stop:
                continue
            identity = self.manifest['files'][part['path']]
            stat = Path(part['path']).stat()
            if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError('Frozen source identity drift')
            key = (part['path'], part['row_group'])
            table = self.cache.pop(key, None)
            if table is None:
                with pq.ParquetFile(part['path']) as reader:
                    table = reader.read_row_group(part['row_group'], columns=columns, use_threads=False)
                if table.nbytes <= self.limit:
                    while self.cache and self.cache_bytes + table.nbytes > self.limit:
                        self.cache_bytes -= self.cache.popitem(last=False)[1].nbytes
                    self.cache_bytes += table.nbytes;self.cache[key] = table
            else:
                self.cache[key] = table
            pieces.append(table.slice(max(begin, part['start']) - part['start'], min(stop, part['end']) - max(begin, part['start'])))
        if not pieces:
            raise ValueError('Missing frozen state rows')
        table = pa.concat_tables(pieces)
        if (table['index'].to_pylist() != list(range(begin, stop))
                or set(table['episode_index'].to_pylist()) != {episode['episode_index']}
                or table['frame_index'].to_pylist() != list(range(start, start + len(table)))):
            raise ValueError('State index / episode mismatch')
        times = np.asarray(table['timestamp'].to_pylist(), dtype=np.float64)
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / episode['fps'], atol=1e-5, rtol=.03):
            raise ValueError('State clock mismatch')
        return table
    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        eidx = bisect_right(self.ends, index);episode = self.episodes[eidx]
        start = int(index) - self.ranges[eidx][0]
        table = self._table(episode, start)
        state = torch.tensor(table[self.manifest['state_key']].to_pylist(), dtype=torch.float32)
        action = torch.tensor(table[self.manifest['action_key']].to_pylist(), dtype=torch.float32)
        if state.shape != (len(table), 14) or action.shape != state.shape or not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError('Invalid native joint14 rows')
        length = min(len(action), self.action_horizon)
        padded = torch.zeros(self.action_horizon, 14);padded[:length] = self.processor.normalize(action[:length], 'action')
        action = joint14_to_80(padded)
        dim_pad = dimension_padding();time_pad = torch.arange(self.action_horizon) >= length
        valid = ~time_pad[:, None] & ~dim_pad[None]
        tasks = table['task_index'].to_pylist()
        if len(set(tasks)) != 1:
            raise ValueError('Window crosses an instruction boundary')
        task = self.manifest['tasks'][episode['root_id']][str(tasks[0])]
        prompt = PROMPT.format(task=task)
        descriptor = self.text['prompts'][hashlib.sha256(prompt.encode()).hexdigest()]
        path = (self.text_index_root / descriptor['path']).resolve()
        if path not in self.text_cache:
            if file_hash(path) != descriptor['sha256']:
                raise ValueError('T5 cache drift')
            pack = torch.load(path, map_location='cpu', weights_only=True)
            context, mask = pack['context'], pack['context_mask']
            if context.shape != (128, 4096) or context.dtype != torch.bfloat16 or mask.shape != (128,) or mask.dtype != torch.bool or not torch.isfinite(context).all() or pack['prompt'] != prompt:
                raise ValueError('Invalid T5 cache payload')
            if len(self.text_cache) >= 32:self.text_cache.popitem(last=False)
            self.text_cache[path] = context, mask
        context, mask = self.text_cache[path]
        visual = ({'video_latents': self.visual_cache.get(episode, start),
                   'video_shape': torch.tensor([3, 3, 9, 256, 256])}
                  if self.visual_cache else {'video': self.decode_video(episode, start)})
        return {'sample_id': int(index), **visual, 'camera_is_pad': torch.zeros(3, dtype=torch.bool),
                'image_is_pad': torch.arange(self.image_frames) * 4 + start >= episode['length'],
                'action': action.masked_fill(~valid, 0), 'action_mask': valid,
                'action_is_pad': time_pad, 'action_dim_is_pad': dim_pad,
                'proprio': joint14_to_80(self.processor.normalize(state[:1], 'state')),
                'proprio_is_pad': torch.zeros(1, dtype=torch.bool), 'proprio_dim_is_pad': dim_pad,
                'prompt': prompt, 'context': context, 'context_mask': mask,
                'video_layout': 'latent_horizontal', 'video_view_names': 'main|left|right',
                'embodiment_id': torch.tensor(self.embodiment_id, dtype=torch.long)}
    @staticmethod
    def decode_video(episode, start):
        import av
        import torch.nn.functional as F
        result = []
        for view in episode['views']:
            path = Path(view['path']);stat = path.stat()
            if stat.st_size != view['bytes'] or stat.st_mtime_ns != view['mtime_ns']:
                raise ValueError('Video identity drift')
            targets = view['start_s'] + np.minimum(start + np.arange(9) * 4, episode['length'] - 1) / episode['fps']
            if targets[-1] >= view['end_s'] + 1e-5:
                raise ValueError('Video range does not contain the state window')
            frames = []
            with av.open(str(path)) as container:
                stream = container.streams.video[0];stream.codec_context.thread_count = 1
                tolerance = max(.02 / episode['fps'], 1.1 * float(stream.time_base))
                container.seek(max(0, int(targets[0] / float(stream.time_base))), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:raise ValueError('Missing video timestamp')
                    stamp = float(frame.pts * stream.time_base)
                    while len(frames) < 9 and abs(stamp - targets[len(frames)]) <= tolerance:
                        frames.append(frame.to_ndarray(format='rgb24'))
                    if len(frames) == 9:break
                    if stamp > targets[len(frames)] + tolerance:raise ValueError('RGB/state synchronization mismatch')
            if len(frames) != 9:raise ValueError('Short or corrupt RGB window')
            rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
            rgb = F.interpolate(rgb, size=(256, 256), mode='bicubic', align_corners=False).clamp(0, 1)
            result.append(rgb.permute(1, 0, 2, 3) * 2 - 1)
        return torch.stack(result)

    def collate_fn(self, samples):
        return default_collate(samples)

