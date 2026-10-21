"""Three real RoboCasa updates plus all-tensor initialization/readback checks."""
import json
import math
from pathlib import Path
import sys
import time

import torch
from wam.trainer import WAMTrainer
from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder

_load = WAMTrainer._load_weight_checkpoint_before_prepare
_train = WAMTrainer.train
_finish = WAMTrainer._finish_training


def forbid_encode(*args, **kwargs):
    raise AssertionError('Cache-only training must not invoke the online visual encoder')


def compare_groups(model, payload):
    counts = {}
    for group in ('mot', 'proprio_encoder', 'action_proprio_encoder'):
        if group not in payload:
            continue
        actual = getattr(model, group).state_dict()
        assert actual.keys() == payload[group].keys(), group
        for name, value in actual.items():
            expected = payload[group][name].to(device=value.device, dtype=value.dtype)
            assert torch.equal(value, expected), f'{group}.{name}'
        counts[group] = len(actual)
    assert counts.get('mot', 0) > 0 and counts.get('proprio_encoder', 0) > 0
    return counts


def load(self, model):
    result = _load(self, model)
    payload = torch.load(self.cfg.init_checkpoint, mmap=True, weights_only=True, map_location='cpu')
    key = 'mixtures.action.embodiment_prompt.embedding.weight'
    prompt = payload['mot'][key].clone()
    assert not prompt[18].count_nonzero()
    actual = model.mot.state_dict()[key][18].detach().cpu()
    assert actual.count_nonzero() and not torch.equal(actual[0], actual[1])
    prompt[18] = actual.to(prompt)
    payload['mot'][key] = prompt
    self._smoke_init = compare_groups(model, payload)
    return result


def train(self):
    assert self.batch_size == self.gradient_accumulation_steps == 1 and self.max_steps == 3
    assert self.cfg.learning_rate == 1e-4 and self.cfg.save_on_train_end
    assert self.cfg.data.train.allow_smoke and self.cfg.data.train.smoke_selection
    assert not self.cfg.save_training_state and self.cfg.save_every == self.cfg.eval_every == 0
    assert self.accelerator.num_processes == 4 and self.global_step == 0
    self._smoke_metrics = []
    def forward(module, args, result):
        sample = args[0]
        assert sample['embodiment_id'].shape == (1,) and (sample['embodiment_id'] == 18).all()
        assert 'video' not in sample and sample['video_latents'].shape == (1, 96, 3, 16, 48)
        metrics = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                   for k, v in result[1].items() if k.startswith('loss')}
        metrics['loss'] = float(result[0].detach())
        assert all(math.isfinite(v) for v in metrics.values())
        self._smoke_metrics.append(metrics)
    self.model.register_forward_hook(forward)
    params = dict(self.model.named_parameters())
    self._smoke_prompt = next(n for n in params if n.endswith('embodiment_prompt.embedding.weight'))
    self._smoke_prompt_before = params[self._smoke_prompt].detach().cpu().clone()
    self._smoke_before, counts = {}, {}
    for name, value in params.items():
        group = next((g for g in ('mixtures.video', 'mixtures.action', 'proprio_encoder') if g in name), None)
        if group and value.requires_grad and counts.get(group, 0) < 8:
            self._smoke_before[name] = value.detach().flatten()[:65536].cpu().clone()
            counts[group] = counts.get(group, 0) + 1
    torch.cuda.reset_peak_memory_stats()
    self._smoke_started = time.perf_counter()
    return _train(self)


def finish(self):
    torch.cuda.synchronize()
    params = dict(self.model.named_parameters())
    deltas = {name: float((params[name].detach().flatten()[:65536].cpu().float() - before.float()).abs().max())
              for name, before in self._smoke_before.items()}
    changed = {g: any(v > 0 for n, v in deltas.items() if g in n)
               for g in ('mixtures.video', 'mixtures.action', 'proprio_encoder')}
    prompt_delta = (params[self._smoke_prompt].detach().cpu().float() - self._smoke_prompt_before.float()).abs()
    rank = self.accelerator.process_index
    report = dict(rank=rank, world_size=4, batch_size=1, step=self.global_step,
        engine_steps=self.model.global_steps, skipped_steps=self.model.skipped_steps,
        pretrain_step=self._unwrap_core_model().model_paths['posttrain_initialization']['step'], exact_initialization_except_fresh_slot18=self._smoke_init,
        metrics=self._smoke_metrics, parameter_groups_changed=changed,
        prompt_slot18_max_delta=float(prompt_delta[18].max()), online_encoder_calls=0,
        grad_norm=float(self.model.get_global_grad_norm()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        wall_seconds=time.perf_counter()-self._smoke_started)
    report['updates_passed'] = (report['step'] == report['engine_steps'] == 3
        and report['skipped_steps'] == 0 and len(self._smoke_metrics) == 3
        and math.isfinite(report['grad_norm']) and all(changed.values())
        and report['prompt_slot18_max_delta'] > 0)
    assert report['updates_passed'], report
    core = self._unwrap_core_model()
    result = _finish(self)
    if rank == 0:
        # Slot filenames are implementation details; the payload step is authoritative.
        paths = list((Path(self.output_dir)/'checkpoints/weights').glob('*.pt'))
        assert len(paths) == 1, paths
        payload = torch.load(paths[0], mmap=True, weights_only=True, map_location='cpu')
        assert payload['step'] == 3
        report['saved_checkpoint'] = str(paths[0])
        report['save_readback_exact'] = compare_groups(core, payload)
    report['passed'] = True
    (Path(self.output_dir)/f'smoke_rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PHOENIX_TRAIN_SMOKE', json.dumps(report), flush=True)
    return result


WAMTrainer._load_weight_checkpoint_before_prepare = load
WAMTrainer.train = train
WAMTrainer._finish_training = finish
JointVisualEncoder.encode = forbid_encode

if __name__ == '__main__':
    import hydra
    from train import run_from_config, _strip_launcher_rank_args
    @hydra.main(config_path=str(Path(__file__).resolve().parents[1]/'configs'), config_name='train', version_base='1.3')
    def main(cfg):
        run_from_config(cfg)
    sys.argv = _strip_launcher_rank_args(sys.argv)
    main()
