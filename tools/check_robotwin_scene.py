"""Simulator-only check of frozen task 0 / episode 0; no policy scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['c2c', 'c2r'], required=True)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    from eval.robotwin.launch import simulator_env, required, ROOT
    if not args.child:
        root = required('ROBOTWIN_ROOT')
        env = simulator_env(root, args.gpu, ROOT/'validation_runs/robotwin_scene_check/cache')
        env['WAM_RW_INPUT_SET'] = required('ROBOTWIN_FROZEN_'+args.mode.upper())
        subprocess.run([required('ROBOTWIN_SIM_PYTHON'), str(Path(__file__).resolve()),
            '--child', '--mode', args.mode], cwd=root, env=env, check=True)
        return
    from eval.robotwin import rollout as r
    from eval.robotwin.episode_metadata import scene_metadata
    from eval.robotwin.input_fixtures import freeze_or_verify
    import yaml
    fixture = Path(os.environ['WAM_RW_INPUT_SET'])
    spec = json.loads((fixture/'SPEC.json').read_text())
    task = spec['tasks'][0]
    selected = json.loads((fixture/task/'CANDIDATES.json').read_text())['selected']['0']
    def check(task_name, unused, setup, model, seed, **kwargs):
        env = r.class_decorator(task_name)
        try:
            setup['eval_mode'] = True
            env.setup_demo(now_ep_num=0, seed=selected['seed'], is_test=True, **setup)
            info = scene_metadata(task_name, env)
            env.set_instruction(selected['instruction'])
            _, report = freeze_or_verify(env, task_name, 0, selected['seed'], selected['instruction'], info)
            report.update(task=task_name, seed=selected['seed'], instruction=selected['instruction'])
            output = ROOT/'validation_runs/robotwin_scene_check'/f'{args.mode}.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report), flush=True)
        finally: r._safe_close_env(env)
        return selected['seed']+1, 0
    r.eval_policy = check
    r.eval_function_decorator = lambda *unused: lambda *unused: None
    cfg = yaml.safe_load((ROOT/'eval/robotwin/policy.yaml').read_text())
    cfg.update(task_name=task, task_config=spec['task_config'], ckpt_setting='scene_check',
        eval_num_episodes=1, eval_video_log=False,
        eval_output_dir=str(ROOT/'validation_runs/robotwin_scene_check'/args.mode))
    r.main(cfg)


if __name__ == '__main__': main()
