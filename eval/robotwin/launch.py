"""One policy server per GPU; frozen tasks sharded in original order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from eval.robotwin.config import inference_paths

ROOT = Path(__file__).resolve().parents[2]


def required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'Set {name}')
    path = Path(value)
    if name == 'ROBOTWIN_SIM_PYTHON':
        # Resolving a venv's python symlink would silently select its base
        # interpreter and bypass the simulator environment's dependencies.
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'Not an executable simulator interpreter: {value}')
        return str(path.absolute())
    return str(path.resolve(strict=True))


def simulator_env(benchmark, gpu, scratch):
    env = os.environ.copy()
    python = Path(required('ROBOTWIN_SIM_PYTHON'))
    prefix = python.parent.parent
    env.update(CUDA_VISIBLE_DEVICES=gpu, ROBOTWIN_ROOT=benchmark,
        PYTHONPATH=f'{ROOT}:{benchmark}:{benchmark}/envs/curobo/src:{benchmark}/description/utils',
        WAM_RW_CLIENT_CACHE_ROOT=str(scratch), WAM_RW_ACTION_HORIZON='32',
        WAM_RW_ACTION_HZ='50', WAM_RW_REPLAN_STEPS='32',
        LD_LIBRARY_PATH=f'{prefix}/lib:/usr/lib/x86_64-linux-gnu',
        PATH=f'{prefix}/bin:/usr/local/cuda/bin:{Path(sys.executable).parent}:'+env.get('PATH', ''),
        CUDA_HOME=env.get('CUDA_HOME', '/usr/local/cuda'), MAX_JOBS='2')
    env.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.cache/robotwin/torch_extensions'))
    vulkan = prefix / 'lib/python3.10/site-packages/sapien/vulkan_library'
    for key, filename in [('VK_ICD_FILENAMES','nvidia_icd.json'),
                          ('__EGL_VENDOR_LIBRARY_FILENAMES','10_nvidia.json')]:
        if (vulkan / filename).is_file(): env.setdefault(key, str(vulkan / filename))
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['c2c', 'c2r'], required=True)
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--max-tasks', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--expected-step', type=int, default=None, help='Deprecated; ignored. Step is read from checkpoint.')
    parser.add_argument('--run-id')
    parser.add_argument('--inference-config', help='Default: source RoboTwin inference config')
    parser.add_argument('--dataset-stats', help='Default: source RoboTwin normalization')
    a = parser.parse_args()
    fixtures = Path(required('ROBOTWIN_FROZEN_'+a.mode.upper()))
    benchmark = required('ROBOTWIN_ROOT')
    checkpoint = required('ROBOTWIN_CHECKPOINT')
    config, stats = map(str, inference_paths(a.inference_config, a.dataset_stats))
    sim_python = required('ROBOTWIN_SIM_PYTHON')
    spec = json.loads((fixtures/'SPEC.json').read_text())
    if not 0 < a.episodes <= spec['episodes_per_task']:
        raise ValueError('Episode count outside frozen set')
    tasks = spec['tasks'] if a.max_tasks < 0 else spec['tasks'][:a.max_tasks]
    gpus = a.gpus.split(',')
    if not tasks or not all(gpus): raise ValueError('Empty task/GPU selection')
    run_id = a.run_id or f'robotwin_{a.mode}_{datetime.now():%Y%m%d_%H%M%S}'
    if Path(run_id).name != run_id: raise ValueError('run-id must be a simple name')
    output = Path(os.environ.get('ROBOTWIN_OUTPUT_ROOT', ROOT/'validation_runs'))/run_id
    output.mkdir(parents=True, exist_ok=False)
    (output/'RUN.json').write_text(json.dumps(dict(mode=a.mode, tasks=tasks,
        episodes=a.episodes, frozen_input=str(fixtures), checkpoint=checkpoint,
        inference_config=config, normalization_stats=stats,
        replan=32, denoise=10, policy_seed=0, hash_validation=False), indent=2))
    print(f'OUTPUT={output}', flush=True)

    def worker(rank, gpu):
        owned = tasks[rank::len(gpus)]
        if not owned: return
        work = output / f'worker_{rank}'
        work.mkdir()
        for task in owned: (work/task).mkdir()
        with tempfile.TemporaryDirectory(prefix='phoenix-rw-') as temp:
            sock = str(Path(temp)/'policy.sock')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            command = [sys.executable, '-m', 'eval.robotwin.joint96_robot37_shared_server',
                '--checkpoint', checkpoint, '--inference-config', config, '--dataset-stats', stats,
                '--socket', sock,
                '--output-dir', str(work/'server'), '--run-output', str(work),
                '--replan','32','--denoise','10','--seed','0']
            with (work/'server.log').open('w') as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+1800
                    while not Path(sock).exists():
                        if server.poll() is not None: raise RuntimeError(f'Policy server failed: {work}/server.log')
                        if time.monotonic()>deadline: raise TimeoutError('Policy server startup')
                        time.sleep(2)
                    for task in owned:
                        simenv = simulator_env(benchmark, gpu, work/'scratch')
                        simenv.update(WAM_RW_INPUT_SET=str(fixtures), WAM_RW_SET_MODE='reuse')
                        command = [sim_python, '-m', 'eval.robotwin.client', '--socket', sock,
                            '--config', str(ROOT/'eval/robotwin/policy.yaml'), '--client-id', f'w{rank}-{task}',
                            '--task-label',task,'--phase',a.mode,'--overrides',
                            '--task_name',task,'--task_config',spec['task_config'],
                            '--ckpt_setting',run_id,'--instruction_type',spec['instruction_type'],
                            '--eval_num_episodes',str(a.episodes),'--eval_video_log','false',
                            '--eval_output_dir',str(work/task),'--replan_steps','32',
                            '--episode_timeout_sec','3600','--rollout_setup_timeout_sec','900']
                        with (work/task/'client.log').open('w') as clientlog:
                            subprocess.run(command,cwd=benchmark,env=simenv,stdout=clientlog,
                                stderr=subprocess.STDOUT,check=True,timeout=900+a.episodes*3900)
                finally:
                    server.terminate()
                    try: server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill(); server.wait()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lambda item: worker(*item), enumerate(gpus)))
    results = [json.loads(p.read_text()) for p in output.glob('worker_*/*/episode_*_RESULT.json')]
    assert len(results) == len(tasks)*a.episodes
    summary = dict(completed=len(results), successes=sum(r['success'] for r in results),
                   technical_errors=0, mode=a.mode, output=str(output))
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
