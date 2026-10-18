"""Expert-screened input collection and immutable replay, separate from policy score."""
import fcntl
import json
import os
import time
from pathlib import Path


def run(task_name, TASK_ENV, args, model, st_seed, test_num=100,
        video_size=None, instruction_type=None, skip_get_obs_within_replan=False,
        progress_path=None, heartbeat_path=None, resume=True,
        expert_timeout_sec=300, rollout_setup_timeout_sec=300,
        episode_timeout_sec=1800, heartbeat_interval_sec=10, max_seed_attempts=20000,
        runtime=None):
    if runtime is None:
        from eval.robotwin import rollout as r
    else:
        # The IPC client loads rollout under a private module name and injects
        # remote-policy callbacks there. Do not import an unpatched second copy.
        from types import SimpleNamespace
        r = SimpleNamespace(**runtime)
    from eval.robotwin.episode_metadata import scene_metadata
    from eval.robotwin.input_fixtures import freeze_or_verify
    directory = Path(os.environ['WAM_RW_INPUT_SET'])
    mode = os.environ['WAM_RW_SET_MODE']
    spec = json.loads((directory / 'SPEC.json').read_text())
    assert mode == 'reuse', 'Only read-only replay is supported'
    assert task_name in spec['tasks'] and 0 < test_num <= spec['episodes_per_task']
    assert args['task_config'] == spec['task_config'] and not args['eval_video_log']
    assert instruction_type == spec['instruction_type']
    args['eval_mode'] = True
    output = Path(progress_path).parent
    taskdir = directory / task_name
    ledger_path = taskdir / 'CANDIDATES.json'
    ledger = json.loads(ledger_path.read_text())
    assert ledger['task'] == task_name
    episodes = []
    initial_calls = ledger['expert_play_once_calls']
    eval_func = r.eval_function_decorator(args['policy_name'], 'eval')
    reset = r.eval_function_decorator(args['policy_name'], 'reset_model')
    last_heartbeat = 0.

    def heartbeat(force=False):
        nonlocal last_heartbeat
        if force or time.monotonic() - last_heartbeat > heartbeat_interval_sec:
            Path(heartbeat_path).touch()
            last_heartbeat = time.monotonic()

    def notify(status, seed=None):
        r._atomic_write_json(Path(progress_path), dict(status=status, task_name=task_name,
            completed_episodes=len(episodes), successes=sum(x['success'] for x in episodes),
            target_episodes=test_num, current_seed=seed, episodes=episodes,
            input_set=str(directory), mode=mode,
            expert_play_once_calls=ledger['expert_play_once_calls']-initial_calls,
            skipped_seeds=[x for x in ledger['attempts'] if x['status'].startswith('rejected')]))
        heartbeat(True)

    # Collection is single-writer per task; replay only reads accepted inputs.
    with (output / 'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in range(test_num):
            result_file = output / f'episode_{index:02d}_RESULT.json'
            if result_file.exists():
                assert resume
                saved = json.loads(result_file.read_text())
                assert saved['seed'] == ledger['selected'][str(index)]['seed']
                episodes.append(saved)
                continue
            env = None
            seed = None
            started = time.monotonic()
            try:
                selected = ledger['selected'].get(str(index))
                counters = (sum(x['success'] for x in episodes), len(episodes))
                if selected is None:
                    raise ValueError('Replay cannot select or replace a missing seed')
                else:
                    seed = selected['seed']
                    env = r.class_decorator(task_name)
                    env.suc, env.test_num = counters
                    notify('setup', seed)
                    with r._time_limit(rollout_setup_timeout_sec, f'replay setup {seed}'):
                        env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
                def forbidden(*a, **kw):
                    raise RuntimeError('Expert must not run after candidate admission')
                env.play_once = forbidden
                notify('scene_validation', seed)
                with r._time_limit(rollout_setup_timeout_sec, f'metadata {seed}'):
                    info = scene_metadata(task_name, env)
                    instruction = selected.get('instruction')
                    if instruction is None:
                        raise ValueError('Frozen instruction missing')
                    env.set_instruction(instruction)
                    initial, signature = freeze_or_verify(env, task_name, index, seed, instruction, info)
                record = dict(index=index, seed=seed, instruction=instruction,
                    language_seed=seed, initial_signature=signature, input_set=str(directory),
                    expert_plan_success=True, expert_check_success=True)
                r._atomic_write_json(output / f'episode_{index:02d}_IDENTITY.json', record)
                reset(model)
                notify('rollout', seed)
                success = False
                with r._time_limit(episode_timeout_sec, f'policy rollout {seed}'):
                    first = True
                    while env.take_action_cnt < env.step_lim:
                        heartbeat()
                        obs = initial if first else env.get_obs()
                        first = False
                        eval_func(env, model, obs)
                        if env.eval_success:
                            success = True
                            break
                record.update(success=success, action_steps=int(env.take_action_cnt),
                    step_lim=int(env.step_lim), elapsed_seconds=time.monotonic()-started)
                # Failure at step_lim is a valid result; never rescreen it.
                r._atomic_write_json(result_file, record)
                episodes.append(record)
                notify('episode_complete', seed)
            except BaseException as exc:
                r._atomic_write_json(output / 'TECHNICAL_ERROR.json', dict(
                    index=index, seed=seed, error=repr(exc), policy_outcome_seed_selection=False))
                notify('technical_error', seed)
                raise
            finally:
                if env is not None:
                    r._safe_close_env(env)
        notify('complete')
    return ledger['selected'][str(test_num-1)]['seed']+1, sum(x['success'] for x in episodes)
