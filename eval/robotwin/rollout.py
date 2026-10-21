import sys
import ast
import os
import subprocess
import contextlib
import json
import signal
import time

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
import traceback

import yaml
from datetime import datetime
import importlib
import argparse

from generate_episode_instructions import *

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


class EvaluationTimeout(TimeoutError):
    pass


def _model_profile_section(model, name):
    section = getattr(model, "profile_section", None)
    return section(name) if callable(section) else contextlib.nullcontext()


def _record_model_timing(model, name, elapsed, *, count_name=None):
    recorder = getattr(model, "add_client_timing", None)
    if callable(recorder):
        recorder(name, elapsed, count_name=count_name)


@contextlib.contextmanager
def _time_limit(seconds, label):
    seconds = int(seconds or 0)
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise EvaluationTimeout(f"{label} exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _atomic_write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def _load_progress(path, *, enabled):
    path = Path(path)
    if not enabled or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        print(f"Ignoring unreadable progress file {path}: {exc}", flush=True)
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_close_env(task_env, *, clear_cache=False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception as exc:
        print(f"close_env warning: {exc}", flush=True)


def _safe_close_video(task_env):
    try:
        if getattr(task_env, "eval_video_ffmpeg", None) is not None:
            task_env._del_eval_video_ffmpeg()
    except Exception as exc:
        print(f"ffmpeg close warning: {exc}", flush=True)


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    resume = parse_bool(usr_args.get("resume", True))
    expert_timeout_sec = int(usr_args.get("expert_timeout_sec", 300))
    rollout_setup_timeout_sec = int(usr_args.get("rollout_setup_timeout_sec", 300))
    episode_timeout_sec = int(usr_args.get("episode_timeout_sec", 1800))
    heartbeat_interval_sec = float(usr_args.get("heartbeat_interval_sec", 10))
    max_seed_attempts = int(usr_args.get("max_seed_attempts", 20000))
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)
    result_suffix = _result_suffix_from_task_config(task_config)
    progress_path = save_dir / f"_progress_{result_suffix}.json"
    heartbeat_path = save_dir / f"_heartbeat_{result_suffix}"
    progress = _load_progress(progress_path, enabled=resume)
    progress.update(
        {
            "version": 1,
            "task_name": task_name,
            "task_config": task_config,
            "target_episodes": eval_num_episodes,
            "status": "model_loading",
            "updated_at": datetime.now().isoformat(),
        }
    )
    _atomic_write_json(progress_path, progress)
    heartbeat_path.touch()

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + ", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + ", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]
    configured_start_seed = usr_args.get("eval_start_seed")
    st_seed = (
        100000 * (1 + seed)
        if configured_start_seed is None or str(configured_start_seed).strip() == ""
        else int(configured_start_seed)
    )
    if st_seed < 0:
        raise ValueError(f"`eval_start_seed` must be non-negative, got: {st_seed}")
    suc_nums = []
    test_num = eval_num_episodes

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   progress_path=progress_path,
                                   heartbeat_path=heartbeat_path,
                                   resume=resume,
                                   expert_timeout_sec=expert_timeout_sec,
                                   rollout_setup_timeout_sec=rollout_setup_timeout_sec,
                                   episode_timeout_sec=episode_timeout_sec,
                                   heartbeat_interval_sec=heartbeat_interval_sec,
                                   max_seed_attempts=max_seed_attempts)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    final_progress = _load_progress(progress_path, enabled=True)
    final_progress.update(
        {
            "status": "complete",
            "completed_episodes": test_num,
            "successes": int(suc_num),
            "next_seed": int(st_seed),
            "updated_at": datetime.now().isoformat(),
        }
    )
    _atomic_write_json(progress_path, final_progress)
    heartbeat_path.touch()
    # return task_reward


def eval_policy(*args, **kwargs):
    from eval.robotwin.rollout_protocol import run
    return run(*args, runtime=globals(), **kwargs)


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
