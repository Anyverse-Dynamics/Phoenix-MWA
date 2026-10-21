"""Evaluate RoboCasa policies with the existing GR1 joint-controller simulator."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=39100)
    p.add_argument("--output", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--execute-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scene-seed-mode", choices=("fixed", "legacy_unseeded", "seed_list"), default="legacy_unseeded")
    p.add_argument("--scene-seed-list", help="Saved JSON manifest, required in seed_list mode")
    p.add_argument("--expected-checkpoint")
    p.add_argument("--expected-package")
    p.add_argument("--sim-root", default=str(Path(__file__).resolve().parents[1] / "third_party/robocasa_runner"))
    a = p.parse_args()
    if a.episodes < 1 or a.max_episode_steps < 1 or not 1 <= a.execute_steps <= 32:
        p.error("Require positive episodes/steps and 1 <= execute-steps <= 32")
    from wam.robocasa_simulation import ensure_python_hash_seed, scene_seed_config, create_seeded_runner
    from wam.robocasa_simulation import create_legacy_unseeded_runner
    from wam.robocasa_simulation import create_seed_list_runner, load_scene_seed_list
    if (a.scene_seed_mode == "seed_list") != bool(a.scene_seed_list):
        p.error("--scene-seed-list is required exclusively with --scene-seed-mode seed_list")
    scene_config = (load_scene_seed_list(a.scene_seed_list, a.episodes) if a.scene_seed_list
                    else scene_seed_config(a.seed, a.scene_seed_mode))
    ensure_python_hash_seed(a.seed)
    for key, value in {"NUMBA_CACHE_DIR": "/tmp/robocasa_delta_numba",
                       "MPLCONFIGDIR": "/tmp/robocasa_delta_mpl",
                       "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}.items():
        os.environ.setdefault(key, value)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    sim_root = Path(a.sim_root).resolve(strict=True)
    # Use the simulator environment's paired robosuite/robocasa installation.
    sys.path.append(str(sim_root))
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from Robocasa_tabletop.eval_files.simulation_env import SimulationConfig, MultiStepConfig
    client = WebsocketClientPolicy(a.host, a.port, connect_timeout=600)
    metadata = client.get_server_metadata()
    if (metadata.get("policy") not in ("robocasa_delta80_joint_v1", "robocasa_absolute47_joint_v1", "robocasa_delta80_slot18_joint_v1") or metadata.get("execute_steps") != a.execute_steps
            or metadata.get("inference_mode") != "joint_video_actions" or metadata.get("seed") != a.seed
            or metadata.get("seed_mode") != "fixed_per_replan"):
        client.close()
        raise ValueError("Wrong policy server or execution horizon/seed strategy")
    if metadata.get("policy") == "robocasa_absolute47_joint_v1" and (
            metadata.get("embodiment_id") != 18 or metadata.get("action_dim") != 47):
        client.close()
        raise ValueError("Wrong absolute47 action space/prompt slot")
    if metadata.get("policy") == "robocasa_delta80_slot18_joint_v1" and (
            metadata.get("embodiment_id") != 18 or metadata.get("action_dim") != 80):
        client.close()
        raise ValueError("Wrong delta80 slot18 action space/prompt slot")
    for key, value in (("checkpoint", a.expected_checkpoint), ("package", a.expected_package)):
        if value is not None and metadata.get(key) != str(Path(value).resolve()):
            client.close()
            raise ValueError(f"Connected to an unexpected policy server: {key}")
    (output / "config.json").write_text(json.dumps(
        {**vars(a), "server": metadata, "scene_seed": scene_config}, indent=2) + "\n")
    class RemotePolicy:
        def request(self, **kwargs):
            response = client.predict_action(kwargs)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error")))
            return response["data"]
        def reset(self, task_description=None):
            return self.request(reset=True, task_description=task_description)
        def step(self, observations):
            import numpy as np
            observation = {k: v for k, v in observations.items() if k.startswith("state.") or k.startswith("video.")}
            observation["annotation.human.coarse_action"] = np.asarray(observations["annotation.human.coarse_action"]).reshape(-1).tolist()
            return self.request(observations=observation)
    policy = RemotePolicy()
    cfg = SimulationConfig(env_name=a.env, n_episodes=a.episodes, n_envs=1, seed=a.seed,
        multistep=MultiStepConfig(n_action_steps=a.execute_steps, max_episode_steps=a.max_episode_steps))
    if a.scene_seed_mode == "seed_list":
        runner = create_seed_list_runner(model=policy, audit_dir=output / "scene_seeds",
                                         episode_seeds=scene_config["episode_seeds"])
    else:
        runner_factory = create_seeded_runner if a.scene_seed_mode == "fixed" else create_legacy_unseeded_runner
        runner = runner_factory(model=policy, audit_dir=output / "scene_seeds")
    try:
        result = runner.run_simulation(cfg, progress_json_path=str(output / "progress.json"))
        (output / "result.json").write_text(json.dumps(result, default=str, indent=2) + "\n")
    finally:
        if runner.env is not None:
            runner.env.close()
        client.close()


if __name__ == "__main__":
    main()
