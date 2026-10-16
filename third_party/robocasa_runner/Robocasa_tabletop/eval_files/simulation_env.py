# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tyro
import dataclasses
import json
import logging
import os

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Robocasa_tabletop.eval_files.base_config import ModalityConfig
from Robocasa_tabletop.eval_files.base_config import BasePolicy
from Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

from Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper
# from Robocasa_tabletop.eval_files.model2robocasa_interface_v1 import PolicyWarper


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2 # 和10 是什么关系
    fps: int = 10 # BUG 数据集中应该是 20？
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    seed: Optional[int] = None
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[BasePolicy] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations: # BUG @JinhuiYE here only one viwes
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.model.step(observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the model."""
        return self.model.get_modality_config()

    def _extract_task_description(self, observations: Optional[Dict[str, Any]]) -> Optional[str]:
        if observations is None:
            return None
        raw_desc = observations.get("annotation.human.coarse_action")
        if raw_desc is None:
            return None
        arr = np.asarray(raw_desc)
        if arr.size == 0:
            return None
        return str(arr.reshape(-1)[0])

    def _maybe_reset_model(self, observations: Optional[Dict[str, Any]] = None) -> None:
        if self.model is None or not hasattr(self.model, "reset"):
            return
        task_desc = self._extract_task_description(observations)
        try:
            self.model.reset(task_desc)
        except TypeError:
            # Backward compatibility with older reset signatures.
            self.model.reset()

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(
        self,
        config: SimulationConfig,
        model: Optional[BasePolicy] = None,
        progress_json_path: Optional[str] = None,
    ) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.
        
        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
            progress_json_path: Optional JSON path updated after each completed episode.
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")
        
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        if config.seed is not None:
            print(f"Simulation seed: {config.seed}")
            random.seed(config.seed)
            np.random.seed(config.seed)
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset
        obs, _ = self.env.reset(seed=config.seed)
        self._maybe_reset_model(obs)
        if progress_json_path is not None:
            write_eval_progress_json(
                progress_json_path,
                config.env_name,
                episode_successes,
                target_episodes=config.n_episodes,
                status="running",
            )
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Print online metric so we can monitor success rate mid-run.
                    current_success_rate = float(np.mean(episode_successes))
                    print(
                        f"[Eval Progress] {completed_episodes}/{config.n_episodes} "
                        f"| success_rate={current_success_rate:.4f}"
                    )
                    if progress_json_path is not None:
                        write_eval_progress_json(
                            progress_json_path,
                            config.env_name,
                            episode_successes,
                            target_episodes=config.n_episodes,
                            status="running",
                        )
                    self._maybe_reset_model(next_obs)
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        if progress_json_path is not None:
            write_eval_progress_json(
                progress_json_path,
                config.env_name,
                episode_successes,
                target_episodes=config.n_episodes,
                status="complete",
            )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_model(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the model."""
        # Get actions from the model
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    if config.seed is not None:
        env_seed = config.seed + idx
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(env_seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(env_seed)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def _episode_payload(task_name: str, episode_successes: List[bool]) -> Dict[str, Any]:
    n = len(episode_successes)
    n_ok = int(np.sum(episode_successes))
    return {
        "task_name": task_name,
        "n_episodes": n,
        "n_successes": n_ok,
        "success_rate": float(n_ok / n) if n else 0.0,
        "episodes": [
            {"episode_index": i, "task_name": task_name, "success": bool(s)}
            for i, s in enumerate(episode_successes)
        ],
    }


def write_json_atomic(out_path: str, payload: Dict[str, Any]) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)


def write_eval_results_json(
    out_path: str,
    task_name: str,
    episode_successes: List[bool],
) -> None:
    """Write per-episode outcomes and aggregate success rate for one Robocasa task."""
    write_json_atomic(out_path, _episode_payload(task_name, episode_successes))


def write_eval_progress_json(
    out_path: str,
    task_name: str,
    episode_successes: List[bool],
    target_episodes: int,
    status: str,
) -> None:
    """Write in-progress per-episode outcomes for monitoring long eval tasks."""
    payload = _episode_payload(task_name, episode_successes)
    payload.update(
        {
            "status": status,
            "completed_episodes": len(episode_successes),
            "target_episodes": int(target_episodes),
            "updated_at_unix": time.time(),
        }
    )
    write_json_atomic(out_path, payload)


def run_evaluation(
    env_name: str,
    model: BasePolicy,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    seed: Optional[int] = None,
    results_json_path: Optional[str] = None,
    progress_json_path: Optional[str] = None,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
        seed: Random seed for deterministic environment initialization.
        results_json_path: If set, write eval_results JSON (episode list + rates) here.
        progress_json_path: If set, update progress JSON after each completed episode.
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config, progress_json_path=progress_json_path)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    if results_json_path is not None:
        write_eval_results_json(results_json_path, results[0], results[1])
        print(f"Wrote eval results JSON to {results_json_path}")
    return results


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 5678
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360 # 
    n_action_steps: int = 3

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Optional[str] = None  # Path to save videos. None disables video recording.
    # Default: <video_out_path>/eval_results.json, or ./eval_results.json when videos are disabled.
    results_json_path: Optional[str] = None
    # Default: sibling progress.json next to results_json_path.
    progress_json_path: Optional[str] = None

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = "results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt"
    dataset_stats_path: Optional[str] = None
    sim_cfg_name: str = "sim_robocasa.yaml"
    sim_cfg_path: Optional[str] = None
    sim_task: Optional[str] = None



def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        dataset_stats_path=args.dataset_stats_path,
        sim_cfg_name=args.sim_cfg_name,
        sim_cfg_path=args.sim_cfg_path,
        sim_task=args.sim_task,
    )
    results_json = args.results_json_path
    if results_json is None:
        if args.video_out_path is not None:
            results_json = str(Path(args.video_out_path) / "eval_results.json")
        else:
            results_json = "eval_results.json"
    progress_json = args.progress_json_path
    if progress_json is None:
        progress_json = str(Path(results_json).with_name("progress.json"))
    run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        results_json_path=results_json,
        progress_json_path=progress_json,
    )

def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
