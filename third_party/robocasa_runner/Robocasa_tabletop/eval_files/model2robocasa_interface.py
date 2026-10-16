from collections import deque
from typing import Optional, Sequence
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
import sys
import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
print('@'*100, PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import json

GR1_EMBODIMENT_ID = 24
_STD_REG = 1e-8
_RANGE_TOL = 1e-4
UNION_STATE_LAYOUT = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
    ("wrist_l_pos", 3),
    ("wrist_l_rot6d", 6),
    ("wrist_r_pos", 3),
    ("wrist_r_rot6d", 6),
)
JOINT_ACTION_DIM = 29
WAIST_PROMPT_PREFIXES = ("unlocked_waist:", "locked_waist:")


def _action_debug_enabled() -> bool:
    return str(os.getenv("FASTWAM_ACTION_DEBUG", "")).strip().lower() in {"1", "true", "yes", "y"}


def _action_debug_max() -> int:
    try:
        return int(os.getenv("FASTWAM_ACTION_DEBUG_MAX", "3"))
    except ValueError:
        return 3


def _debug_array_summary(name: str, value) -> None:
    if not _action_debug_enabled():
        return
    try:
        arr = np.asarray(value, dtype=np.float32)
        flat = arr.reshape(-1)
        head = flat[: min(12, flat.size)].tolist()
        finite = np.isfinite(arr)
        finite_ratio = float(finite.mean()) if finite.size else 1.0
        print(
            f"[ACTION_DEBUG] {name} shape={arr.shape} "
            f"min={float(np.nanmin(arr)):.6f} max={float(np.nanmax(arr)):.6f} "
            f"mean={float(np.nanmean(arr)):.6f} std={float(np.nanstd(arr)):.6f} "
            f"finite={finite_ratio:.4f} head={head}",
            flush=True,
        )
    except Exception as exc:
        print(f"[ACTION_DEBUG] {name} summary_failed={exc}", flush=True)


def _strip_waist_prompt_prefix(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    stripped = str(text).strip()
    lowered = stripped.lower()
    for prefix in WAIST_PROMPT_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def _is_numeric_scalar(value) -> bool:
    return isinstance(value, (int, float))


def _to_numpy_tree(value):
    if isinstance(value, dict):
        return {k: _to_numpy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == 0:
            return np.asarray(value, dtype=np.float32)
        if _is_numeric_scalar(value[0]):
            return np.asarray(value, dtype=np.float32)
        if isinstance(value[0], list):
            try:
                return np.asarray(value, dtype=np.float32)
            except Exception:
                return [_to_numpy_tree(v) for v in value]
        return [_to_numpy_tree(v) for v in value]
    return value


class _SingleFieldLinearNormalizer:
    def __init__(self, stats: Dict[str, np.ndarray], mode: str):
        self.stats = stats
        self.mode = mode
        self.scale, self.offset = self._build_scale_offset(stats, mode)

    @staticmethod
    def _build_scale_offset(stats: Dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode == "z-score":
            input_mean = np.asarray(stats["mean"], dtype=np.float32)
            input_std = np.asarray(stats["std"], dtype=np.float32)
            scale = 1.0 / (input_std + _STD_REG)
            offset = -input_mean / (input_std + _STD_REG)
            return scale, offset

        if mode == "min/max":
            input_min = np.asarray(stats["min"], dtype=np.float32)
            input_max = np.asarray(stats["max"], dtype=np.float32)
        elif mode == "q01/q99":
            input_min = np.asarray(stats["q01"], dtype=np.float32)
            input_max = np.asarray(stats["q99"], dtype=np.float32)
        else:
            const_min, const_max = map(float, mode.split("/"))
            input_min = np.full_like(np.asarray(stats["min"], dtype=np.float32), const_min)
            input_max = np.full_like(np.asarray(stats["max"], dtype=np.float32), const_max)

        input_range = input_max - input_min
        ignore_dim = input_range < _RANGE_TOL
        input_range[ignore_dim] = 2.0
        scale = 2.0 / input_range
        offset = -1.0 - scale * input_min
        offset[ignore_dim] = -input_min[ignore_dim]
        return scale, offset

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x * self.scale + self.offset
        return np.clip(y, -5.0, 5.0)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale


class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, # @Jinhui
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        connect_timeout: float = 900.0,
        n_action_steps=2,
        embodiment_id: Optional[int] = None,
        dataset_stats_path: Optional[str] = None,
        sim_cfg_name: str = "sim_robocasa.yaml",
        sim_cfg_path: Optional[str] = None,
        sim_task: Optional[str] = None,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port, connect_timeout=float(connect_timeout))
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps
        # Integer id for action expert (must match training dataloader); GR1 / fourier_gr1_unified -> 24
        self.embodiment_id = embodiment_id if embodiment_id is not None else GR1_EMBODIMENT_ID

        self.task_description = None
        self._state_key = "default"
        self._action_key = "default"
        self._state_normalizer, self._action_normalizer = self._build_normalizers(
            dataset_stats_path=dataset_stats_path,
            sim_cfg_name=sim_cfg_name,
            sim_cfg_path=sim_cfg_path,
            sim_task=sim_task,
        )
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self._debug_step_count = 0
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: Optional[str | tuple] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    @staticmethod
    def _instruction_at(instructions, idx: int):
        if isinstance(instructions, str) or instructions is None:
            return instructions
        if isinstance(instructions, Sequence):
            if len(instructions) == 0:
                return ""
            if len(instructions) == 1:
                return instructions[0]
            return instructions[idx]
        return instructions


    def step(
        self, 
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """

        task_description = observations['annotation.human.coarse_action'][0] # tuple
        task_description = _strip_waist_prompt_prefix(task_description)
        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        images = ego_view
        state = {}
        state["left_arm"] = observations["state.left_arm"]      # (N, 1, 7)
        state["left_hand"] = observations["state.left_hand"]    # (N, 1, 6)
        state["right_arm"] = observations["state.right_arm"]    # (N, 1, 7)
        state["right_hand"] = observations["state.right_hand"]  # (N, 1, 6)
        state["waist"] = observations["state.waist"]            # (N, 1, 3)
        for key in ("wrist_l_pos", "wrist_l_rot6d", "wrist_r_pos", "wrist_r_rot6d"):
            obs_key = f"state.{key}"
            if obs_key in observations:
                state[key] = observations[obs_key]

        # print('@'*100,'observations',observations.keys())

        zeros_ref = state["left_arm"]
        input_state = []
        for key, dim in UNION_STATE_LAYOUT:
            value = state.get(key)
            if value is None:
                value = np.zeros((*zeros_ref.shape[:-1], dim), dtype=zeros_ref.dtype)
            input_state.append(value)
        input_state = np.concatenate(input_state, axis=-1)
        if input_state.shape[-1] != sum(dim for _, dim in UNION_STATE_LAYOUT):
            raise ValueError(f"Expected state dim 47, got {input_state.shape[-1]}")
        # Keep state in raw transformed space and let server-side processor apply
        # the single normalization pass. Avoid close-loop double normalization.
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        # Keep raw uint8 images from env and let server-side policy preprocessing
        # own the only resize/normalize path for train-aligned behavior.
        images = [[np.asarray(img, dtype=np.uint8) for img in sample] for sample in images] # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)

        # prepare vla input
        examples = []
        batch_size = len(images)
        instructions = [self.task_description] if isinstance(self.task_description, str) else self.task_description
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": self._instruction_at(instructions, b),
                "state": input_state[b],  # N_history, 58 #Hack BUG
                "embodiment_id": int(self.embodiment_id),
            }
            examples.append(example)
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        response = self.client.predict_action(vla_input)
        if not response.get("ok") or response.get("status") != "ok":
            err = response.get("error") or {}
            raise RuntimeError(
                f"Policy server inference failed: {err.get('message', err)} (full response keys: {list(response.keys())})"
            )
        if "data" not in response:
            raise RuntimeError(f"Unexpected policy server response: {response}")

        if "denormalized_actions" in response["data"]:
            raw_actions = np.asarray(response["data"]["denormalized_actions"], dtype=np.float32)
        elif "normalized_actions" in response["data"]:
            assert False
            normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
            raw_actions = self._denormalize_actions_with_stats(normalized_actions)
        else:
            raise RuntimeError(
                "Policy server response missing both `denormalized_actions` and `normalized_actions`."
            )
        
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        if raw_actions.shape[-1] < JOINT_ACTION_DIM:
            raise ValueError(f"Expected action dim >= {JOINT_ACTION_DIM}, got {raw_actions.shape[-1]}")

        debug_this = _action_debug_enabled() and self._debug_step_count < _action_debug_max()
        if debug_this:
            print(f"[ACTION_DEBUG] interface_step={self._debug_step_count}", flush=True)
            _debug_array_summary("interface.raw_actions_from_server_47d", raw_actions)
            _debug_array_summary("interface.raw_actions_first29_executed", raw_actions[..., :JOINT_ACTION_DIM])
            self._debug_step_count += 1

        raw_action = {
            # The policy is trained on the 47-D union layout.
            # RoboCasa environment control still consumes the first 29 joint dims only.
            "action.left_arm": raw_actions[:, :self.n_action_steps, 0:7],      # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],    # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26],  # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, :self.n_action_steps, 26:29],        # (B, n_action_steps, 3)
        }
        if debug_this:
            for _dbg_key, _dbg_value in raw_action.items():
                _debug_array_summary(f"interface.{_dbg_key}", _dbg_value)

        return {"actions": raw_action}

    @staticmethod
    def _is_none_like(value: Optional[str]) -> bool:
        return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})

    @staticmethod
    def _resolve_path(path_str: str, *, base: Path) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        return path.resolve()

    def _load_data_cfg(
        self,
        sim_cfg_name: str,
        sim_cfg_path: Optional[str],
        sim_task: Optional[str],
    ) -> tuple[DictConfig, Path]:
        configs_root = (PROJECT_ROOT / "configs").resolve()
        if not self._is_none_like(sim_cfg_path):
            cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
            try:
                config_name = cfg_path.relative_to(configs_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
                ) from exc
        else:
            config_name = str(sim_cfg_name)

        overrides = []
        if not self._is_none_like(sim_task):
            overrides.append(f"task={str(sim_task)}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(config_name=config_name, overrides=overrides)
        if "data" not in cfg or "train" not in cfg.data:
            raise ValueError(f"Cannot find `data.train` in config: {config_name}")
        return cfg.data.train, configs_root

    def _resolve_dataset_stats_path(
        self,
        data_cfg: DictConfig,
        config_dir: Path,
        cli_path: Optional[str],
    ) -> Path:
        raw: Optional[str] = None
        if not self._is_none_like(cli_path):
            raw = str(cli_path).strip()
        elif data_cfg.get("pretrained_norm_stats") is not None:
            p = data_cfg.get("pretrained_norm_stats")
            if not self._is_none_like(p):
                raw = str(p)

        if raw is None:
            raise FileNotFoundError(
                "Dataset stats are required for normalization. "
                "Pass `dataset_stats_path` or set `data.train.pretrained_norm_stats` in config."
            )
        resolved = self._resolve_path(raw, base=config_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"dataset_stats not found: {resolved}")
        return resolved

    @staticmethod
    def _load_dataset_stats(stats_path: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        converted = _to_numpy_tree(data)
        return converted

    @staticmethod
    def _pick_mode(
        data_cfg: DictConfig,
        branch: str,
        key: str,
        default_mode: str,
    ) -> str:
        exception_mode = data_cfg.processor.get("norm_exception_mode")
        if exception_mode is None:
            return default_mode
        branch_exception = exception_mode.get(branch)
        if branch_exception is None:
            return default_mode
        return str(branch_exception.get(key, default_mode))

    @classmethod
    def _pick_branch_mode(
        cls,
        data_cfg: DictConfig,
        branch: str,
        keys: Sequence[str],
        default_mode: str,
    ) -> str:
        modes = [cls._pick_mode(data_cfg, branch, key, default_mode) for key in keys]
        unique_modes = set(modes)
        if len(unique_modes) != 1:
            raise ValueError(
                f"Mixed normalization modes for {branch}: {list(zip(keys, modes))}. "
                "This evaluator expects one shared mode per flattened branch."
            )
        return modes[0]

    @staticmethod
    def _extract_branch_stats(
        stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        branch: str,
        keys: Sequence[str],
        use_stepwise_action_norm: bool,
    ) -> Dict[str, np.ndarray]:
        branch_stats = stats[branch]
        if branch == "action":
            prefix = "stepwise_" if use_stepwise_action_norm else "global_"
        else:
            prefix = "global_"
        merged: Dict[str, list[np.ndarray]] = {}

        for key in keys:
            if key not in branch_stats:
                raise KeyError(
                    f"Missing {branch} stats for key '{key}'. Available keys: {sorted(branch_stats.keys())}"
                )
            full = branch_stats[key]
            for stat_name, value in full.items():
                if not stat_name.startswith(prefix):
                    continue
                merged.setdefault(stat_name.removeprefix(prefix), []).append(
                    np.asarray(value, dtype=np.float32)
                )

        return {name: np.concatenate(parts, axis=-1) for name, parts in merged.items()}

    def _build_normalizers(
        self,
        dataset_stats_path: Optional[str],
        sim_cfg_name: str,
        sim_cfg_path: Optional[str],
        sim_task: Optional[str],
    ) -> tuple[_SingleFieldLinearNormalizer, _SingleFieldLinearNormalizer]:
        data_cfg, config_dir = self._load_data_cfg(
            sim_cfg_name=sim_cfg_name,
            sim_cfg_path=sim_cfg_path,
            sim_task=sim_task,
        )
        if not data_cfg.get("processor"):
            raise ValueError("Config must define `data.train.processor` for normalization setup.")
        stats_path = self._resolve_dataset_stats_path(
            data_cfg=data_cfg,
            config_dir=config_dir,
            cli_path=dataset_stats_path,
        )
        stats = self._load_dataset_stats(stats_path)

        state_keys = [str(meta.key) for meta in data_cfg.shape_meta.state]
        action_keys = [str(meta.key) for meta in data_cfg.shape_meta.action]
        self._state_key = ",".join(state_keys)
        self._action_key = ",".join(action_keys)
        default_mode = str(data_cfg.processor.norm_default_mode)
        use_stepwise_action_norm = bool(data_cfg.processor.get("use_stepwise_action_norm", False))
        state_mode = self._pick_branch_mode(data_cfg, "state", state_keys, default_mode)
        action_mode = self._pick_branch_mode(data_cfg, "action", action_keys, default_mode)

        state_stats = self._extract_branch_stats(
            stats=stats, branch="state", keys=state_keys, use_stepwise_action_norm=use_stepwise_action_norm
        )
        action_stats = self._extract_branch_stats(
            stats=stats, branch="action", keys=action_keys, use_stepwise_action_norm=use_stepwise_action_norm
        )
        state_normalizer = _SingleFieldLinearNormalizer(stats=state_stats, mode=state_mode)
        action_normalizer = _SingleFieldLinearNormalizer(stats=action_stats, mode=action_mode)
        return state_normalizer, action_normalizer

    def _normalize_state_with_stats(self, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        return self._state_normalizer.forward(state_arr)

    def _denormalize_actions_with_stats(self, normalized_actions: np.ndarray) -> np.ndarray:
        actions_arr = np.asarray(normalized_actions, dtype=np.float32)
        return self._action_normalizer.backward(actions_arr)



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Build 44-D state for Motus:
        - keep arm joints as raw values (7 + 7)
        - apply sin/cos to hand + waist joints (6 + 6 + 3 -> 30)
        total = 14 + 30 = 44
        """
        periodic_keys = {"left_hand", "right_hand", "waist"}
        for key in state.keys():
            if key in periodic_keys:
                sin_state = np.sin(state[key])
                cos_state = np.cos(state[key])
                state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    
    
