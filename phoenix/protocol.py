"""Model-only Phoenix protocol; no Torch, ROS or robot SDK imports."""
import copy
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import CAMERAS, HORIZON, VIDEO_FRAMES, DENOISING_STEPS

PROTOCOL = "phoenix.prometheus.v1"


def array(value, shape, dtype, name):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {dtype} {shape}")
    return value.copy()


class InferenceInterface:
    def __init__(self, model, config, *, model_version, normalization_sha256):
        self.model, self.config = model, config
        self.lock = threading.Lock()
        self.metadata = dict(protocol=PROTOCOL, model="phoenix", model_version=model_version,
            normalization_sha256=normalization_sha256, action_horizon=32, video_frames=9,
            text_context_sha256=config["text_context_sha256"],
            denoising_steps=10, action_dim=14, action_hz=30, camera_keys=list(CAMERAS),
            action_representation="abs_qpos", gripper_unit=config["gripper_unit"],
            modes=["sync", "rtc"], rtc_method="condition", rtc=copy.deepcopy(config["rtc"]),
            instruction=config["instruction"], seed=config["seed"])

    def infer(self, request):
        started = time.perf_counter()
        if not isinstance(request, dict) or request.get("type") != "infer":
            raise ValueError("Expected infer request")
        if set(request)-{"type","request_id","control_epoch","start_tick","model_version","mode","examples"}:
            raise ValueError("Unexpected Phoenix request fields; use the native action_prefix")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id)>128:
            raise ValueError("Invalid request_id")
        for key in ("control_epoch", "start_tick"):
            if type(request.get(key)) is not int or request[key]<0:
                raise ValueError(f"Invalid {key}")
        if request.get("model_version") != self.metadata["model_version"]:
            raise ValueError("Model identity differs")
        mode = request.get("mode")
        if mode not in ("sync", "rtc"):
            raise ValueError("Choose sync or rtc")
        examples = request.get("examples")
        if not isinstance(examples, list) or len(examples)!=1 or not isinstance(examples[0], dict):
            raise ValueError("Expected one observation")
        sample = examples[0]
        if set(sample)-{"state", "lang", "action_prefix", *CAMERAS} or sample.get("lang") != self.config["instruction"]:
            raise ValueError("Unexpected instruction or fields")
        if "rtc" in request:
            raise ValueError("Training RTC uses examples[0].action_prefix")
        state = array(sample["state"], (14,), np.dtype("float32"), "state")
        images = {}
        for key in CAMERAS:
            value = sample[key]
            if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[-1]!=3 or not 1<=min(value.shape[:2]) or max(value.shape[:2])>4096:
                raise ValueError("Expected three uint8 HWC RGB images")
            images[key] = value.copy()
        prefix = sample.get("action_prefix",np.empty((0,14),np.float32))
        if not isinstance(prefix,np.ndarray) or prefix.ndim!=2 or not 0<=len(prefix)<=self.config["rtc"]["max_prefix_steps"]:
            raise ValueError("Prefix exceeds the Training RTC range")
        prefix = array(prefix,(len(prefix),14),np.dtype("float32"),"action_prefix")
        if mode=="sync" and len(prefix):
            raise ValueError("Sync requires an empty prefix")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("One model request may be in flight")
        try:
            actions = array(self.model.predict(state, images, prefix=prefix), (32,14), np.dtype("float32"), "model actions")
            if not np.array_equal(actions[:len(prefix)],prefix):
                raise ValueError("Model changed the committed physical prefix")
        finally:
            self.lock.release()
        return dict(type="inference_result",status="ok",ok=True,request_id=request_id,
            model_version=self.metadata["model_version"],control_epoch=request["control_epoch"],
            start_tick=request["start_tick"],mode=mode,guided=bool(len(prefix)),prefix_length=len(prefix),
            inference_s=time.perf_counter()-started,data=dict(denormalized_actions=actions[None]))


@dataclass(frozen=True)
class Plan:
    actions: np.ndarray
    epoch: int
    start_tick: int
    roundtrip_s: float
    guided: bool


class PhoenixPolicy:
    def __init__(self, transport, config):
        self.transport, self.config = transport, config
        metadata = transport.metadata
        expected = dict(protocol=PROTOCOL,model="phoenix",action_horizon=32,video_frames=9,
            denoising_steps=10,action_dim=14,action_hz=30,camera_keys=list(CAMERAS),
            action_representation="abs_qpos",rtc_method="condition",rtc=config["rtc"],
            model_version=config["model_version"],normalization_sha256=config["normalization_sha256"],
            text_context_sha256=config["text_context_sha256"],
            instruction=config["instruction"],seed=config["seed"],gripper_unit=config["gripper_unit"])
        if any(metadata.get(key)!=value for key,value in expected.items()):
            raise ValueError("Phoenix model/configuration identity differs")
        self.epoch, self.serial = 0, 0
        self.closed = False

    def invalidate(self):
        self.epoch += 1

    def infer(self, state, images, *, mode, start_tick, action_prefix=None):
        if self.closed:
            raise RuntimeError("Policy is closed")
        epoch = self.epoch
        self.serial += 1
        request = dict(type="infer",request_id=f"{epoch}:{self.serial}",control_epoch=epoch,
            start_tick=start_tick,model_version=self.config["model_version"],mode=mode,
            examples=[dict(state=state,lang=self.config["instruction"],**images)])
        prefix=np.empty((0,14),np.float32) if action_prefix is None else action_prefix
        request["examples"][0]["action_prefix"]=prefix
        started = time.perf_counter()
        reply = self.transport.infer(request)
        elapsed = time.perf_counter()-started
        if epoch!=self.epoch or self.closed:
            raise RuntimeError("Stale response after stop/reset")
        if not reply.get("ok") or reply.get("status")!="ok":
            raise RuntimeError(reply.get("error","Model inference failed"))
        expected = {key:request[key] for key in ("request_id","control_epoch","start_tick","model_version","mode")}
        expected.update(type="inference_result",guided=bool(len(prefix)),prefix_length=len(prefix))
        if any(type(reply.get(key)) is not type(value) or reply.get(key)!=value for key,value in expected.items()):
            raise ValueError("Response request/model/origin/epoch differs")
        actions = array(reply.get("data",{}).get("denormalized_actions"),(1,32,14),np.dtype("float32"),"response actions")[0]
        if not np.array_equal(actions[:len(prefix)],prefix):
            raise ValueError("Response changed the committed physical prefix")
        actions.setflags(write=False)
        return Plan(actions, epoch, start_tick, elapsed, bool(len(prefix)))

    def close(self):
        self.closed = True
        self.invalidate()
        self.transport.close()


class FixedChunkPolicy:
    """Fixed queue semantics parameterized by horizon, execution length and delay."""
    def __init__(self, policy, mode):
        if mode not in ("rtc","sync"):
            raise ValueError("Choose sync or rtc")
        self.policy, self.mode = policy, mode
        self.previous = None

    def reset(self):
        self.policy.invalidate()
        self.previous = None

    def infer(self, state, images, *, start_tick):
        cfg = self.policy.config["rtc"]
        s,d = cfg["execution_horizon"],cfg["inference_delay"]
        aligned = None
        if self.mode=="rtc" and self.previous is not None:
            aligned = self.previous[s:s+d].copy()
        plan = self.policy.infer(state,images,mode=self.mode,start_tick=start_tick,action_prefix=aligned)
        if aligned is not None and plan.roundtrip_s+1/30 > d/30:
            raise TimeoutError("RTC round trip exceeded the committed queue budget")
        if self.mode=="sync":
            result = plan.actions.copy()
        else:
            result = plan.actions[d if self.previous is not None else 0:s+d].copy()
        self.previous = plan.actions.copy()
        return result,plan
