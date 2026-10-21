"""Phoenix Training RTC prefix preparation and checkpoint contracts."""
import copy
import torch

def prepare_training_rtc(action, noisy_action, timestep, valid, *, max_prefix=10, prefix_lengths=None,
                no_prefix_probability=0.0):
    if action.ndim != 3 or noisy_action.shape != action.shape or valid.shape != action.shape:
        raise ValueError("RTC requires matching [B,H,D] tensors")
    if action.shape[1]!=32:
        raise ValueError("Phoenix Training RTC requires H32")
    if timestep.shape != (action.shape[0],) or valid.dtype != torch.bool:
        raise ValueError("RTC requires timestep[B] and bool validity")
    if type(max_prefix) is not int or not 0 <= max_prefix < action.shape[1]:
        raise ValueError("Invalid maximum RTC prefix")
    if (isinstance(no_prefix_probability, bool) or not isinstance(no_prefix_probability, (float, int))
            or not 0 <= no_prefix_probability <= 1):
        raise ValueError("RTC no_prefix_probability must be a finite number in [0,1]")
    lengths = valid.any(-1).long().cumprod(1).sum(1)
    if bool((lengths == 0).any()):
        raise ValueError("RTC requires a contiguous valid first action")
    maximum = (lengths - 1).clamp(max=max_prefix)
    if prefix_lengths is None:
        prefix_lengths = (torch.rand(maximum.shape, device=action.device) * (maximum + 1)).long()
        # Default p=0 preserves uniform sampling over the inclusive range.
        # An optional nonzero value adds an independently configured P0 branch.
        if no_prefix_probability:
            force_zero = torch.rand(maximum.shape, device=action.device) < no_prefix_probability
            prefix_lengths = prefix_lengths.masked_fill(force_zero, 0)
    elif (prefix_lengths.dtype != torch.long or prefix_lengths.shape != maximum.shape
          or bool(((prefix_lengths < 0) | (prefix_lengths > maximum)).any())):
        raise ValueError("Invalid explicit RTC prefix lengths")
    prefix = torch.arange(action.shape[1], device=action.device)[None] < prefix_lengths[:, None]
    conditioned = torch.where(prefix[..., None] & valid, action, noisy_action).masked_fill(~valid, 0)
    times = timestep[:, None].expand(action.shape[:2]).clone().masked_fill(prefix, 0)
    return conditioned, times, valid & ~prefix[..., None], prefix_lengths


def training_contract(max_prefix=10,no_prefix_probability=0.0):
    if type(max_prefix) is not int or not 0<=max_prefix<32:
        raise ValueError("Training RTC prefix must be below H32")
    if type(no_prefix_probability) not in (int,float) or not 0<=no_prefix_probability<=1:
        raise ValueError("Invalid zero-prefix mixture")
    return dict(schema="phoenix.training_rtc.v1",action_horizon=32,video_frames=9,
        raw_window_frames=33,action_hz=30,
        current_observation_encoding=dict(mode="pretrained_single_current_frame_v1",numerical_batch=1),
        rtc=dict(method="condition",start_step=0,uniform_prefix=[0,max_prefix],
                 no_prefix_probability=no_prefix_probability))


def validate_training_contract(contract,config):
    expected=training_contract(config["rtc"]["max_prefix_steps"],
                               contract.get("rtc",{}).get("no_prefix_probability",0.0))
    if contract!=expected:
        raise ValueError("Requires the Phoenix H32 Training RTC checkpoint contract; a raw pretraining/H64 checkpoint cannot serve")


def configure_training_rtc(model,*,max_prefix=10,no_prefix_probability=0.0):
    contract=training_contract(max_prefix,no_prefix_probability)
    model.action_expert.training_time_rtc_enabled=True
    model.action_expert.training_time_rtc_max_delay_steps=max_prefix
    model.action_expert.training_time_rtc_joint96_posttrain=True
    model.action_expert.training_time_rtc_no_prefix_probability=no_prefix_probability
    return contract


def initialize_from_pretrained(model,payload):
    from .checkpoint import _load_joint96_checkpoint
    # Weight initialization does not add a completed Training RTC marker.
    payload=dict(payload)
    payload["joint96_metadata"]=copy.deepcopy(payload["joint96_metadata"])
    if payload["joint96_metadata"]["schema"]=="internw0.local_joint96.weights.v1":
        payload["joint96_metadata"]["schema"]="phoenix.local_joint96.weights.v1"
    result=_load_joint96_checkpoint(model,payload)
    return result


def rtc_checkpoint_metadata(model,*,optimizer_step,max_prefix=10,no_prefix_probability=0.0):
    if type(optimizer_step) is not int or optimizer_step<=0:
        raise ValueError("Completed Training RTC metadata requires a positive optimizer step")
    if (getattr(model.action_expert,"training_time_rtc_enabled",False) is not True
            or getattr(model.action_expert,"training_time_rtc_max_delay_steps",None)!=max_prefix
            or getattr(model.action_expert,"training_time_rtc_joint96_posttrain",False) is not True
            or getattr(model.action_expert,"training_time_rtc_no_prefix_probability",None)!=no_prefix_probability):
        raise ValueError("Model RTC configuration differs from the saved contract")
    from .checkpoint import joint96_checkpoint_metadata
    model.soft_prompt_training_contract=training_contract(max_prefix,no_prefix_probability)
    return joint96_checkpoint_metadata(model)


def masked_flow_loss(prediction,target,loss_mask):
    if prediction.shape!=target.shape or loss_mask.shape!=target.shape or loss_mask.dtype!=torch.bool:
        raise ValueError("Matching prediction/target/bool mask tensors required")
    count=loss_mask.sum()
    if not bool(count):raise ValueError("No valid suffix target")
    error=(prediction.float()-target.float()).square()
    return error.masked_select(loss_mask).sum()/count
