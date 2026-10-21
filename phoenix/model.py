"""Strict Phoenix restoration and H32/9-frame joint sampling."""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .checkpoint import _load_joint96_checkpoint
from .config import CAMERAS, JOINT_SLOTS, sha256
from .protocol import InferenceInterface


class Normalizer:
    def __init__(self,content,unit):
        if content.get("schema")!="phoenix.joint14.normalization.v1" or content.get("gripper_unit")!=unit:
            raise ValueError("Normalization schema/units differ")
        self.values={}
        for kind in ("state","action"):
            mean=np.asarray(content[kind]["mean"],np.float32)
            std=np.asarray(content[kind]["std"],np.float32)
            if mean.shape!=(14,) or std.shape!=(14,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
                raise ValueError("Expected finite mean/std [14], positive std")
            self.values[kind]=(mean,std)

    def normalize(self,value,kind):
        mean,std=self.values[kind]
        return (value-mean)/std

    def decode(self,value):
        mean,std=self.values["action"]
        return value*std+mean


class Model:
    def __init__(self,model,config,normalizer,context,mask,embodiment_id):
        self.model,self.config,self.normalizer=model,config,normalizer
        self.context,self.mask,self.embodiment_id=context,mask,embodiment_id

    @torch.no_grad()
    def predict(self,state,images,*,prefix):
        from wam.inference.frame_joint import infer_frame_joint_window
        from wam.inference.online_action_policy import RTCActionPrefixCondition
        m,cfg=self.model,self.config
        views=[]
        for key in CAMERAS:
            image=torch.from_numpy(images[key]).permute(2,0,1)[None].float()/255
            views.append(F.interpolate(image,size=(256,256),mode="bicubic",align_corners=False).clamp(0,1)[0]*2-1)
        rgb=torch.stack(views)[None].to(device=m.device,dtype=m.torch_dtype)
        proprio=torch.zeros((1,80),device=m.device,dtype=torch.float32)
        proprio[:,JOINT_SLOTS]=torch.as_tensor(self.normalizer.normalize(state,"state"),device=m.device)
        pad=torch.ones((1,80),dtype=torch.bool,device=m.device)
        pad[:,JOINT_SLOTS]=False
        condition=None
        normalized_prefix=None
        if len(prefix):
            target=torch.zeros((1,32,80),device=m.device,dtype=m.torch_dtype)
            target[:,:len(prefix),JOINT_SLOTS]=torch.as_tensor(self.normalizer.normalize(prefix,"action"),device=m.device,dtype=m.torch_dtype)
            normalized_prefix=target[:,:len(prefix)].float().cpu()
            condition=RTCActionPrefixCondition(target,len(prefix))
        result=infer_frame_joint_window(m,prompt=None,input_image=rgb,num_frames=9,action_horizon=32,
            proprio=proprio,context=self.context,context_mask=self.mask,
            camera_is_pad=torch.zeros((1,3),dtype=torch.bool,device=m.device),
            action_dim_is_pad=pad,proprio_dim_is_pad=pad,
            embodiment_ids=torch.tensor([self.embodiment_id],dtype=torch.long,device=m.device),
            num_inference_steps=10,seed=cfg["seed"],decode_video=False,rtc_prefix_condition=condition)
        if normalized_prefix is not None and not torch.equal(result["action"][:len(prefix)],normalized_prefix[0]):
            raise ValueError("Native denoiser changed the normalized prefix")
        actions=result["action"].float().cpu().numpy()[:,JOINT_SLOTS]
        physical=np.ascontiguousarray(self.normalizer.decode(actions),dtype=np.float32)
        # Restore physical float32 precision only after native prefix verification.
        physical[:len(prefix)]=prefix
        return physical


def load_interface(config,*,check_only=False):
    from wam.utils.config_resolvers import register_default_resolvers
    from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
    from wam.utils.torch24_compat import install_safe_serialization_compat
    import wam.model.factory as factory
    register_default_resolvers()
    install_safe_serialization_compat()
    cfg=copy.deepcopy(config)
    digest=sha256(cfg["checkpoint"])
    stats_digest=sha256(cfg["normalization"])
    if digest!=cfg["model_version"] or stats_digest!=cfg["normalization_sha256"]:
        raise ValueError("Checkpoint/normalization differs from the selected Phoenix assets")
    if sha256(cfg["text_context"])!=cfg["text_context_sha256"]:
        raise ValueError("Text context differs from the selected Phoenix assets")
    normalizer=Normalizer(json.loads(Path(cfg["normalization"]).read_text()),cfg["gripper_unit"])
    payload=torch.load(cfg["checkpoint"],map_location="cpu",mmap=True,weights_only=True)
    if payload.get("training_inputs",{}).get("normalization",stats_digest)!=stats_digest:
        raise ValueError("Inference normalization differs from Training RTC normalization")
    payload=dict(payload)
    payload["joint96_metadata"]=copy.deepcopy(payload["joint96_metadata"])
    metadata=payload["joint96_metadata"]
    if metadata["schema"]=="internw0.local_joint96.weights.v1":
        metadata["schema"]="phoenix.local_joint96.weights.v1"
    contract=metadata.get("soft_prompt",{}).get("training_contract",{})
    from .training import validate_training_contract
    validate_training_contract(contract,cfg)
    if type(payload.get("step")) is not int or payload["step"]<=0:
        raise ValueError("Select a completed Training RTC optimizer checkpoint")
    registry=metadata["soft_prompt"]["registry"]
    names=registry["embodiments"]
    if cfg["embodiment"] not in names:
        raise ValueError("Requested hardware identity is absent from the saved soft prompt")
    embodiment_id=names.index(cfg["embodiment"])
    if registry.get("slot_status",["assigned"]*len(names))[embodiment_id]!="assigned":
        raise ValueError("Requested hardware identity is a reserved prompt slot")
    definition=OmegaConf.load(cfg["model_config"])
    model_cfg=definition.model if "model" in definition else definition
    model_cfg=OmegaConf.create(OmegaConf.to_container(model_cfg,resolve=True))
    rtc=model_cfg.action_dit_config.get("training_time_rtc",{})
    if rtc.get("enabled") is not True or rtc.get("max_delay_steps")!=cfg["rtc"]["max_prefix_steps"]:
        raise ValueError("Model definition does not match the trained RTC contract")
    if not model_cfg.get("joint96_three_view"):
        raise ValueError("Expected the Phoenix Joint96 model")
    model_cfg.joint_visual_codec=bundled_config(OmegaConf.to_container(model_cfg.joint_visual_codec,resolve=True))
    model_cfg.load_text_encoder=False
    model_cfg.mot_attention_backend="sdpa"
    model_cfg.mot_checkpoint_mixed_attn=False
    text=torch.load(cfg["text_context"],map_location="cpu",weights_only=True)
    if text.get("instruction")!=cfg["instruction"]:
        raise ValueError("Text context instruction differs")
    context,mask=text["context"],text["mask"]
    if context.ndim==2:context=context[None]
    if mask.ndim==1:mask=mask[None]
    if context.ndim!=3 or context.shape[0]!=1 or context.shape[-1]!=4096 or mask.shape!=context.shape[:2] or mask.dtype!=torch.bool or not torch.isfinite(context).all():
        raise ValueError("Expected context [1,L,4096] and bool mask [1,L]")
    report=dict(model="phoenix",action_horizon=32,video_frames=9,denoising_steps=10,
        model_version=digest,normalization_sha256=stats_digest,embodiment=cfg["embodiment"],
        checkpoint_step=payload.get("step"),robot_actions_sent=0)
    if check_only:return report
    original=factory.load_wan22_ti2v_5b_components
    def checkpoint_components(**kwargs):
        kwargs["skip_dit_load_from_pretrain"]=True
        return original(**kwargs)
    with tempfile.TemporaryDirectory(prefix="phoenix-model-") as folder:
        registry_path=Path(folder)/"phoenix_embodiments.json"
        registry_path.write_text(json.dumps(registry))
        model_cfg.action_dit_config.embodiment_registry_path=str(registry_path)
        factory.load_wan22_ti2v_5b_components=checkpoint_components
        try:
            model=instantiate(model_cfg,model_dtype=torch.bfloat16,device="cpu")
        finally:
            factory.load_wan22_ti2v_5b_components=original
        _load_joint96_checkpoint(model,payload)
    del payload
    model.to(cfg["device"]).eval().requires_grad_(False)
    model.set_runtime_device(cfg["device"])
    context=context.to(device=cfg["device"],dtype=model.torch_dtype)
    mask=mask.to(cfg["device"])
    return InferenceInterface(Model(model,cfg,normalizer,context,mask,embodiment_id),cfg,
                              model_version=digest,normalization_sha256=stats_digest)
