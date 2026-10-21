"""Export only a Phoenix model definition from a saved configuration."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-config",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    from wam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    cfg=OmegaConf.load(args.source_config)
    model=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
    model.action_dit_config.embodiment_registry_path=None
    model.action_dit_config.training_time_rtc=dict(enabled=True,max_delay_steps=10)
    model.mot_attention_backend="sdpa"
    model.joint_visual_codec.source_root="bundled://dino-stvae-v1"
    model.joint_visual_codec.source_sha256=None
    model.joint_visual_codec.stvae_checkpoint_path="${oc.env:PHOENIX_STVAE_CHECKPOINT}"
    model.joint_visual_codec.dino_kwargs.checkpoint_path="${oc.env:PHOENIX_ENCODER_CHECKPOINT}"
    for key in ("starvla_source_path","dinov3_source_path"):
        model.joint_visual_codec.dino_kwargs.pop(key,None)
    for key in list(model):
        if key.startswith("init_checkpoint"):
            model[key]=None if key.endswith("path") else model[key]
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:OmegaConf.save(OmegaConf.create(dict(model=model)),stream)
    print(args.output)


if __name__=="__main__":main()
