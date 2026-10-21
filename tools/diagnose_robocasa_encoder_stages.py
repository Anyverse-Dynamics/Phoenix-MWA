"""Compare an unchanged PPU reference to the standalone NVIDIA frozen encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from wam.datasets.robocasa_visual_cache import atomic_json, decode_episode, load_plan, pack_rgb, window_mapping
from wam.model.backbones.wan22.helpers.loader import _load_registered_model
from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder


def stats(a,b):
    a,b=a.float().flatten(),b.float().flatten()
    return {'max_abs':float((a-b).abs().max()), 'rmse':float((a-b).square().mean().sqrt()),
            'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
            'reference_rms':float(b.square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser()
    for name in ('reference','config','vae','plan','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    cfg=OmegaConf.to_container(OmegaConf.load(a.config).model.joint_visual_codec,resolve=True)
    vae=_load_registered_model(a.vae,'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda:0')
    encoder=JointVisualEncoder.from_config(cfg,vae=vae,device='cuda:0')
    rgb=ref['rgb'].to('cuda:0')
    transport=rgb.to(torch.bfloat16).permute(0,2,3,4,1,5).reshape(1,3,9,256,768).contiguous()
    plan=load_plan(a.plan)
    ep=next(e for e in plan['episodes'] if e['task']=='gr1_unified.PnPBottleToCabinetClose' and e['episode']==0)
    images,_=decode_episode(ep)
    decoded=pack_rgb(images,window_mapping(ep['length'],0))
    report={'rgb_bitwise_equal':torch.equal(decoded,ref['rgb']),
            'rgb_sha256':hashlib.sha256(decoded.numpy().tobytes()).hexdigest(),
            'device':torch.cuda.get_device_name(),'torch':str(torch.__version__)}
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):
        direct_dino=encoder.stvae_encoder.dino.encode_raw_video_latents(transport,num_views=3).cpu()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            autocast_dino=encoder.stvae_encoder.dino.encode_raw_video_latents(transport,num_views=3).cpu()
        pad=torch.tensor([[False,True,True]],device='cuda:0')
        clip=encoder.encode(rgb,camera_is_pad=pad)[0,...,:16].cpu()
        original=encoder.stvae_encoder.dino.encode_raw_video_latents
        try:
            encoder.stvae_encoder.dino.encode_raw_video_latents=lambda *args,**kwargs:ref['dino'].to('cuda:0')
            forced=encoder.encode(rgb,camera_is_pad=pad)[0,...,:16].cpu()
        finally:
            encoder.stvae_encoder.dino.encode_raw_video_latents=original
        # Diagnostic only: PyTorch releases differ in math-SDPA intermediates.
        # Never silently apply this setting to a production model.
        variants={}
        flag=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        try:
            for reduced in (False,True):
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduced)
                result=encoder.encode(rgb,camera_is_pad=pad)[0,...,:16].cpu()
                variants[str(reduced)]=stats(result[48:],ref['clip'][48:])
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(flag)
    report.update(dino=stats(autocast_dino,ref['dino']),
                  direct_vs_autocast_dino=stats(direct_dino,autocast_dino),
                  vae=stats(clip[:48],ref['clip'][:48]),
                  stvae=stats(clip[48:],ref['clip'][48:]),
                  stvae_with_ppu_dino=stats(forced[48:],ref['clip'][48:]),
                  diagnostic_math_sdpa_reduced_precision=variants)
    atomic_json(a.output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
