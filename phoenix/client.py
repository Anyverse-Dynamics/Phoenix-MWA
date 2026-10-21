import argparse
import json
from pathlib import Path

import numpy as np

from .config import read_config,CAMERAS
from .protocol import PhoenixPolicy,FixedChunkPolicy
from prometheus_env.websocket_policy import WebsocketPolicyClient


def observation(raw):
    return np.asarray(raw["robot/qpos"],dtype=np.float32),{key:raw[key] for key in CAMERAS}


def main():
    p=argparse.ArgumentParser(description="Phoenix H32 RTC/sync client")
    p.add_argument("--config",default="configs/phoenix.yaml")
    p.add_argument("--mode",choices=("rtc","sync"),default="rtc")
    p.add_argument("--input",type=Path,help="Recorded NPZ observation; leaves robot disconnected")
    p.add_argument("--output",type=Path)
    p.add_argument("--execute",action="store_true",help="Connect to the configured robot environment")
    p.add_argument("--chunks",type=int,default=3)
    args=p.parse_args()
    if args.chunks<1:p.error("chunks must be positive")
    if args.execute and args.input:p.error("Choose recorded replay or robot execution")
    if not args.execute and (not args.input or not args.output):p.error("Recorded replay requires input and output")
    if args.output and args.output.exists():raise FileExistsError(args.output)
    cfg=read_config(args.config)
    transport=WebsocketPolicyClient(cfg["policy_addr"],inference_timeout_s=300)
    policy=None
    try:
        policy=PhoenixPolicy(transport,cfg)
        if args.input:
            with np.load(args.input,allow_pickle=False) as source:
                state=source["state"].copy()
                images={key:source[key].copy() for key in CAMERAS}
                prefix=source["action_prefix"].copy() if "action_prefix" in source.files else None
            plan=policy.infer(state,images,mode=args.mode,start_tick=0,action_prefix=prefix)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            report=dict(mode=args.mode,horizon=32,video_frames=9,denoising_steps=10,
                guided=plan.guided,roundtrip_s=plan.roundtrip_s,robot_actions_sent=0)
            with args.output.open("xb") as stream:np.savez_compressed(stream,actions=plan.actions,report=json.dumps(report))
            print(json.dumps(report,indent=2))
        else:
            from prometheus_env import connect
            if cfg["gripper_unit"]!="m":raise ValueError("ARX SDK rollout requires verified metre gripper targets")
            input("Phoenix client ready. Press Enter to start the configured robot session: ")
            env=connect(cfg["env_addr"])
            chunks=FixedChunkPolicy(policy,args.mode)
            keys=("robot/qpos",*CAMERAS)
            try:
                rtc=cfg["rtc"]
                watermark=rtc["inference_delay"] if args.mode=="rtc" else 0
                raw,info=env.reset(expect=dict(action_space="abs_qpos",action_dim=14,action_hz=30,
                    action_chunk_horizon=rtc["execution_horizon"]+watermark if watermark else 32,
                    policy_low_watermark=watermark,fixed_chunk_switch=args.mode=="rtc"),obs_keys=keys)
                origin=info.get("sent_action_count",0)
                for _ in range(args.chunks):
                    state,images=observation(raw)
                    actions,plan=chunks.infer(state,images,start_tick=origin)
                    raw,_,terminated,truncated,info=env.step(actions,obs_keys=keys)
                    origin=info["sent_action_count"]
                    print(json.dumps(dict(mode=args.mode,submitted=len(actions),sent_tick=origin,guided=plan.guided,roundtrip_s=plan.roundtrip_s)),flush=True)
                    if terminated or truncated:break
            finally:
                policy.invalidate()
                try:env.prepare_terminal(reason="phoenix_client_exit")
                finally:env.close()
    finally:
        if policy is None:transport.close()
        else:policy.close()


if __name__=="__main__":main()
