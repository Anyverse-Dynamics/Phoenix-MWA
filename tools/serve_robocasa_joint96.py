"""PPU-side server using the existing RoboCasa msgpack/websocket transport."""
import argparse
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


class ServerAdapter:
    def __init__(self, policy):
        self.policy = policy

    def predict_action(self, observations=None, reset=False, task_description=None, **kwargs):
        if reset:
            self.policy.reset(task_description)
            return {"reset": True}
        if observations is None:
            raise ValueError("RoboCasa server requires named raw observations")
        return self.policy.step(observations)


def main():
    from wam.inference.robocasa_resources import DEFAULT_PACKAGE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--package", default=str(DEFAULT_PACKAGE))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=39100)
    p.add_argument("--execute-steps", type=int, default=12)
    p.add_argument("--denoising-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--idle-timeout", type=int, default=1800)
    p.add_argument("--sim-root", default=str(Path(__file__).resolve().parents[1] / "third_party/robocasa_runner"))
    a = p.parse_args()
    sys.path.append(str(Path(a.sim_root).resolve(strict=True)))
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from wam.inference.robocasa_joint96 import RoboCasaJoint96Policy
    policy = RoboCasaJoint96Policy.from_checkpoint(a.checkpoint, a.config, a.package, device=a.device,
        execute_steps=a.execute_steps, num_inference_steps=a.denoising_steps, seed=a.seed)
    server = WebsocketPolicyServer(policy=ServerAdapter(policy), host=a.host, port=a.port, idle_timeout=a.idle_timeout,
        metadata={"policy": f"robocasa_{policy.action_space}_joint_v1", "embodiment_id": policy.embodiment_id,
                  "action_dim": policy.action_dim, "execute_steps": a.execute_steps,
                  "inference_mode": "joint_video_actions", "seed": a.seed, "seed_mode": "fixed_per_replan",
                  "text_mode": "online_t5_runtime_cache", "text_context_len": 128,
                  "denoising_steps": a.denoising_steps, "checkpoint": str(Path(a.checkpoint).resolve()),
                  "package": str(Path(a.package).resolve()),
                  "checkpoint_progress": policy.checkpoint_progress})
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
