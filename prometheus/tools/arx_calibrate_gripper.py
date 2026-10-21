from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one ARX5 gripper with an interactive terminal.")
    parser.add_argument("--model", default="X5")
    parser.add_argument("--interface", default="can1")
    parser.add_argument(
        "--init_open_readout",
        type=float,
        default=100.0,
        help=(
            "Temporary permissive value used only to let Arx5JointController initialize. "
            "Use the printed fully-open readout after calibration as the real config value."
        ),
    )
    parser.add_argument("--width", type=float, default=0.082)
    parser.add_argument("--no_background", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("Run this script from an interactive terminal; gripper calibration must wait for Enter.")

    import arx5_interface as arx5

    print(
        f"Preparing {args.model} gripper calibration on {args.interface}.\n"
        f"Temporary controller init config: gripper_open_readout={args.init_open_readout}, "
        f"gripper_width={args.width}.\n"
        "This temporary value is not the final calibration value.\n"
        "Stop all other ARX processes on this CAN interface before continuing.",
        flush=True,
    )
    input("Press Enter only when the robot is safe and the CAN interface is idle...")

    robot_cfg = arx5.RobotConfigFactory.get_instance().get_config(args.model)
    robot_cfg.gripper_open_readout = float(args.init_open_readout)
    robot_cfg.gripper_width = float(args.width)

    ctrl_cfg = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)
    ctrl_cfg.background_send_recv = not args.no_background

    controller = arx5.Arx5JointController(robot_cfg, ctrl_cfg, args.interface)
    print(
        "Controller initialized. Now follow the SDK prompts exactly:\n"
        "1. fully close the gripper, then press Enter;\n"
        "2. fully open the gripper, then press Enter.\n"
        "After it prints 'Fully-open joint position readout', use that value as the real "
        "gripper_open_readout for this CAN arm.",
        flush=True,
    )
    controller.calibrate_gripper()


if __name__ == "__main__":
    main()