"""Explicit preflight/train entry point for the Phoenix visual encoder."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_intermixed_args()
    from omegaconf import OmegaConf
    from .config import preflight
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    report = preflight(cfg)
    if args.command == "preflight":
        print(json.dumps(report, indent=2))
        return

    from accelerate.utils import set_seed
    from .build import build_model
    from .data.multiview import make_shared_auto_multiview_ego_video_dataloader
    from .trainer import build_accelerator, build_optimizer, build_scheduler, setup_directories
    from .multiview_trainer import StratifiedViewMultiTeacherDistillationTrainer

    accelerator = build_accelerator(cfg.trainer)
    # The same initial model on every rank; rank-specific data RNG is set in prepare().
    set_seed(int(cfg.seed))
    cfg.output_dir = str(Path(cfg.run_root_dir) / str(cfg.run_id))
    # All ranks validate the run directory before any rank writes it.
    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.trainer.get("resume_from"):
        raise FileExistsError(f"Nonempty run directory requires trainer.resume_from: {output_dir}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        setup_directories(cfg)
    accelerator.wait_for_everyone()
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg.trainer)
    scheduler = build_scheduler(optimizer, cfg.trainer)
    loaders = make_shared_auto_multiview_ego_video_dataloader(cfg, world_size=accelerator.num_processes)
    trainer = StratifiedViewMultiTeacherDistillationTrainer(
        cfg, model, loaders, optimizer, scheduler, accelerator, output_dir,
    )
    trainer.prepare()
    trainer.train()


if __name__ == "__main__":
    main()
