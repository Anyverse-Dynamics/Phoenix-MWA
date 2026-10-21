"""Construct the reviewed DINOv3-L/16 + adapter training recipe."""

import torch

from wam.encoder.external_sources import SourceContract, verify_source_contract
from wam.encoder.multi_teacher_distillation import (
    DINOOnlyVideoEncoder,
    LocalVGGTBackend,
    LocalVJEPABackend,
    SharedSpatiotemporalAdapter,
    VGGTObjective,
)
from wam.encoder.multi_teacher_distillation.factories import (
    build_dinov3_large,
    build_vggt_omega,
    build_vjepa2_1_large,
)
from .model import (
    EncoderPretrainModel,
    SpatialBlockMaskGenerator,
    StudentVideoEncoder,
    VJEPAObjective,
    VJEPAPredictor,
)
from .config import bundled_dino_root, verify_bundled_dino


def build_model(config):
    cfg = config.framework.multi_teacher_distillation
    mode = str(cfg.mode)
    max_steps = int(config.trainer.max_train_steps)

    dino_source = SourceContract(
        path=str(bundled_dino_root()),
        package="dinov3",
        expected_commit=None,
    )
    dino_checkpoint_prefix = cfg.dino.get("checkpoint_prefix")
    provenance = {
        "dino_source_commit": verify_bundled_dino(str(cfg.dino.source_commit)),
        "dino_checkpoint": str(cfg.dino.checkpoint),
    }
    if dino_checkpoint_prefix is not None:
        provenance["dino_checkpoint_prefix"] = str(dino_checkpoint_prefix)
    dino = build_dinov3_large(
        dino_source,
        cfg.dino.checkpoint,
        checkpoint_prefix=dino_checkpoint_prefix,
    )

    vggt_backend = None
    vggt_objective = None
    vjepa_backend = None
    vjepa_objective = None
    mask_generator = None

    if mode == "dino_only":
        student = DINOOnlyVideoEncoder(dino)
    else:
        adapter_cfg = cfg.adapter
        use_activation_checkpointing = bool(config.trainer.activation_checkpointing)
        adapter = SharedSpatiotemporalAdapter(
            width=int(adapter_cfg.width),
            depth=int(adapter_cfg.depth),
            num_heads=int(adapter_cfg.num_heads),
            mlp_ratio=float(adapter_cfg.mlp_ratio),
            num_register_tokens=int(adapter_cfg.num_register_tokens),
            supervision_taps=adapter_cfg.supervision_taps,
            use_activation_checkpointing=use_activation_checkpointing,
            layer_scale_init=None,
        )
        student = StudentVideoEncoder(
            dino,
            adapter,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    if mode in {"vggt_only", "joint"}:
        vggt_source = SourceContract(
            path=str(cfg.vggt.source_path),
            package="vggt_omega",
            expected_commit=str(cfg.vggt.source_commit),
        )
        provenance["vggt_source_commit"] = verify_source_contract(vggt_source)["commit"]
        provenance["vggt_checkpoint"] = str(cfg.vggt.checkpoint)
        teacher = build_vggt_omega(vggt_source, cfg.vggt.checkpoint)
        teacher.aggregator.to(dtype=torch.bfloat16)
        vggt_backend = LocalVGGTBackend(teacher, frames_chunk_size=int(cfg.vggt.frames_chunk_size))
        vggt_objective = VGGTObjective(adapter_dim=int(cfg.adapter.width))

    if mode in {"vjepa_only", "joint"}:
        vjepa_source = SourceContract(
            path=str(cfg.vjepa.source_path),
            package="src",
            expected_commit=str(cfg.vjepa.source_commit),
        )
        provenance["vjepa_source_commit"] = verify_source_contract(vjepa_source)["commit"]
        provenance["vjepa_checkpoint"] = str(cfg.vjepa.checkpoint)
        teacher = build_vjepa2_1_large(vjepa_source, cfg.vjepa.checkpoint).to(dtype=torch.bfloat16)
        vjepa_backend = LocalVJEPABackend(teacher)
        predictor = VJEPAPredictor(
            width=int(cfg.vjepa.predictor_width),
            depth=int(cfg.vjepa.predictor_depth),
            num_heads=int(cfg.vjepa.predictor_num_heads),
        )
        vjepa_objective = VJEPAObjective(predictor)
        total_vjepa_updates = max_steps if mode == "vjepa_only" else max_steps // 2
        mask_generator = SpatialBlockMaskGenerator(
            curriculum_steps=round(total_vjepa_updates * float(cfg.vjepa.mask_curriculum_fraction)),
            start_union_min=float(cfg.vjepa.mask_start_union_min),
            start_union_max=float(cfg.vjepa.mask_start_union_max),
            union_min=float(cfg.vjepa.mask_union_min),
            union_max=float(cfg.vjepa.mask_union_max),
            small_area_start=float(cfg.vjepa.mask_small_area_start),
            small_area_end=float(cfg.vjepa.mask_small_area_end),
            large_area_start=float(cfg.vjepa.mask_large_area_start),
            large_area_end=float(cfg.vjepa.mask_large_area_end),
        )

    return EncoderPretrainModel(
        student=student,
        mode=mode,
        max_steps=max_steps,
        vggt_backend=vggt_backend,
        vggt_objective=vggt_objective,
        vjepa_backend=vjepa_backend,
        vjepa_objective=vjepa_objective,
        mask_generator=mask_generator,
        final_relation_weight_max=float(cfg.vggt.final_relation_weight),
        final_relation_warmup_fraction=float(cfg.vggt.final_relation_warmup_fraction),
        context_warmup_fraction=float(cfg.vjepa.context_weight_warmup_fraction),
        context_weight_max=float(cfg.vjepa.context_weight_max),
        ema_momentum=float(cfg.ema_momentum),
        provenance=provenance,
    )
