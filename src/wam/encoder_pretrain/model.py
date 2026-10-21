"""Training contract around the encoder runtime shared with deployment.

The bundled runtime implements the dense DINO mask and curriculum objective
of the source recipe. Keeping one implementation preserves inference parity.
"""
import torch

from wam.encoder.multi_teacher_distillation import (
    MultiTeacherDistillationModel, SpatialBlockMaskGenerator, StudentVideoEncoder,
    VJEPAObjective, VJEPAPredictor,
)


class EncoderPretrainModel(MultiTeacherDistillationModel):
    resume_checkpoint_format = "phoenix.encoder_pretrain.resume.v1"

    @torch.no_grad()
    def update_ema(self):
        super().update_ema()
        # These buffers are reconstructed, not saved. Averaging constants can
        # introduce rounding drift that disappears on export or resume.
        online_modules = dict(self.student.named_modules())
        for name, ema_module in self.ema_student.named_modules():
            online_module = online_modules[name]
            for buffer_name in online_module._non_persistent_buffers_set:
                buffer = online_module._buffers[buffer_name]
                if buffer is not None:
                    ema_module._buffers[buffer_name].copy_(buffer)

    def forward(self, batch, teacher_type, global_step):
        if teacher_type not in {"vggt", "vjepa"}:
            raise ValueError(f"Unknown teacher: {teacher_type}")
        return super().forward(batch, teacher_type, global_step)
