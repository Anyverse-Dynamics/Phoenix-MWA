"""Small real DiT fixture, extracted from upstream test_soft_prompt.policy."""
from pathlib import Path
from test_joint96_mask_loss import make_model
from wam.model.modules.experts.action_dit import ActionDiT


def policy(enabled=True, *, checkpoint=False):
    model = make_model(checkpoint=checkpoint)
    if enabled:
        expert = ActionDiT(hidden_dim=24, action_dim=80, ffn_dim=48, text_dim=16,
            freq_dim=16, eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=2,
            num_embodiments=11, soft_prompt_tokens=8, use_gradient_checkpointing=checkpoint,
            embodiment_registry_path=str(Path(__file__).resolve().parents[1]/'configs/embodiment_registry/local_robot11_v1.json'))
        model.mot.mixtures['action'] = expert
        model.action_expert = expert
    model.model_paths = {'joint_visual_encoder': {'source_sha256': 'encoder-code', 'stvae': {
        'stvae_step': 100000, 'teacher_sha256': 'teacher', 'stvae_checkpoint_sha256': 'stvae'}}}
    model.action_proprio_encoder = model.understanding = None
    return model
