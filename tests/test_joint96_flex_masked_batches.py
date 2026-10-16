"""Exact padding semantics for all-Flex eager Joint96 attention."""
import copy
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from wam.model.modules.conditioning.frame_attention import build_joint96_attention_mask
from wam.model.modules.mot import mixture_of_transformers as mot_module
from test_joint96_mask_loss import make_model, make_sample, run_model


OPTIONS = dict(fwd_BLOCK_M=64, fwd_BLOCK_N=64, fwd_num_stages=2,
               bwd_BLOCK_M1=64, bwd_BLOCK_N1=64, bwd_BLOCK_M2=64, bwd_BLOCK_N2=64, bwd_num_stages=2)


def make_mot(*, checkpoint=False, **kwargs):
    core = make_model(checkpoint=checkpoint)
    return mot_module.MoT(
        core.mot.mixtures, attention_backend="flex", flex_masked_batches=True,
        mot_checkpoint_mixed_attn=checkpoint, flex_training_kernel_options=OPTIONS, **kwargs,
    )


def dense_mask(block, batch, seq, device):
    b = torch.arange(batch, device=device)[:, None, None]
    q = torch.arange(seq, device=device)[None, :, None]
    k = torch.arange(seq, device=device)[None, None, :]
    return block.mask_mod(b, torch.zeros_like(b), q, k)


@pytest.mark.parametrize("kind", ["all_valid", "mixed", "query_only", "key_only", "empty", "batched"])
def test_blockmask_matches_sdpa_and_preserves_invalid_query_self_guard(kind):
    with patch.object(torch.cuda, "is_available", return_value=False):
        mot = make_mot()
    seq, batch = 22, 2
    structural = build_joint96_attention_mask(video_seq_len=18, action_seq_len=4,
                                               video_tokens_per_frame=6, device="cpu")
    key = torch.ones(batch, seq, dtype=torch.bool)
    query = key.clone()
    if kind in ("mixed", "key_only", "query_only"):
        key[0, [2, 3, 8, 9, 14, 15]] = False
        query.copy_(key)
        query[1, -1] = False
    if kind == "query_only":
        key = None
    if kind == "key_only":
        query = None
    if kind == "empty":
        key[0] = False
        query[0] = False
    if kind == "batched":
        structural = structural[None, None].expand(batch, 1, -1, -1).clone()
        structural[1, :, 6:18, :6] = False
    expected = mot._apply_query_mask(mot._apply_key_mask(structural, key), query)
    block = mot._build_flex_block_mask(attention_mask=expected, batch_size=batch,
                                      query_len=seq, key_len=seq, device=torch.device("cpu"))
    actual = dense_mask(block, batch, seq, "cpu")
    torch.testing.assert_close(actual, expected.expand(batch, 1, seq, seq)[:, 0])
    if kind in ("mixed", "query_only", "empty"):
        rows = actual[~query]
        assert (rows.sum(-1) == 1).all()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_cpu_actual_flex_loss_gradients_and_padding_invariance(checkpoint):
    with patch.object(torch.cuda, "is_available", return_value=False):
        reference = make_model(checkpoint=checkpoint)
        flex = make_model(checkpoint=checkpoint)
        flex.mot = make_mot(checkpoint=checkpoint)
        # Use the same expert objects for the heads and the MoT blocks.
        flex.video_expert, flex.action_expert = flex.mot.mixtures["video"], flex.mot.mixtures["action"]
    reference.mot.load_state_dict(flex.mot.state_dict())
    reference.proprio_encoder.load_state_dict(flex.proprio_encoder.state_dict())
    sample = make_sample()
    changed = copy.deepcopy(sample)
    changed["video_latents"][0, :, :, :, 16:32] = 999.
    results = []
    for model, batch in ((reference, sample), (flex, sample), (flex, changed)):
        model.mot.zero_grad(set_to_none=True)
        batch = copy.deepcopy(batch)
        batch["video_latents"].requires_grad_()
        guard = patch.object(mot_module, "flash_attention", side_effect=AssertionError("SDPA fallback"))
        if model is flex:
            guard.start()
        try:
            loss, _ = run_model(model, batch)
            loss.backward()
        finally:
            if model is flex:
                guard.stop()
        results.append((loss.detach(), {name: p.grad.clone() for name, p in model.mot.named_parameters() if p.grad is not None}))
        assert not batch["video_latents"].grad[0, :, :, :, 16:32].any()
    for loss, grads in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=2e-5, atol=2e-6)
        assert grads.keys() == results[0][1].keys()
        for name, value in grads.items():
            torch.testing.assert_close(value, results[0][1][name], rtol=3e-4, atol=2e-6, msg=name)


def test_flag_rejects_incompatible_backend_or_layer_compile_and_preserves_state_dict():
    with patch.object(torch.cuda, "is_available", return_value=False):
        core = make_model()
        assert set(make_mot().state_dict()) == set(core.mot.state_dict())
        for kwargs in (dict(attention_backend="sdpa"), dict(attention_backend="flex", compile_mode="default")):
            with pytest.raises(ValueError, match="flex_masked_batches"):
                mot_module.MoT(core.mot.mixtures, flex_masked_batches=True,
                               mot_checkpoint_mixed_attn=False, **kwargs)


@pytest.mark.skipif(os.environ.get("WAM_TEST_FLEX_CUDA") != "1", reason="Explicit isolated CUDA validation only")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_cuda_real_production_attention_matches_sdpa(checkpoint):
    torch.manual_seed(911)
    mot = make_mot(checkpoint=checkpoint).cuda()
    mot.num_heads, mot.attn_head_dim = 24, 128
    structural = build_joint96_attention_mask(video_seq_len=576, action_seq_len=32,
                                               video_tokens_per_frame=192, device="cuda")
    valid = torch.ones(16, 608, device="cuda", dtype=torch.bool)
    pos = torch.arange(576, device="cuda")
    valid[:4, :576] &= ~(((pos % 24) >= 8) & ((pos % 24) < 16))
    valid[4, 605:] = False
    effective = mot._apply_query_mask(mot._apply_key_mask(structural, valid), valid)
    qkv = [torch.randn(16, 608, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    upstream = torch.randn_like(qkv[0]).masked_fill(~valid[:, :, None], 0)
    expected = F.scaled_dot_product_attention(
        *[t.reshape(16, 608, 24, 128).transpose(1, 2) for t in qkv], attn_mask=effective,
    ).transpose(1, 2).reshape(16, 608, 3072)
    expected.backward(upstream)
    grads = [t.grad.clone() for t in qkv]
    for t in qkv:
        t.grad = None
    with patch.object(mot_module, "flash_attention", side_effect=AssertionError("SDPA fallback")):
        actual = mot._mixed_attention(*qkv, attention_mask=structural, flex_key_mask=valid, query_mask=valid)
        actual.backward(upstream)
    for actual_tensor, reference in zip([actual, *[t.grad for t in qkv]], [expected, *grads]):
        assert torch.isfinite(actual_tensor).all()
        torch.testing.assert_close(actual_tensor, reference, rtol=.05, atol=.02)
    for tensor in qkv:
        assert not tensor.grad[~valid].any()
