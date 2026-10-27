# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in MiniMax FlashMLA adapter.

Native numerical/graph tests live in FlashMLA. These cases guard serving
segmentation, quantization-scale ownership, and old-wheel compatibility.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.minimax_sparse_ops import flash_mla_gfx938 as ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-c-test-cpu")


@pytest.mark.parametrize("group", [1, 8])
@pytest.mark.parametrize("live_bounds", [False, True])
def test_indexer_request_segmentation_and_old_stage2(monkeypatch, group, live_bounds):
    calls = {}

    def stage1(q, k, q_lens, kv_lens, **kwargs):
        calls.update(q_lens=q_lens.tolist(), kv_lens=kv_lens.tolist(), stage1=kwargs)
        return torch.zeros((4, 128, q.shape[0]))

    def stage2(score, topk, **kwargs):
        calls["stage2"] = kwargs
        return torch.zeros((score.shape[-1], 4, 16), dtype=torch.int32)

    monkeypatch.setattr(ops, "_load_flash_mla", lambda: (None,) * 4 + (stage1, stage2))
    monkeypatch.setattr(ops, "flash_mla_stage1_graph_available", lambda: True)
    monkeypatch.setattr(ops, "_force_local_pages", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "flash_mla",
        SimpleNamespace(cuda=SimpleNamespace(MSA128_STAGE2_LIVE_BOUNDS=live_bounds)),
    )
    rows = 2 * group
    bounds = torch.arange(rows, dtype=torch.int32) + 257
    result = ops.flash_mla_sparse_decode_indexer(
        torch.empty((rows, 4, 128), dtype=torch.bfloat16),
        torch.empty((512, 1, 128), dtype=torch.bfloat16),
        torch.zeros((2, 128), dtype=torch.int32),
        bounds,
        128,
        16,
        1,
        1,
        verify_group_size=group,
    )
    assert result.shape == (4, rows, 16)
    assert calls["q_lens"] == [group, group]
    assert calls["kv_lens"] == [16384, 16384]
    assert "scheduler_metadata" not in calls["stage1"]
    assert torch.equal(calls["stage1"]["k_end"], bounds)
    assert ("k_end" in calls["stage2"]) is live_bounds


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("group", [1, 8])
def test_verify_final_length_and_scale_applied_once(monkeypatch, dtype, group):
    calls = {}

    def expand(blocks, **kwargs):
        calls["expand"] = kwargs
        return torch.zeros((2, group, 4, 2048), dtype=torch.int32)

    def forward(q, *args, **kwargs):
        calls["main"] = kwargs
        return torch.ones(q.shape, dtype=torch.bfloat16), None

    monkeypatch.setattr(
        ops, "_load_flash_mla", lambda: (None, forward, None, expand, None, None)
    )
    monkeypatch.setattr(ops, "flash_mla_gfx938_available", lambda: True)
    rows = 2 * group
    bounds = torch.cat((torch.arange(group) + 129, torch.arange(group) + 385)).to(
        torch.int32
    )
    output = ops.flash_mla_sparse_decode_main(
        torch.empty((rows, 64, 128), dtype=dtype),
        torch.empty((512, 4, 128), dtype=dtype),
        torch.empty((512, 4, 128), dtype=dtype),
        torch.zeros((4, rows, 16), dtype=torch.int32),
        torch.zeros((2, 4), dtype=torch.int32),
        bounds,
        128,
        object(),
        sm_scale=0.25,
        q_scale=2.0,
        k_scale=3.0,
        v_scale=5.0,
        verify_group_size=group,
    )
    assert calls["expand"]["seqused_k"].tolist() == [128 + group, 384 + group]
    assert calls["expand"]["seqused_k"].is_contiguous()
    if dtype == torch.float8_e4m3fn:
        assert calls["main"]["softmax_scale"] == 0.25
        assert calls["main"]["q_scale"].item() == 2.0
        assert calls["main"]["k_scale"].item() == 3.0
        assert calls["main"]["v_scale"].item() == 5.0
        assert torch.all(output == 1.0)  # Native owns V dequantization.
    else:
        assert calls["main"]["softmax_scale"] == 1.5
        assert "q_scale" not in calls["main"]
        assert torch.all(output == 5.0)


def test_reject_old_stage1_decode_before_native_launch(monkeypatch):
    monkeypatch.setattr(ops, "flash_mla_stage1_graph_available", lambda: False)
    with pytest.raises(ops.FlashMLAGfx938UnavailableError, match="capture-safe"):
        ops.flash_mla_sparse_decode_indexer(None, None, None, None, 128, 16, 1, 1)
