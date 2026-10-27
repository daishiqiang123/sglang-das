# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Grouped SWA verify against an independent FP32 attention oracle."""

import pytest
import torch

from sglang.kernels.ops.attention import decode_attention, verify_mla
from sglang.srt.utils import is_gfx938_supported
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=40, suite="stage-b-test-1-hcu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx938_supported(), reason="requires gfx938"
)


@pytest.fixture(autouse=True, params=[False, True])
def kv_layout(monkeypatch, request):
    # Stride extraction and admission checks share the production selector.
    monkeypatch.setattr(verify_mla, "_kv_layout_hcu_fa", request.param)
    monkeypatch.setattr(decode_attention, "_kv_layout_hcu_fa", request.param)


def _inputs(prefixes, dtype):
    torch.manual_seed(20261027)
    batch, length, hq, hkv, dim = len(prefixes), 7, 64, 4, 128
    total = sum(prefixes)
    pages = max(1, (total + 63) // 64)
    kb = torch.randn(pages, 64, hkv, dim, device="cuda").to(dtype)
    vb = torch.randn(pages, 64, hkv, dim, device="cuda").to(dtype)
    if verify_mla._kv_layout_hcu_fa:
        kb = kb.permute(0, 2, 1, 3).contiguous()
        vb = vb.permute(0, 2, 3, 1).contiguous()
    q = torch.randn(batch * length, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * length, hkv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qo = torch.arange(0, (batch + 1) * length, length, device="cuda", dtype=torch.int32)
    kvp = torch.tensor(
        [0, *torch.tensor(prefixes).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    kvi = torch.arange(total, device="cuda", dtype=torch.int64)
    return q, k, v, kb, vb, qo, kvp, kvi


def _oracle(args, prefixes, window):
    q, k, v, kb, vb, _, _, indices = args
    if verify_mla._kv_layout_hcu_fa:
        kb = kb.permute(0, 2, 1, 3).contiguous()
        vb = vb.permute(0, 3, 1, 2).contiguous()
    length = 7
    outputs, offset = [], 0
    for batch, prefix in enumerate(prefixes):
        live = indices[offset : offset + prefix]
        keys = torch.cat(
            (
                kb.reshape(-1, 4, 128)[live].float() * 0.5,
                k[batch * length : (batch + 1) * length].float(),
            )
        )
        values = torch.cat(
            (
                vb.reshape(-1, 4, 128)[live].float() * 0.25,
                v[batch * length : (batch + 1) * length].float(),
            )
        )
        keys, values = (
            keys.repeat_interleave(16, dim=1),
            values.repeat_interleave(16, dim=1),
        )
        queries = q[batch * length : (batch + 1) * length].float().transpose(0, 1)
        scores = queries @ keys.permute(1, 2, 0) * (128**-0.5)
        query_pos = prefix + torch.arange(length, device=q.device)
        key_pos = torch.arange(prefix + length, device=q.device)
        mask = (key_pos[None, :] <= query_pos[:, None]) & (
            query_pos[:, None] <= key_pos[None, :] + window
        )
        probs = scores.masked_fill(~mask[None, :, :], -torch.inf).softmax(-1)
        outputs.append((probs @ values.transpose(0, 1)).transpose(0, 1))
        offset += prefix
    return torch.cat(outputs).to(q.dtype)


def _run(args, output, window):
    q, k, v, kb, vb, qo, kvp, kvi = args
    assert verify_mla.verify_shared_kv_fwd(
        q,
        k,
        v,
        output,
        kb,
        vb,
        qo,
        kvp,
        kvi,
        None,
        True,
        None,
        7,
        0.5,
        0.25,
        sliding_window_size=window,
        page_size=64,
        allow_sliding_window=True,
    )


@pytest.mark.parametrize("prefixes", [(1, 65), (0, 4096), (4096, 17)])
@pytest.mark.parametrize("window", [1, 128, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_swa_matches_fp32_oracle(prefixes, window, dtype):
    args = _inputs(prefixes, dtype)
    output = torch.empty_like(args[0])
    _run(args, output, window)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, _oracle(args, prefixes, window), atol=2e-2, rtol=1e-2
    )


def test_graph_replay_reads_updated_queries_and_prefix_indices():
    prefixes = (4096, 17)
    args = _inputs(prefixes, torch.float8_e4m3fn)
    output = torch.empty_like(args[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(args, output, 4096)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run(args, output, 4096)
        for multiplier in [-0.75, 0.5]:
            args[0].mul_(multiplier)
            args[-1].copy_(args[-1].roll(1))
            graph.replay()
            torch.testing.assert_close(
                output, _oracle(args, prefixes, 4096), atol=2e-2, rtol=1e-2
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
