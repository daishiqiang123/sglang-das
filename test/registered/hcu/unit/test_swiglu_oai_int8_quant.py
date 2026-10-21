"""Split SwiGLU-OAI INT8 quantization, padding and graph replay contracts."""

import pytest
import torch

from sglang.kernels.ops.activation.swiglu_oai_int8_quant import swiglu_oai_int8_quant
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


def reference(x, counts, alpha, beta, limit):
    gate, up = x.float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    value = (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(x.dtype).float()
    scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    q = (value / scale).round().clamp(-127, 127).to(torch.int8)
    if counts is not None:
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < counts[:, None]
        q = torch.where(valid[..., None], q, 0)
        scale = torch.where(valid[..., None], scale, 1.0)
    return q, scale


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [None, 7.0])
def test_split_oai_matches_reference(masked, dtype, limit):
    torch.manual_seed(121)
    shape = (3, 5, 384) if masked else (13, 384)
    x = (torch.randn(shape, device="cuda") * 9).to(dtype)
    counts = (
        torch.tensor([5, 2, 0], device="cuda", dtype=torch.int32) if masked else None
    )
    got, scale = swiglu_oai_int8_quant(x, 1.702, 1.0, limit, counts)
    expected, expected_scale = reference(x, counts, 1.702, 1.0, limit)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-4, atol=1e-6)
    assert (got.to(torch.int16) - expected.to(torch.int16)).abs().max().item() <= 1
    if masked:
        assert not got[2].any()
        assert torch.all(scale[2] == 1)


def test_split_oai_bias_and_graph_replay():
    x = torch.empty((2, 4, 128), device="cuda", dtype=torch.bfloat16)
    x[..., :64] = 1
    x[..., 64:] = 0
    counts = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    swiglu_oai_int8_quant(x, 1.702, 1.0, 7.0, counts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, scale = swiglu_oai_int8_quant(x, 1.702, 1.0, 7.0, counts)
    graph.replay()
    assert (got[0].float() * scale[0]).mean().item() > 0.8
    for count in [1, 3]:
        x.normal_()
        counts.fill_(count)
        graph.replay()
        expected, expected_scale = reference(x, counts, 1.702, 1.0, 7.0)
        torch.testing.assert_close(scale, expected_scale, rtol=1e-4, atol=1e-6)
        assert (got.to(torch.int16) - expected.to(torch.int16)).abs().max().item() <= 1
