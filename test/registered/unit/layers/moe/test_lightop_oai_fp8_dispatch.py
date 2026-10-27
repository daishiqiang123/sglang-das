# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The optional LightOp path must preserve old callers and old wheels."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import ep_moe_kernels as kernels
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class _Kernel:
    def __init__(self, calls):
        self.calls = calls

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, kwargs))


@pytest.mark.parametrize(
    "enabled,hcu,dtype,alpha,limit,swiglu_limit,ue8m0,expected",
    [
        (False, True, torch.bfloat16, 1.702, 7, None, False, "triton"),
        (True, False, torch.bfloat16, 1.702, 7, None, False, "triton"),
        (True, True, torch.float16, 1.702, 7, None, False, "triton"),
        (True, True, torch.bfloat16, 0, 0, None, False, "triton"),
        (True, True, torch.bfloat16, 1.702, 7, 10, False, "triton"),
        (True, True, torch.bfloat16, 1.702, 7, None, True, "triton"),
        (True, True, torch.bfloat16, 1.702, 7, None, False, "lightop"),
    ],
)
@pytest.mark.parametrize("experts", [4, 16])
def test_optional_dispatch_preserves_other_contracts(
    monkeypatch,
    enabled,
    hcu,
    dtype,
    alpha,
    limit,
    swiglu_limit,
    ue8m0,
    expected,
    experts,
):
    monkeypatch.setenv("SGLANG_USE_LIGHTOP_OAI_FP8_QUANT", str(int(enabled)))
    monkeypatch.setattr(kernels, "_is_hcu", hcu)
    native, fallback, loads = [], [], []
    monkeypatch.setattr(
        kernels, "_silu_and_mul_post_quant_single_group_kernel", _Kernel(fallback)
    )

    def loader(flat):
        loads.append(flat)
        return lambda *args: native.append(args)

    monkeypatch.setattr(kernels, "_get_lightop_oai_fp8_quant", loader)
    x = torch.empty((experts, 2, 6144), dtype=dtype)
    q = torch.empty((experts, 2, 3072), dtype=torch.float8_e4m3fn)
    scales = torch.empty((experts, 2, 1), dtype=torch.float32)
    counts = torch.zeros(experts, dtype=torch.int32)
    kernels.silu_and_mul_masked_post_quant_fwd(
        x,
        q,
        scales,
        3072,
        counts,
        scale_ue8m0=ue8m0,
        gemm1_alpha=alpha,
        gemm1_clamp_limit=limit,
        swiglu_limit=swiglu_limit,
    )
    if expected == "lightop":
        assert len(native) == 1 and not fallback
        assert loads == [experts == 16]
        assert native[0][0] is x and native[0][3] is counts
    else:
        assert len(fallback) == 1 and not native and not loads
        assert fallback[0][1]["GEMM1_ALPHA"] == alpha
        assert fallback[0][1]["APPLY_SWIGLU_LIMIT"] is (swiglu_limit is not None)


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_USE_LIGHTOP_OAI_FP8_QUANT", raising=False)
    assert envs.SGLANG_USE_LIGHTOP_OAI_FP8_QUANT.get() is False


@pytest.mark.parametrize("flat", [False, True])
def test_old_lightop_wheel_falls_back(monkeypatch, flat):
    old = ModuleType("lightop")
    old.op = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lightop", old)
    kernels._get_lightop_oai_fp8_quant.cache_clear()
    try:
        assert kernels._get_lightop_oai_fp8_quant(flat) is None
    finally:
        kernels._get_lightop_oai_fp8_quant.cache_clear()


def test_missing_native_symbol_runs_existing_triton(monkeypatch):
    monkeypatch.setenv("SGLANG_USE_LIGHTOP_OAI_FP8_QUANT", "1")
    monkeypatch.setattr(kernels, "_is_hcu", True)
    monkeypatch.setattr(kernels, "_get_lightop_oai_fp8_quant", lambda flat: None)
    calls = []
    monkeypatch.setattr(
        kernels, "_silu_and_mul_post_quant_single_group_kernel", _Kernel(calls)
    )
    kernels.silu_and_mul_masked_post_quant_fwd(
        torch.empty((4, 2, 6144), dtype=torch.bfloat16),
        torch.empty((4, 2, 3072), dtype=torch.float8_e4m3fn),
        torch.empty((4, 2, 1), dtype=torch.float32),
        3072,
        torch.zeros(4, dtype=torch.int32),
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7,
    )
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
