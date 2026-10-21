"""Split SwiGLU OAI with input-dtype rounding and dynamic INT8 quantization."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _swiglu_oai_int8_quant_kernel(
    X,
    Q,
    S,
    Counts,
    H: tl.constexpr,
    M: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = True
    if MASKED:
        valid = row % M < tl.load(Counts + row // M)
    gate = tl.load(X + row * (2 * H) + cols, valid & (cols < H), 0).to(tl.float32)
    up = tl.load(X + row * (2 * H) + H + cols, valid & (cols < H), 0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
    value = gate * tl.sigmoid(ALPHA * gate) * (up + BETA)
    # Match explicit OAI -> BF16/FP16 -> dynamic INT8 quantization.
    value = value.to(X.dtype.element_ty).to(tl.float32)
    value = tl.where(cols < H, value, 0.0)
    amax = tl.maximum(tl.max(tl.abs(value), 0), 1e-10)
    scale = tl.where(valid, amax / 127.0, 1.0)
    quant = libdevice.nearbyint(tl.div_rn(value, scale))
    quant = tl.maximum(tl.minimum(quant, 127.0), -127.0)
    tl.store(Q + row * H + cols, quant.to(tl.int8), cols < H)
    tl.store(S + row, scale)


def swiglu_oai_int8_quant(input, alpha, beta=1.0, limit=None, tokens_per_expert=None):
    """Return signed INT8 [..,H] and FP32 per-token scales [..,1].

    Input uses split gate/up halves. Optional counts mask [experts,capacity,2H]
    tensors with counts in [0,capacity]; padding is zero with unit scale.
    """
    if input.ndim not in (2, 3) or input.shape[-1] % 2 or not input.is_contiguous():
        raise ValueError("Expected contiguous [tokens,2H] or [experts,tokens,2H]")
    if input.shape[-1] == 0:
        raise ValueError("Hidden dimension must be positive")
    if not input.is_cuda or input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Expected floating CUDA/HIP input")
    hidden, capacity = input.shape[-1] // 2, input.shape[-2]
    if tokens_per_expert is not None:
        if (
            input.ndim != 3
            or tokens_per_expert.shape != (input.shape[0],)
            or tokens_per_expert.dtype != torch.int32
            or tokens_per_expert.device != input.device
            or not tokens_per_expert.is_contiguous()
        ):
            raise ValueError("Expert counts must be contiguous device int32 [experts]")
    if limit is not None and limit <= 0:
        raise ValueError("Clamp limit must be positive")
    quant = torch.empty(
        (*input.shape[:-1], hidden), dtype=torch.int8, device=input.device
    )
    scales = torch.empty(
        (*input.shape[:-1], 1), dtype=torch.float32, device=input.device
    )
    rows = input.numel() // input.shape[-1]
    if rows:
        _swiglu_oai_int8_quant_kernel[(rows,)](
            input,
            quant,
            scales,
            tokens_per_expert,
            hidden,
            capacity,
            float(alpha),
            float(beta),
            float(limit or 0),
            limit is not None,
            tokens_per_expert is not None,
            triton.next_power_of_2(hidden),
            num_warps=4,
        )
    return quant, scales
