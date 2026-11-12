"""Store MiniMax CP8 FP8 caches directly from a BF16 AllGather payload."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cast_main_kv(value, FP8: tl.constexpr):
    clamped = tl.clamp(
        value.to(tl.float32), -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL
    )
    return clamped.to(FP8, fp_downcast_rounding="rtne")


@triton.jit
def _cp_fp8_cache_scatter_kernel(
    raw_ptr,
    reorder_ptr,
    slot_ptr,
    k_cache_ptr,
    v_cache_ptr,
    index_cache_ptr,
    num_rows,
    raw_rows,
    cache_rows,
    raw_row_stride,
    k_row_stride,
    v_row_stride,
    index_row_stride,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    src = tl.load(reorder_ptr + row, mask=row < num_rows, other=-1).to(tl.int64)
    slot = tl.load(slot_ptr + row, mask=row < num_rows, other=-1).to(tl.int64)
    valid = (
        (row < num_rows)
        & (src >= 0)
        & (src < raw_rows)
        & (slot >= 0)
        & (slot < cache_rows)
    )
    # Main KV reserves slot zero. The K-only index pool accepts it.
    main_mask = (valid & (slot > 0))[:, None]
    index_mask = valid[:, None]
    source = raw_ptr + src[:, None] * raw_row_stride

    kv_cols = tl.arange(0, 512)[None, :]
    k = tl.load(source + kv_cols, mask=main_mask, other=0)
    tl.store(
        k_cache_ptr + slot[:, None] * k_row_stride + kv_cols,
        _cast_main_kv(k, k_cache_ptr.dtype.element_ty),
        mask=main_mask,
    )
    v = tl.load(source + 512 + kv_cols, mask=main_mask, other=0)
    tl.store(
        v_cache_ptr + slot[:, None] * v_row_stride + kv_cols,
        _cast_main_kv(v, v_cache_ptr.dtype.element_ty),
        mask=main_mask,
    )

    index_cols = tl.arange(0, 128)[None, :]
    index_k = tl.load(source + 1024 + index_cols, mask=index_mask, other=0)
    # The index pool uses the plain E4M3FN cast for unit scale.
    tl.store(
        index_cache_ptr + slot[:, None] * index_row_stride + index_cols,
        index_k.to(index_cache_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=index_mask,
    )


def minimax_cp_fp8_cache_scatter(
    raw: torch.Tensor,
    reorder: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    index_cache: torch.Tensor,
    *,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    index_k_scale: Optional[float] = None,
) -> None:
    """Scatter rank-major BF16 K512/V512/index-K128 into E4M3FN caches.

    Valid destination slots must be unique, as supplied by the CP planner.
    Callers select the unit-scale, token-major cache path before gathering.
    The gather result and the scatter are consumed on the same stream.
    """
    if any(scale not in (None, 1.0) for scale in (k_scale, v_scale, index_k_scale)):
        raise ValueError("MiniMax CP FP8 scatter requires unit scales")
    if (
        not raw.is_cuda
        or raw.dtype != torch.bfloat16
        or raw.ndim != 2
        or raw.shape[1] != 1152
        or raw.stride(1) != 1
        or raw.stride(0) < 1152
    ):
        raise ValueError("Expected a device BF16 payload with 1152 columns")
    if any(
        t.device != raw.device
        or t.dtype != torch.int64
        or t.ndim != 1
        or not t.is_contiguous()
        for t in (reorder, slots)
    ) or reorder.numel() != slots.numel():
        raise ValueError("Expected matching contiguous device int64 row and slot maps")
    caches = (k_cache, v_cache, index_cache)
    for cache, heads in zip(caches, (4, 4, 1)):
        if (
            cache.device != raw.device
            or cache.dtype != torch.float8_e4m3fn
            or cache.ndim != 3
            or cache.shape[1:] != (heads, 128)
            or cache.stride(2) != 1
            or cache.stride(1) != 128
            or cache.stride(0) < heads * 128
            or cache.shape[0] != k_cache.shape[0]
        ):
            raise ValueError("Expected token-major E4M3FN caches with equal capacity")
    if slots.numel() == 0:
        return

    block_rows = 4
    with torch.cuda.device(raw.device):
        _cp_fp8_cache_scatter_kernel[(triton.cdiv(slots.numel(), block_rows),)](
            raw,
            reorder,
            slots,
            *caches,
            slots.numel(),
            raw.shape[0],
            k_cache.shape[0],
            raw.stride(0),
            *(cache.stride(0) for cache in caches),
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
