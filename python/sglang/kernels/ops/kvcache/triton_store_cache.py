from typing import Literal, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)

# DeepSeek-V4 MLA paged FP8 cache layout
_MLA_HEAD_DIM = 512  # full MLA token dim (elements per input row)
_MLA_NOPE_DIM = 448  # nope sub-dim (elements)
_MLA_TILE_SIZE = 64  # FP8 tile width (also rope copy stride)
_MLA_SLOT_BYTES = 576  # bytes per slot in the paged FP8 cache
_MLA_BF16_SLOT_ELEMS = _MLA_SLOT_BYTES // 2  # bf16-view slot stride (elements)
_MLA_BF16_ROPE_OFFSET = _MLA_NOPE_DIM // 2  # bf16-view rope offset (elements)
_MLA_SCALES_PER_TOKEN = 8  # UE8M0 scales per token (7 nope tiles + 1 padding)
_MLA_NUM_TILES = 8  # 7 nope quant tiles + 1 rope copy tile
_MLA_ROPE_TILE_ID = 7  # tile id reserved for the rope copy

# C4 indexer paged FP8 cache layout
_INDEXER_HEAD_DIM = 128

_UE8M0_EXPONENT_BIAS = 127


@triton.jit
def _triton_fused_store_flashmla_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_bf16_ptr,
    cache_u8_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_BF16: tl.constexpr,
    S_OFFSET: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    BF16_SLOT_ELEMS: tl.constexpr,
    BF16_ROPE_OFFSET: tl.constexpr,
    SCALES_PER_TOKEN: tl.constexpr,
    ROPE_TILE_ID: tl.constexpr,
    UE8M0_BIAS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    if token_id >= N:
        return

    # Per-layer caches exceed 2 GiB; int32 byte offsets wrap and write outside the buffer.
    loc = tl.load(indices_ptr + token_id).to(tl.int64)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    if tile_id == ROPE_TILE_ID:
        rope_lane = tl.arange(0, TILE_SIZE)
        rope_vals = tl.load(input_ptr + token_id * HEAD_DIM + NOPE_DIM + rope_lane)
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16
            + slot * BF16_SLOT_ELEMS
            + BF16_ROPE_OFFSET
            + rope_lane
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)
    else:
        tile_lane = tl.arange(0, TILE_SIZE)
        x_bf16 = tl.load(
            input_ptr + token_id * HEAD_DIM + tile_id * TILE_SIZE + tile_lane
        )
        x_fp32 = x_bf16.to(tl.float32)

        abs_max = tl.max(tl.abs(x_fp32))
        scale = tl.maximum(abs_max, EPS) / FP8_MAX

        # cast scale to ue8m0 format
        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)
        inv_scale = tl.exp2(-ceil_log2)

        x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
            cache_fp8_ptr.dtype.element_ty
        )

        nope_offset = (
            page * BYTES_PER_PAGE + slot * SLOT_BYTES + tile_id * TILE_SIZE + tile_lane
        )
        tl.store(cache_fp8_ptr + nope_offset, x_fp8)

        ue8m0 = (ceil_log2.to(tl.int32) + UE8M0_BIAS).to(tl.uint8)
        scale_offset = (
            page * BYTES_PER_PAGE + S_OFFSET + slot * SCALES_PER_TOKEN + tile_id
        )
        tl.store(cache_u8_ptr + scale_offset, ue8m0)


def triton_fused_store_flashmla(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the SWA (flashmla) KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_bf16 = cache.view(torch.bfloat16)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    _triton_fused_store_flashmla_kernel[(N, _MLA_NUM_TILES)](
        input,
        cache_fp8,
        cache_bf16,
        cache,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_BF16=bytes_per_page // 2,
        S_OFFSET=page_size * _MLA_SLOT_BYTES,
        TILE_SIZE=_MLA_TILE_SIZE,
        HEAD_DIM=_MLA_HEAD_DIM,
        NOPE_DIM=_MLA_NOPE_DIM,
        SLOT_BYTES=_MLA_SLOT_BYTES,
        BF16_SLOT_ELEMS=_MLA_BF16_SLOT_ELEMS,
        BF16_ROPE_OFFSET=_MLA_BF16_ROPE_OFFSET,
        SCALES_PER_TOKEN=_MLA_SCALES_PER_TOKEN,
        ROPE_TILE_ID=_MLA_ROPE_TILE_ID,
        UE8M0_BIAS=_UE8M0_EXPONENT_BIAS,
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


@triton.jit
def _triton_fused_store_indexer_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_f32_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_F32: tl.constexpr,
    SCALE_PAGE_OFFSET_F32: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int64)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    lane = tl.arange(0, HEAD_DIM)
    x_fp32 = tl.load(input_ptr + token_id * HEAD_DIM + lane).to(tl.float32)

    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX
    inv_scale = 1.0 / scale

    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        cache_fp8_ptr.dtype.element_ty
    )

    if PRESHUFFLE_TILE:
        token_tile_id = slot // PRESHUFFLE_TILE
        token_in_tile = slot % PRESHUFFLE_TILE
        col_tile_id = lane // PRESHUFFLE_TILE
        col_in_tile = lane % PRESHUFFLE_TILE
        fp8_offset = (
            page * BYTES_PER_PAGE
            + token_tile_id * (PRESHUFFLE_TILE * HEAD_DIM)
            + col_tile_id * (PRESHUFFLE_TILE * PRESHUFFLE_TILE)
            + token_in_tile * PRESHUFFLE_TILE
            + col_in_tile
        )
    else:
        fp8_offset = page * BYTES_PER_PAGE + slot * HEAD_DIM + lane
    tl.store(cache_fp8_ptr + fp8_offset, x_fp8)

    f32_offset = page * BYTES_PER_PAGE_F32 + SCALE_PAGE_OFFSET_F32 + slot
    tl.store(cache_f32_ptr + f32_offset, scale)


def triton_fused_store_indexer(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the C4 indexer KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    bytes_per_page_f32 = bytes_per_page // 4
    scale_page_offset_f32 = (_INDEXER_HEAD_DIM * page_size) // 4

    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_f32 = cache.view(torch.float32)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    _triton_fused_store_indexer_kernel[(N,)](
        input,
        cache_fp8,
        cache_f32,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_F32=bytes_per_page_f32,
        SCALE_PAGE_OFFSET_F32=scale_page_offset_f32,
        HEAD_DIM=_INDEXER_HEAD_DIM,
        PRESHUFFLE_TILE=(
            INDEXER_K_CACHE_PRESHUFFLE_TILE
            if aiter_can_use_preshuffle_paged_mqa()
            else 0
        ),
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


def triton_fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    """ROCm dispatch for fused_store_cache()."""
    if type == "flashmla":
        triton_fused_store_flashmla(input, cache, indices, page_size)
    else:
        triton_fused_store_indexer(input, cache, indices, page_size)


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
