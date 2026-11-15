"""Fixed-shape GPU page planning for MiniMax native MSA128 verify8."""

from functools import lru_cache
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl


class PackedPlan(NamedTuple):
    indices: torch.Tensor
    miss_src: torch.Tensor
    miss_dst: torch.Tensor
    miss_count: torch.Tensor
    verify_src: torch.Tensor
    verify_dst: torch.Tensor
    verify_count: torch.Tensor


@triton.jit
def _max_slot(left, right):
    return tl.maximum(left, right)


@triton.jit
def _pack_plan(
    Reqs,
    Real,
    TopK,
    Ends,
    Host,
    Device,
    Tags,
    HotForSelection,
    PageSlots,
    MissSrc,
    MissDst,
    MissCount,
    VerifySrc,
    VerifyDst,
    VerifyCount,
    B: tl.constexpr,
    REQ_SLOTS: tl.constexpr,
    HOST_STRIDE: tl.constexpr,
    DEVICE_STRIDE: tl.constexpr,
    BUFFER_SIZE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    TOP_CHOICE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    TOP: tl.constexpr,
    COUNT: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    STABLE: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(Reqs + row).to(tl.int32)
    present = (row < tl.load(Real)) & (request >= 0) & (request < REQ_SLOTS)
    request = tl.where(present, request, 0)
    first_hot = tl.load(Device + request * DEVICE_STRIDE, mask=present, other=0)
    active = present & (first_hot > 0)

    item = tl.arange(0, COUNT)
    query = item // (HEADS * TOP)
    head = item // TOP % HEADS
    choice = item % TOP
    logical = tl.load(
        TopK
        + head * TOP_HEAD_STRIDE
        + (row * WIDTH + query) * TOP_QUERY_STRIDE
        + choice * TOP_CHOICE_STRIDE
    )
    last_end = tl.load(Ends + row * WIDTH + WIDTH - 1)
    prefix = last_end - WIDTH
    valid = active & (logical >= 0) & (logical * PAGE < last_end)
    valid = valid & (logical * PAGE < HOST_STRIDE)
    # Keep a unique key for each input, including invalid slots.
    sentinel: tl.constexpr = 0x7FFFFE00 // COUNT
    keys = tl.where(valid, logical, sentinel) * COUNT + item
    keys = tl.sort(keys, descending=False)
    page = keys // COUNT
    order = keys % COUNT
    valid = page < sentinel
    previous = tl.gather(page, tl.maximum(item - 1, 0), axis=0)
    first = valid & ((item == 0) | (page != previous))
    if STABLE:
        map_addr = PageSlots + request * MAP_STRIDE + page
        prior = tl.load(map_addr, mask=first, other=-1)
        prior_valid = first & (prior >= 0) & (prior < COUNT)
        tag = tl.load(Tags + request * COUNT + prior, mask=prior_valid, other=-1)
        keep = prior_valid & (tag == page)
        occupied = tl.histogram(tl.where(keep, prior, -1), COUNT)
        free_slots = tl.sort(tl.where(occupied == 0, item, COUNT), descending=False)
        fresh_idx = tl.maximum(tl.cumsum((first & ~keep).to(tl.int32), axis=0) - 1, 0)
        first_slot = tl.where(keep, prior, tl.gather(free_slots, fresh_idx, axis=0))
        first_idx = tl.associative_scan(tl.where(first, item, 0), 0, _max_slot)
        slot = tl.gather(first_slot, first_idx, axis=0)
    else:
        slot = tl.maximum(tl.cumsum(first.to(tl.int32), axis=0) - 1, 0)
    if BUFFER_SIZE < COUNT * PAGE:
        fits = slot < BUFFER_SIZE // PAGE
        tl.device_assert(
            tl.sum((first & ~fits).to(tl.int32), axis=0) == 0,
            "MiniMax verify exceeded the compact hot-page bound",
        )
        valid = valid & fits
        first = first & fits
    hot_slot = (
        tl.load(Device + request * DEVICE_STRIDE + slot * PAGE, mask=valid, other=0).to(
            tl.int32
        )
        // PAGE
    )
    tl.store(
        HotForSelection + row * COUNT + order,
        tl.where(valid & (hot_slot > 0), hot_slot, -1),
    )

    past = valid & (page * PAGE < prefix)
    host_slot = tl.load(
        Host + request * HOST_STRIDE + page * PAGE, mask=first & past, other=-1
    )
    label = tl.load(Tags + request * COUNT + slot, mask=first, other=-1)
    reusable = first & (page < prefix // PAGE) & (hot_slot > 0) & (host_slot >= 0)
    miss = (
        first & past & (host_slot >= 0) & (hot_slot > 0) & ~(reusable & (label == page))
    )
    m = tl.cumsum(miss.to(tl.int32), axis=0) - 1
    tl.store(MissSrc + row * COUNT + m, host_slot // PAGE, mask=miss)
    tl.store(MissDst + row * COUNT + m, hot_slot, mask=miss)
    tl.store(MissCount + row, tl.sum(miss.to(tl.int32), axis=0))
    # Each touched slot has exactly one writer. Unused slots still hold their old bytes.
    if STABLE:
        # All waves must finish reading the old tags before any slot is reused.
        tl.debug_barrier()
        tl.store(map_addr, slot, mask=first)
    tl.store(Tags + request * COUNT + slot, tl.where(reusable, page, -1), mask=first)

    left = prefix // PAGE
    right = (last_end - 1) // PAGE
    hot_left = tl.max(tl.where(valid & (page == left), hot_slot, 0), axis=0)
    hot_right = tl.max(tl.where(valid & (page == right), hot_slot, 0), axis=0)
    step = tl.arange(0, WIDTH)
    position = prefix + step
    target = tl.where(position // PAGE == left, hot_left, hot_right)
    scratch = tl.load(
        Device + request * DEVICE_STRIDE + BUFFER_SIZE + step, mask=active, other=0
    )
    copy = active & (target > 0) & (scratch > 0)
    v = tl.cumsum(copy.to(tl.int32), axis=0) - 1
    tl.store(VerifySrc + row * WIDTH + v, scratch, mask=copy)
    tl.store(VerifyDst + row * WIDTH + v, target * PAGE + position % PAGE, mask=copy)
    tl.store(VerifyCount + row, tl.sum(copy.to(tl.int32), axis=0))


@triton.jit
def _expand_indices(
    Reqs,
    TopK,
    Ends,
    Host,
    Device,
    HotForSelection,
    Indices,
    B: tl.constexpr,
    REQ_SLOTS: tl.constexpr,
    HOST_STRIDE: tl.constexpr,
    DEVICE_STRIDE: tl.constexpr,
    BUFFER_SIZE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    TOP_CHOICE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    TOP: tl.constexpr,
    COUNT: tl.constexpr,
    SELECTION_TILE: tl.constexpr,
    DIRECT_VERIFY_INDICES: tl.constexpr,
):
    row = tl.program_id(1)
    selection = tl.program_id(0) * SELECTION_TILE + tl.arange(0, SELECTION_TILE)
    offset = tl.arange(0, PAGE)
    query = selection // (HEADS * TOP)
    head = selection // TOP % HEADS
    choice = selection % TOP
    hot = tl.load(HotForSelection + row * COUNT + selection)
    logical = tl.load(
        TopK
        + head * TOP_HEAD_STRIDE
        + (row * WIDTH + query) * TOP_QUERY_STRIDE
        + choice * TOP_CHOICE_STRIDE
    )
    position = logical[:, None] * PAGE + offset[None, :]
    end = tl.load(Ends + row * WIDTH + query)
    prefix = tl.load(Ends + row * WIDTH + WIDTH - 1) - WIDTH
    request = tl.load(Reqs + row).to(tl.int32)
    request = tl.where((request >= 0) & (request < REQ_SLOTS), request, 0)
    mask = (hot[:, None] > 0) & (position < end[:, None])
    host = tl.load(
        Host + request * HOST_STRIDE + logical * PAGE,
        mask=(hot > 0) & (logical * PAGE < prefix),
        other=-1,
    )
    step = tl.minimum(tl.maximum(position - prefix, 0), WIDTH - 1)
    scratch = tl.load(
        Device + request * DEVICE_STRIDE + BUFFER_SIZE + step,
        mask=mask & (position >= prefix),
        other=0,
    )
    valid = mask & tl.where(position < prefix, host[:, None] >= 0, scratch > 0)
    indices = (row * COUNT + selection[:, None]) * PAGE + offset[None, :]
    physical = hot[:, None] * PAGE + offset[None, :]
    if DIRECT_VERIFY_INDICES:
        # The current verify tokens already live in per-request scratch.
        physical = tl.where(position < prefix, physical, scratch)
    tl.store(Indices + indices, tl.where(valid, physical, -1))


def _hip_stable_plan(*args, headwise=False):
    from sglang.kernels.jit.utils import load_jit

    return _hip_module(load_jit, headwise).plan(*args)


@lru_cache(maxsize=1)
def _hip_module(load_jit, headwise=False):
    return load_jit(
        "minimax_verify_owner_plan",
        headwise,
        cuda_files=["kvcacheio/minimax_verify_plan.cuh"],
        cuda_wrappers=[("plan", f"minimax_verify::plan<{str(headwise).lower()}>")],
    )


def pack_verify_plan(
    reqs,
    real,
    topk,
    ends,
    host,
    device,
    tags,
    buffer_size,
    page_slots=None,
    *,
    indices_output=None,
    direct_verify_indices=False,
    headwise=False,
):
    batch = reqs.numel()
    width, heads, top, page = 8, 4, 16, 128
    count = width * heads * top
    opts = {"device": reqs.device}
    hot = torch.empty((batch, count), dtype=torch.int32, **opts)
    shape_indices = (batch, width, heads, top * page)
    if indices_output is None:
        indices = torch.empty(shape_indices, dtype=torch.int32, **opts)
    else:
        indices = indices_output
        assert (
            indices.shape == shape_indices
            and indices.dtype == torch.int32
            and indices.device == reqs.device
            and indices.is_contiguous()
        )
    plan_rows = batch * heads if headwise else batch
    plan_width = count // heads if headwise else count
    miss_src = torch.empty((plan_rows, plan_width), dtype=torch.int64, **opts)
    miss_dst = torch.empty_like(miss_src, dtype=torch.int32)
    miss_count = torch.empty((plan_rows,), dtype=torch.int32, **opts)
    verify_src = torch.empty((batch, width), dtype=torch.int64, **opts)
    verify_dst = torch.empty((batch, width), dtype=torch.int32, **opts)
    verify_count = torch.empty((batch,), dtype=torch.int32, **opts)
    shape = (
        batch,
        host.size(0),
        host.stride(0),
        device.stride(0),
        buffer_size,
        *topk.stride(),
        page,
        width,
        heads,
        top,
        count,
    )
    ends = ends.contiguous()
    use_hip = (
        page_slots is not None
        and (plan_width * page <= buffer_size <= 256 * page if headwise else buffer_size == count * page)
        and os.environ.get("M31_HISPARSE_HIP_STABLE_PLAN") == "1"
        and reqs.dtype == host.dtype == device.dtype == tags.dtype == torch.int64
        and topk.dtype == ends.dtype == torch.int32
        and reqs.is_contiguous()
        and page_slots.size(1) <= 2048
    )
    if headwise and not (use_hip and direct_verify_indices):
        raise ValueError("Head-wise HiSparse requires HIP planning and direct Verify indices")
    if use_hip:
        _hip_stable_plan(
            reqs,
            real,
            topk,
            ends,
            host,
            device,
            tags,
            hot,
            page_slots,
            miss_src,
            miss_dst,
            miss_count,
            verify_src,
            verify_dst,
            verify_count,
            headwise=headwise,
        )
    else:
        _pack_plan[(batch,)](
            reqs,
            real,
            topk,
            ends,
            host,
            device,
            tags,
            hot,
            tags if page_slots is None else page_slots,
            miss_src,
            miss_dst,
            miss_count,
            verify_src,
            verify_dst,
            verify_count,
            *shape,
            0 if page_slots is None else page_slots.stride(0),
            page_slots is not None,
            num_warps=4,
            debug=buffer_size < count * page,
        )
    selection_tile = 16
    _expand_indices[(count // selection_tile, batch)](
        reqs,
        topk,
        ends,
        host,
        device,
        hot,
        indices,
        *shape,
        selection_tile,
        direct_verify_indices,
        num_warps=4,
    )
    return PackedPlan(
        indices, miss_src, miss_dst, miss_count, verify_src, verify_dst, verify_count
    )
