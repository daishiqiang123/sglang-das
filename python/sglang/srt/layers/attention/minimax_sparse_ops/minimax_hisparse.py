"""Page staging for MiniMax's native four-KV-head FlashMLA verify path."""

import os
from typing import NamedTuple

import torch


class VerifyPagePlan(NamedTuple):
    indices: torch.Tensor
    labels: torch.Tensor
    miss_src: torch.Tensor
    miss_dst: torch.Tensor
    miss_count: torch.Tensor
    verify_src: torch.Tensor
    verify_dst: torch.Tensor
    verify_count: torch.Tensor


def plan_verify_pages(
    blocks: torch.Tensor,
    ends: torch.Tensor,
    host_pages: torch.Tensor,
    hot_pages: torch.Tensor,
    labels: torch.Tensor,
    verify_slots: torch.Tensor,
    active: torch.Tensor,
    page_size: int,
) -> VerifyPagePlan:
    """Build a fixed-shape plan for [request, query, KV head, TopK] logical pages.

    A staging slot holds the entire page for all heads. Sort/deduplicate the
    selected history pages across queries and heads, preserving the original
    per-query and per-head indices. Partial history pages are refreshed on the
    next forward, after its accepted tokens have reached the host pool.
    """
    batch, width, heads, topk = blocks.shape
    flat = blocks.reshape(batch, -1).to(torch.int64)
    prefix = ends[:, -1:] - width
    host_page = host_pages.gather(1, flat.clamp(min=0, max=host_pages.size(1) - 1))
    selected = (
        active[:, None]
        & (flat >= 0)
        & (flat < host_pages.size(1))
        & (flat * page_size < ends[:, -1:])
    )
    sentinel = torch.iinfo(torch.int64).max
    pages, order = torch.where(selected, flat, sentinel).sort(dim=1)
    valid = pages != sentinel
    first = (
        torch.cat([torch.ones_like(valid[:, :1]), pages[:, 1:] != pages[:, :-1]], dim=1)
        & valid
    )
    slots = (first.cumsum(dim=1) - 1).clamp_min(0)
    dst = hot_pages.gather(1, slots)
    src = host_page.gather(1, order)
    reusable = valid & (pages < torch.div(prefix, page_size, rounding_mode="floor"))
    hit = reusable & (labels.gather(1, slots) == pages)
    historical = valid & (pages < (prefix + page_size - 1) // page_size)
    miss = first & ~hit & historical & (src >= 0) & (dst > 0)
    # Pack misses to the left without a GPU-dependent output size.
    miss_order = (~miss).to(torch.int32).argsort(dim=1)
    miss_src = src.gather(1, miss_order).contiguous()
    miss_dst = dst.gather(1, miss_order).to(torch.int32).contiguous()
    miss_count = miss.sum(dim=1, dtype=torch.int32)
    new_labels = torch.full_like(labels, -1).scatter_reduce_(
        1, slots, torch.where(reusable & (dst > 0), pages, -1), reduce="amax"
    )

    # The native Stage3 fast path loads a logical block from one physical page.
    # Patch its speculative tail after swapping in the historical bytes.
    offsets = torch.arange(page_size, device=blocks.device)
    verify_pos = prefix + torch.arange(width, device=blocks.device)
    verify_page = verify_pos // page_size
    selected_slot = torch.searchsorted(pages, verify_page.contiguous())
    selected_slot = selected_slot.clamp_max(pages.size(1) - 1)
    verify_hot = dst.gather(1, selected_slot)
    copy_verify = (
        active[:, None]
        & (verify_slots > 0)
        & (verify_hot > 0)
        & (pages.gather(1, selected_slot) == verify_page)
    )
    verify_dst = verify_hot * page_size + verify_pos % page_size
    verify_order = (~copy_verify).to(torch.int32).argsort(dim=1)
    hot = (
        torch.empty_like(dst)
        .scatter_(1, order, torch.where(valid & (dst > 0), dst, -1))
        .reshape(batch, width, heads, topk, 1)
    )
    positions = blocks.to(torch.int64).unsqueeze(-1) * page_size + offsets
    prefix = prefix.reshape(batch, 1, 1, 1, 1)
    has_history = host_page.reshape(batch, width, heads, topk, 1) >= 0
    verify_offsets = (positions - prefix).clamp(min=0, max=width - 1)
    has_verify = (
        (verify_slots > 0)
        .gather(1, verify_offsets.reshape(batch, -1))
        .reshape_as(positions)
    )
    valid_token = (
        (hot > 0)
        & (positions < ends.reshape(batch, width, 1, 1, 1))
        & torch.where(positions < prefix, has_history, has_verify)
    )
    indices = torch.where(valid_token, hot * page_size + offsets, -1)
    return VerifyPagePlan(
        indices.flatten(-2).to(torch.int32),
        new_labels,
        miss_src,
        miss_dst,
        miss_count,
        verify_slots.gather(1, verify_order).contiguous(),
        verify_dst.gather(1, verify_order).to(torch.int32).contiguous(),
        copy_verify.sum(dim=1, dtype=torch.int32),
    )


class MiniMaxHiSparseVerify:
    """Own residency tags and the scratch page used by target verify."""

    width = 8
    heads = 4
    topk = 16
    page_size = 128

    def __init__(self, coordinator, request_slots: int, layers: int):
        self.coordinator = coordinator
        self.pages = self.width * self.heads * self.topk
        self.buffer_size = coordinator.device_buffer_size
        self.compact_hot_buffer = self.buffer_size < self.pages * self.page_size
        # Each query/head reserves one local page. A verify8 group spans at most
        # two local pages, leaving 8 * 4 * 15 other page selections.
        self.query_chunk_size = int(
            os.environ.get("M31_HISPARSE_VERIFY_QUERY_CHUNK", "8")
        )
        if self.query_chunk_size not in (4, 8):
            raise ValueError("MiniMax HiSparse verify query chunk must be 4 or 8")
        min_pages = self.query_chunk_size * self.heads * (self.topk - 1) + 2
        if self.compact_hot_buffer and (
            os.environ.get("M31_HISPARSE_AB_COMPACT_HOT") != "1"
            or self.buffer_size < min_pages * self.page_size
            or os.environ.get("M31_HISPARSE_FAST_PLAN") != "1"
            or os.environ.get("M31_HISPARSE_STABLE_SLOTS") == "1"
        ):
            raise ValueError(
                "Compact MiniMax hot slots require the forced-local verify8 planner"
            )
        self.request_slots = request_slots
        self.fast_plan = os.environ.get("M31_HISPARSE_FAST_PLAN") == "1"
        self.fused_commit = os.environ.get("M31_HISPARSE_BATCH_COMMIT") == "1"
        self.direct_verify_indices = (
            self.fast_plan
            and os.environ.get("M31_HISPARSE_DIRECT_VERIFY_INDICES") == "1"
        )
        self.tiled_swap = os.environ.get("M31_HISPARSE_TILED_SWAP", "")
        if self.tiled_swap:
            self.swap_launch = tuple(map(int, self.tiled_swap.split(",")))
            if self.swap_launch not in ((128, 16), (128, 32), (256, 16), (256, 32)):
                raise ValueError("Invalid MiniMax HiSparse tiled swap launch")
        pool = coordinator.mem_pool_device
        if pool.size < coordinator.padded_buffer_size:
            raise ValueError("MiniMax HiSparse KV pool cannot hold one request buffer")
        k, v = pool.get_kv_buffer(pool.start_layer)
        if (
            not k.is_contiguous()
            or not v.is_contiguous()
            or k.shape != v.shape
            or k.shape[-2:] != (4, 128)
        ):
            raise ValueError("MiniMax HiSparse requires contiguous token-major KV4")
        # A distinct scratch label row for each padded graph row. Padding may
        # reuse a live request id, so it must never write that request's tags.
        self.labels = torch.full(
            (layers, 2 * request_slots, self.pages),
            -1,
            dtype=torch.int64,
            device=coordinator.device,
        )

        self.page_slots = self._allocate_page_slots(layers)

    def bind_host_commit_buffers(self) -> None:
        if not self.fused_commit:
            return
        device_pool = self.coordinator.mem_pool_device
        host_pool = self.coordinator.mem_pool_host
        self.host_commit_buffers = [
            (
                device_pool.k_data_ptrs[start:end],
                device_pool.v_data_ptrs[start:end],
                host_pool.k_buffer[start:end],
                host_pool.v_buffer[start:end],
            )
            for start, end in host_pool.commit_layer_ranges()
        ]

    def _allocate_page_slots(self, layers: int) -> torch.Tensor | None:
        if not self.fast_plan or os.environ.get("M31_HISPARSE_STABLE_SLOTS") != "1":
            return None
        host_tokens = self.coordinator.req_to_host_pool.shape[1]
        logical_pages = (host_tokens + self.page_size - 1) // self.page_size
        return torch.full(
            (layers, self.request_slots, logical_pages),
            -1,
            dtype=torch.int32,
            device=self.labels.device,
        )

    def reset(self, req_pool_idx: int) -> None:
        self.labels[:, req_pool_idx].fill_(-1)
        if self.page_slots is not None:
            self.page_slots[:, req_pool_idx].fill_(-1)

    def _scratch_slots(self, reqs: torch.Tensor) -> torch.Tensor:
        return self.coordinator.req_to_device_buffer[
            reqs, self.buffer_size : self.buffer_size + self.width
        ]

    def prepare(self, batch, verify_locs: torch.Tensor) -> None:
        c = self.coordinator
        for req in batch.reqs:
            rid = req.kv.req_pool_idx
            allocated = int(c.req_to_host_pool_allocated_len[rid])
            extra = req.kv.kv_allocated_len - allocated
            if extra > 0:
                c.mem_pool_host.alloc_paged_token_slots(
                    c.req_to_host_pool,
                    c.req_to_host_pool_allocated_len,
                    rid,
                    allocated,
                    extra,
                )
        c.mem_pool_device.full_to_hisparse_device_index_mapping[verify_locs] = (
            self._scratch_slots(batch.req_pool_indices).flatten()
        )

    def commit(
        self,
        reqs: torch.Tensor,
        prefix: torch.Tensor,
        verify_locs: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mha

        c = self.coordinator
        positions = prefix[:, None] + torch.arange(self.width, device=reqs.device)
        host_locs = c.req_to_host_pool[reqs[:, None], positions].to(torch.int32)
        scratch = self._scratch_slots(reqs).contiguous()
        counts = accepted.to(torch.int32).contiguous()
        if self.fused_commit:
            from sglang.kernels.ops.kvcache.hisparse import commit_mha_all_layers

            for buffers in self.host_commit_buffers:
                commit_mha_all_layers(
                    scratch, host_locs, counts, *buffers, c.item_size_bytes
                )
        else:
            real_reqs = torch.full(
                (1,), reqs.numel(), device=reqs.device, dtype=torch.int32
            )
            for i in range(c.mem_pool_device.layer_num):
                layer = c.mem_pool_device.start_layer + i
                copy_cache_planned_mha(
                    scratch,
                    host_locs,
                    counts,
                    real_reqs,
                    c.mem_pool_device.get_key_buffer(layer),
                    c.mem_pool_device.get_value_buffer(layer),
                    c.mem_pool_host.k_buffer[i],
                    c.mem_pool_host.v_buffer[i],
                    c.item_size_bytes,
                )
        # Only the accepted prefix was backed up. The scratch page remains
        # owned by this request and is overwritten by the next verify.
        c.mem_pool_device.full_to_hisparse_device_index_mapping.index_fill_(
            0, verify_locs, 0
        )

    def swap(
        self,
        reqs: torch.Tensor,
        layer_id: int,
        topk_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        indices_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mha

        c = self.coordinator
        batch = reqs.numel()
        if batch > self.request_slots or topk_idx.shape != (
            self.heads,
            batch * self.width,
            self.topk,
        ):
            raise ValueError("MiniMax HiSparse expects fixed-width verify8 TopK")
        layer = layer_id - c.mem_pool_device.start_layer
        tags = self.labels[layer]
        if self.fast_plan:
            from .minimax_hisparse_plan import pack_verify_plan

            plan = pack_verify_plan(
                reqs,
                c.num_real_reqs,
                topk_idx,
                seq_lens,
                c.req_to_host_pool,
                c.req_to_device_buffer,
                tags,
                self.buffer_size,
                None if self.page_slots is None else self.page_slots[layer],
                indices_output=indices_output,
                direct_verify_indices=self.direct_verify_indices,
            )
        else:
            rows = torch.arange(batch, device=reqs.device)
            safe_reqs = reqs.clamp(min=0, max=self.request_slots - 1).to(torch.int64)
            active = (
                (rows < c.num_real_reqs[0])
                & (reqs >= 0)
                & (reqs < self.request_slots)
                & (c.req_to_device_buffer[safe_reqs, 0] > 0)
            )
            reqs = torch.where(active, safe_reqs, 0)
            tag_rows = torch.where(active, reqs, self.request_slots + rows)
            blocks = topk_idx.permute(1, 0, 2).reshape(
                batch, self.width, self.heads, self.topk
            )
            host_pages = torch.div(
                c.req_to_host_pool[reqs, :: self.page_size],
                self.page_size,
                rounding_mode="floor",
            )
            hot_pages = (
                c.req_to_device_buffer[
                    reqs, : self.pages * self.page_size : self.page_size
                ]
                // self.page_size
            )
            plan = plan_verify_pages(
                blocks,
                seq_lens.reshape(batch, self.width),
                host_pages,
                hot_pages,
                tags.index_select(0, tag_rows),
                self._scratch_slots(reqs),
                active,
                self.page_size,
            )
        if self.tiled_swap:
            from sglang.kernels.ops.kvcache.hisparse import swap_mha_pages_tiled

            swap_mha_pages_tiled(
                plan.miss_src,
                plan.miss_dst,
                plan.miss_count,
                c.num_real_reqs,
                c.mem_pool_host.k_buffer[layer],
                c.mem_pool_host.v_buffer[layer],
                c.mem_pool_device.get_key_buffer(layer_id),
                c.mem_pool_device.get_value_buffer(layer_id),
                *self.swap_launch,
            )
        else:
            copy_cache_planned_mha(
                plan.miss_src,
                plan.miss_dst,
                plan.miss_count,
                c.num_real_reqs,
                c.mem_pool_host.k_buffer[layer],
                c.mem_pool_host.v_buffer[layer],
                c.mem_pool_device.get_key_buffer(layer_id),
                c.mem_pool_device.get_value_buffer(layer_id),
                self.page_size * c.item_size_bytes,
            )
        k = c.mem_pool_device.get_key_buffer(layer_id)
        v = c.mem_pool_device.get_value_buffer(layer_id)
        if not self.direct_verify_indices:
            copy_cache_planned_mha(
                plan.verify_src,
                plan.verify_dst,
                plan.verify_count,
                c.num_real_reqs,
                k,
                v,
                k,
                v,
                c.item_size_bytes,
            )
        if not self.fast_plan:
            tags.index_copy_(0, tag_rows, plan.labels)
        return plan.indices


def flash_mla_chunked_verify(*, query_chunk_size, hot_indices_fn, **kwargs):
    """Bound resident pages by staging subsets of verify queries exactly.

    Score/TopK and DSpark width are unchanged. Each pass retains all selected
    pages for its query subset, consumes them before another pass can reuse
    hot slots, and copies only those output rows into the final result.
    """
    from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
        flash_mla_sparse_decode_main,
    )

    width = kwargs["verify_group_size"]
    if width != 8 or query_chunk_size not in (4, 8):
        raise ValueError("Chunked HiSparse requires verify8 and query chunks 4 or 8")
    q, topk = kwargs["q"], kwargs["topk_idx"]
    batch = q.shape[0] // width
    output = torch.empty_like(q, dtype=torch.bfloat16)
    # Keep the same native verify8 split/reduction schedule as the unsplit path.
    # Masked queries are discarded, never used as approximate attention.
    for start in range(0, width, query_chunk_size):
        selected = torch.full_like(topk, -1)
        selected.view(4, batch, width, 16)[
            :, :, start : start + query_chunk_size
        ].copy_(topk.view(4, batch, width, 16)[:, :, start : start + query_chunk_size])
        slots = hot_indices_fn(selected, kwargs["seq_lens"])
        result = flash_mla_sparse_decode_main(
            **dict(kwargs, topk_idx=selected, hot_indices=slots)
        )
        if result.dtype != output.dtype:
            raise ValueError("Unexpected FlashMLA output dtype in chunked verify")
        output.view(batch, width, *q.shape[1:])[
            :, start : start + query_chunk_size
        ].copy_(
            result.view(batch, width, *q.shape[1:])[:, start : start + query_chunk_size]
        )
    return output
