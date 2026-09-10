"""HCU MLA helpers for true zigzag prefill context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class HCUMLACPRingSourceLayout:
    """Natural early/late token geometry for one compact-KV source rank."""

    token_count: int
    early_token_count: int
    early_lens: tuple[int, ...]
    late_lens: tuple[int, ...]


def clear_hcu_mla_cp_ring_state(forward_batch: Any) -> None:
    """Release the per-layer compact-ring tensors after attention finishes."""
    forward_batch.mla_cp_hcu_ring_active = False
    forward_batch.mla_cp_local_k = None
    forward_batch.mla_cp_local_k_rope = None
    forward_batch.mla_cp_prefix_k = None
    forward_batch.mla_cp_prefix_k_rope = None


def hcu_mla_use_ring_prefill_cp(forward_batch: Any) -> bool:
    """Whether the HCU compact-latent MLA ring can own this prefill.

    Keep this as a strict runtime gate.  A CP caller must fail closed when any
    prerequisite is absent; there is no expanded full-KV/local-Q fallback.
    """
    from sglang.srt.layers.cp.utils import is_cp_v2_active
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import get_bool_env_var, is_hcu

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    split_list = getattr(metadata, "split_list", None)
    mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        is_hcu()
        and get_bool_env_var("SGLANG_HCU_MLA_CP_RING", "true")
        and get_parallel().attn_cp_size > 1
        and not get_parallel().dcp_enabled
        and is_cp_v2_active(forward_batch)
        and metadata is not None
        and split_list
        and min(int(length) for length in split_list) > 0
        and mode is not None
        and bool(getattr(forward_batch, "mha_one_shot", False))
        and mode.is_context_parallel_extend()
        and not mode.is_mixed()
        and not mode.is_target_verify()
        and not mode.is_draft_extend_v2()
        # The separate chunked-prefix-LSE scheduler path needs its own ring
        # integration.  Ordinary chunked prefill with a non-zero
        # extend_prefix_lens_cpu is supported below.
        and not bool(getattr(forward_batch, "attn_attend_prefix_cache", False))
    )


def select_mha_prefix_kv_indices(
    kv_indices: torch.Tensor,
    seq_lens: list[int],
    prefix_lens: list[int],
) -> torch.Tensor:
    """Select the natural packed prefix rows from one-shot KV indices."""
    if len(seq_lens) != len(prefix_lens):
        raise ValueError(
            "HCU MLA CP prefix index geometry mismatch: "
            f"seq_lens={seq_lens}, prefix_lens={prefix_lens}."
        )
    if sum(int(length) for length in seq_lens) != kv_indices.numel():
        raise ValueError(
            "HCU MLA CP one-shot KV indices do not match sequence lengths: "
            f"indices={kv_indices.numel()}, seq_lens={seq_lens}."
        )

    chunks = torch.split(kv_indices, [int(length) for length in seq_lens], dim=0)
    selected = []
    for request_id, (indices, prefix_len) in enumerate(zip(chunks, prefix_lens)):
        prefix_len = int(prefix_len)
        if prefix_len < 0 or prefix_len > indices.numel():
            raise ValueError(
                "HCU MLA CP prefix length is outside the request: "
                f"request={request_id}, prefix={prefix_len}, seq={indices.numel()}."
            )
        if prefix_len:
            selected.append(indices[:prefix_len])
    if not selected:
        return kv_indices.new_empty((0,))
    return selected[0] if len(selected) == 1 else torch.cat(selected)


def get_zigzag_cp_rank_chunk_indices(
    batch_size: int, cp_size: int, cp_rank: int
) -> list[int]:
    """Map one rank to natural zigzag chunks in local tensor order."""
    if batch_size <= 0 or cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(
            f"Invalid zigzag topology: bs={batch_size}, cp={cp_size}, rank={cp_rank}."
        )
    segments = 2 * cp_size
    return list(range(cp_rank, batch_size * segments, segments)) + list(
        range(segments - cp_rank - 1, batch_size * segments, segments)
    )


def build_hcu_mla_cp_ring_source_layouts(
    metadata: Any, *, cp_size: int
) -> tuple[HCUMLACPRingSourceLayout, ...]:
    """Build compact-KV layouts for every source rank."""
    batch_size = int(metadata.bs)
    expected_splits = batch_size * 2 * cp_size
    if len(metadata.split_list) != expected_splits:
        raise ValueError(
            "HCU MLA CP ring requires zigzag split metadata: "
            f"splits={len(metadata.split_list)}, expected={expected_splits}."
        )
    layouts = []
    for source_rank in range(cp_size):
        chunk_indices = get_zigzag_cp_rank_chunk_indices(
            batch_size, cp_size, source_rank
        )
        early = tuple(
            int(metadata.split_list[index]) for index in chunk_indices[:batch_size]
        )
        late = tuple(
            int(metadata.split_list[index]) for index in chunk_indices[batch_size:]
        )
        layouts.append(
            HCUMLACPRingSourceLayout(
                token_count=sum(early) + sum(late),
                early_token_count=sum(early),
                early_lens=early,
                late_lens=late,
            )
        )
    return tuple(layouts)


def build_hcu_mla_cp_ring_cache_locs(
    cache_locs: torch.Tensor, metadata: Any, *, cp_size: int
) -> tuple[torch.Tensor, ...]:
    """Map each compact ring shard to natural persistent-cache locations."""
    logical_tokens = sum(int(length) for length in metadata.split_list)
    if cache_locs.shape[0] < logical_tokens:
        raise ValueError(
            "HCU MLA CP ring cache locations are shorter than logical tokens: "
            f"locations={cache_locs.shape[0]}, logical={logical_tokens}."
        )
    natural_chunks = torch.split(
        cache_locs[:logical_tokens],
        [int(length) for length in metadata.split_list],
        dim=0,
    )
    return tuple(
        torch.cat(
            [
                natural_chunks[index]
                for index in get_zigzag_cp_rank_chunk_indices(
                    int(metadata.bs), cp_size, source_rank
                )
            ]
        )
        for source_rank in range(cp_size)
    )


def run_hcu_mla_cp_ring(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    forward_batch: Any,
    layer: Any,
    token_to_kv_pool: Any,
    *,
    run_segment: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    merge_segment: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    prefix_block_size: int = 32768,
) -> torch.Tensor:
    """Run true zigzag MLA PCP by rotating compact latent KV.

    Q stays rank-local.  The ring communicates only latent K and compact
    K-RoPE; each source shard is expanded through ``kv_b_proj`` immediately
    before its visible attention rectangles are computed.  No full expanded
    K/V tensor is staged on a rank.
    """
    metadata = forward_batch.attn_cp_metadata
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    group = parallel.attn_cp_group
    cp_size = int(group.world_size)
    cp_rank = int(group.rank_in_group)
    batch_size = int(metadata.bs)
    layouts = build_hcu_mla_cp_ring_source_layouts(metadata, cp_size=cp_size)
    local_layout = layouts[cp_rank]

    local_latent = forward_batch.mla_cp_local_k
    local_rope = forward_batch.mla_cp_local_k_rope
    if local_latent.shape[0] != local_layout.token_count:
        raise ValueError(
            "HCU MLA CP ring local latent length mismatch: "
            f"latent={local_latent.shape[0]}, expected={local_layout.token_count}."
        )
    if local_rope.shape[0] != local_layout.token_count:
        raise ValueError("HCU MLA CP ring local latent/RoPE lengths differ.")
    if layer.kv_b_proj is None:
        raise ValueError("HCU MLA CP ring requires kv_b_proj on RadixAttention.")

    local_prev_lens = tuple(int(x) for x in metadata.actual_seq_q_prev_list)
    local_next_lens = tuple(int(x) for x in metadata.actual_seq_q_next_list)
    if len(local_prev_lens) != batch_size or len(local_next_lens) != batch_size:
        raise ValueError("HCU MLA CP ring query metadata does not match batch size.")
    q_prev_tokens = sum(local_prev_lens)
    q_next_tokens = sum(local_next_lens)
    logical_q_tokens = q_prev_tokens + q_next_tokens
    q = q[:logical_q_tokens]
    q_prev, q_next = q[:q_prev_tokens], q[q_prev_tokens:]

    latent_width = local_latent.shape[1] * local_latent.shape[2]
    rope_width = local_rope.shape[1] * local_rope.shape[2]
    max_rank_tokens = max(layout.token_count for layout in layouts)

    packed = local_latent.new_zeros((max_rank_tokens, latent_width + rope_width))
    packed[: local_layout.token_count, :latent_width].copy_(local_latent.flatten(1))
    packed[: local_layout.token_count, latent_width:].copy_(local_rope.flatten(1))

    cache_locs_by_rank = build_hcu_mla_cp_ring_cache_locs(
        forward_batch.out_cache_loc, metadata, cp_size=cp_size
    )
    output_prev = lse_prev = None
    output_next = lse_next = None

    def expand_compact(latent: torch.Tensor, rope: torch.Tensor):
        projected = layer.kv_b_proj(latent.squeeze(1))[0].view(
            -1, layer.tp_k_head_num, layer.v_head_dim * 2
        )
        k_nope, value = projected.split(
            [layer.v_head_dim, layer.v_head_dim], dim=-1
        )
        expanded_rope = rope.expand(-1, layer.tp_k_head_num, -1)
        return (
            torch.cat((k_nope, expanded_rope), dim=-1).contiguous(),
            value.contiguous(),
        )

    def accumulate_state(
        old_output: Optional[torch.Tensor],
        old_lse: Optional[torch.Tensor],
        q_part: torch.Tensor,
        k_part: torch.Tensor,
        v_part: torch.Tensor,
        q_lens: tuple[int, ...] | list[int],
        kv_lens: tuple[int, ...] | list[int],
        *,
        causal: bool,
    ):
        new_output, new_lse = run_segment(
            q_part,
            k_part,
            v_part,
            list(q_lens),
            list(kv_lens),
            causal=causal,
        )
        if old_output is None:
            return new_output, new_lse
        return merge_segment(old_output, old_lse, new_output, new_lse)

    def merge_state(
        old_output: Optional[torch.Tensor],
        old_lse: Optional[torch.Tensor],
        new_output: torch.Tensor,
        new_lse: torch.Tensor,
    ):
        """Merge one already-computed rectangle without launching FA again."""
        if old_output is None:
            return new_output, new_lse
        return merge_segment(old_output, old_lse, new_output, new_lse)

    packed_query_cache = None

    def pack_query_halves():
        """Interleave q_prev/q_next per request for one varlen FA batch."""
        nonlocal packed_query_cache
        if packed_query_cache is not None:
            return packed_query_cache
        parts = []
        prev_start = next_start = 0
        for prev_len, next_len in zip(local_prev_lens, local_next_lens):
            prev_end = prev_start + int(prev_len)
            next_end = next_start + int(next_len)
            parts.extend(
                (q_prev[prev_start:prev_end], q_next[next_start:next_end])
            )
            prev_start, next_start = prev_end, next_end
        if not parts:
            packed_query_cache = q_prev.new_empty((0, *q_prev.shape[1:]))
        else:
            packed_query_cache = torch.cat(parts, dim=0)
        return packed_query_cache

    def pack_source_halves(
        early_part: torch.Tensor,
        late_part: torch.Tensor,
        early_lens: tuple[int, ...],
        late_lens: tuple[int, ...],
    ):
        """Interleave a source's early/late slabs per request."""
        parts = []
        early_start = late_start = 0
        for early_len, late_len in zip(early_lens, late_lens):
            early_end = early_start + int(early_len)
            late_end = late_start + int(late_len)
            parts.extend((early_part[early_start:early_end], late_part[late_start:late_end]))
            early_start, late_start = early_end, late_end
        if not parts:
            return early_part.new_empty((0, *early_part.shape[1:]))
        return torch.cat(parts, dim=0)

    def split_query_halves(
        packed_part: torch.Tensor,
        prev_lens: tuple[int, ...],
        next_lens: tuple[int, ...],
    ):
        """Undo per-request query packing while preserving the legacy layout."""
        prev_parts = []
        next_parts = []
        packed_start = 0
        for prev_len, next_len in zip(prev_lens, next_lens):
            prev_end = packed_start + int(prev_len)
            next_end = prev_end + int(next_len)
            prev_parts.append(packed_part[packed_start:prev_end])
            next_parts.append(packed_part[prev_end:next_end])
            packed_start = next_end
        if not prev_parts:
            empty = packed_part.new_empty((0, *packed_part.shape[1:]))
            return empty, empty
        return torch.cat(prev_parts, dim=0), torch.cat(next_parts, dim=0)

    def store_source_cache(
        source_rank: int,
        source_latent: torch.Tensor,
        source_rope: torch.Tensor,
        source_layout: HCUMLACPRingSourceLayout,
    ):
        source_cache_locs = cache_locs_by_rank[source_rank]
        if source_cache_locs.shape[0] != source_layout.token_count:
            raise ValueError("HCU MLA CP ring cache/source token lengths differ.")
        token_to_kv_pool.set_mla_kv_buffer(
            layer, source_cache_locs, source_latent, source_rope
        )

    for ring_step in range(cp_size):
        recv_packed = requests = None
        if ring_step + 1 < cp_size:
            recv_packed = torch.empty_like(packed)
            next_rank = group.ranks[(cp_rank + 1) % cp_size]
            prev_rank = group.ranks[(cp_rank - 1) % cp_size]
            requests = torch.distributed.batch_isend_irecv(
                [
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        recv_packed,
                        prev_rank,
                        group.device_group,
                    ),
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        packed,
                        next_rank,
                        group.device_group,
                    ),
                ]
            )

        source_rank = (cp_rank - ring_step) % cp_size
        source_layout = layouts[source_rank]
        source_payload = packed[: source_layout.token_count]
        source_latent = source_payload[:, :latent_width].reshape(
            source_layout.token_count,
            local_latent.shape[1],
            local_latent.shape[2],
        )
        source_rope = source_payload[:, latent_width:].reshape(
            source_layout.token_count, local_rope.shape[1], local_rope.shape[2]
        )
        early_end = source_layout.early_token_count
        if ring_step == 0:
            source_k, source_v = local_k, local_v
        else:
            source_k, source_v = expand_compact(source_latent, source_rope)
        early_k, late_k = source_k[:early_end], source_k[early_end:]
        early_v, late_v = source_v[:early_end], source_v[early_end:]

        # For the local source, q_prev followed by q_next and early_k
        # followed by late_k form one equal-length causal sequence.  A
        # causal FA call therefore exactly covers the old three calls:
        # q_prev->early (causal), q_next->early (non-causal), and
        # q_next->late (causal).
        if source_rank == cp_rank:
            packed_q = pack_query_halves()
            packed_k = pack_source_halves(
                early_k,
                late_k,
                source_layout.early_lens,
                source_layout.late_lens,
            )
            packed_lens = tuple(
                int(prev_len) + int(next_len)
                for prev_len, next_len in zip(
                    local_prev_lens, local_next_lens
                )
            )
            packed_k_lens = tuple(
                int(early_len) + int(late_len)
                for early_len, late_len in zip(
                    source_layout.early_lens,
                    source_layout.late_lens,
                )
            )
            if packed_lens != packed_k_lens:
                raise ValueError(
                    "HCU MLA packed local geometry mismatch: "
                    f"q={packed_lens}, kv={packed_k_lens}."
                )
            packed_output, packed_lse = run_segment(
                packed_q,
                packed_k,
                pack_source_halves(
                    early_v,
                    late_v,
                    source_layout.early_lens,
                    source_layout.late_lens,
                ),
                list(packed_lens),
                list(packed_k_lens),
                causal=True,
            )
            new_prev, new_next = split_query_halves(
                packed_output, local_prev_lens, local_next_lens
            )
            new_prev_lse, new_next_lse = split_query_halves(
                packed_lse, local_prev_lens, local_next_lens
            )
            output_prev, lse_prev = merge_state(
                output_prev, lse_prev, new_prev, new_prev_lse
            )
            output_next, lse_next = merge_state(
                output_next, lse_next, new_next, new_next_lse
            )
        elif source_rank < cp_rank:
            # This source's early slab is non-causal for both query
            # halves.  Pack the two query halves per request while keeping
            # the single compact source expanded only once.
            packed_q = pack_query_halves()
            packed_lens = tuple(
                int(prev_len) + int(next_len)
                for prev_len, next_len in zip(
                    local_prev_lens, local_next_lens
                )
            )
            packed_output, packed_lse = run_segment(
                packed_q,
                early_k,
                early_v,
                list(packed_lens),
                list(source_layout.early_lens),
                causal=False,
            )
            new_prev, new_next = split_query_halves(
                packed_output, local_prev_lens, local_next_lens
            )
            new_prev_lse, new_next_lse = split_query_halves(
                packed_lse, local_prev_lens, local_next_lens
            )
            output_prev, lse_prev = merge_state(
                output_prev, lse_prev, new_prev, new_prev_lse
            )
            output_next, lse_next = merge_state(
                output_next, lse_next, new_next, new_next_lse
            )
        else:
            # A later source is visible only to q_next, and its early and
            # late slabs share the same non-causal horizon.
            packed_k = pack_source_halves(
                early_k,
                late_k,
                source_layout.early_lens,
                source_layout.late_lens,
            )
            packed_v = pack_source_halves(
                early_v,
                late_v,
                source_layout.early_lens,
                source_layout.late_lens,
            )
            packed_k_lens = tuple(
                int(early_len) + int(late_len)
                for early_len, late_len in zip(
                    source_layout.early_lens,
                    source_layout.late_lens,
                )
            )
            new_next, new_next_lse = run_segment(
                q_next,
                packed_k,
                packed_v,
                list(local_next_lens),
                list(packed_k_lens),
                causal=False,
            )
            output_next, lse_next = merge_state(
                output_next, lse_next, new_next, new_next_lse
            )

        store_source_cache(
            source_rank, source_latent, source_rope, source_layout
        )

        if requests is not None:
            for request in requests:
                request.wait()
            packed = recv_packed

    prefix_lens = [int(x) for x in forward_batch.extend_prefix_lens_cpu]
    prefix_latent = getattr(forward_batch, "mla_cp_prefix_k", None)
    prefix_rope = getattr(forward_batch, "mla_cp_prefix_k_rope", None)
    if any(prefix_lens):
        if prefix_latent is None or prefix_rope is None:
            raise ValueError("HCU MLA CP ring prefix compact KV is missing.")
        if prefix_latent.shape[0] != sum(prefix_lens):
            raise ValueError("HCU MLA CP ring prefix compact KV length mismatch.")
        prefix_latent_by_req = torch.split(prefix_latent, prefix_lens, dim=0)
        prefix_rope_by_req = torch.split(prefix_rope, prefix_lens, dim=0)
        prev_starts = (0, *accumulate(local_prev_lens))
        next_starts = (0, *accumulate(local_next_lens))
        for request_id, prefix_len in enumerate(prefix_lens):
            for block_start in range(0, prefix_len, prefix_block_size):
                block_end = min(prefix_len, block_start + prefix_block_size)
                prefix_k, prefix_v = expand_compact(
                    prefix_latent_by_req[request_id][block_start:block_end],
                    prefix_rope_by_req[request_id][block_start:block_end],
                )
                block_len = block_end - block_start
                prev_slice = slice(prev_starts[request_id], prev_starts[request_id + 1])
                next_slice = slice(next_starts[request_id], next_starts[request_id + 1])
                merged_prev, merged_prev_lse = accumulate_state(
                    output_prev[prev_slice],
                    lse_prev[prev_slice],
                    q_prev[prev_slice],
                    prefix_k,
                    prefix_v,
                    [local_prev_lens[request_id]],
                    [block_len],
                    causal=False,
                )
                merged_next, merged_next_lse = accumulate_state(
                    output_next[next_slice],
                    lse_next[next_slice],
                    q_next[next_slice],
                    prefix_k,
                    prefix_v,
                    [local_next_lens[request_id]],
                    [block_len],
                    causal=False,
                )
                output_prev[prev_slice].copy_(merged_prev)
                lse_prev[prev_slice].copy_(merged_prev_lse)
                output_next[next_slice].copy_(merged_next)
                lse_next[next_slice].copy_(merged_next_lse)

    if output_prev is None or output_next is None:
        raise RuntimeError("HCU MLA CP ring did not initialize both query halves.")
    return torch.cat((output_prev, output_next), dim=0)


__all__ = [
    "HCUMLACPRingSourceLayout",
    "build_hcu_mla_cp_ring_cache_locs",
    "build_hcu_mla_cp_ring_source_layouts",
    "clear_hcu_mla_cp_ring_state",
    "get_zigzag_cp_rank_chunk_indices",
    "hcu_mla_use_ring_prefill_cp",
    "run_hcu_mla_cp_ring",
    "select_mha_prefix_kv_indices",
]
