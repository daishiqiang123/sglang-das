"""HCU MLA helpers for true zigzag prefill context parallelism."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)
_HCU_MLA_RING_TRACE_EMITTED = False
_HCU_MLA_RING_TRACE_KEYS: set[tuple[int, int, int, int]] = set()


@dataclass(frozen=True)
class HCUMLACPRingSourceLayout:
    """Natural early/late token geometry for one compact-KV source rank."""

    token_count: int
    early_token_count: int
    early_lens: tuple[int, ...]
    late_lens: tuple[int, ...]


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


def get_zigzag_mla_cp_ring_visibility(cp_rank: int, source_rank: int):
    """Return early->early, early->late and late->late visibility."""
    return source_rank <= cp_rank, True, source_rank >= cp_rank


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
    global _HCU_MLA_RING_TRACE_EMITTED

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
    if not _HCU_MLA_RING_TRACE_EMITTED:
        logger.info(
            "KIMI_MLA_PCP_RING active: cp_rank=%s, cp_size=%s, "
            "local_q_tokens=%s, local_compact_kv_tokens=%s, "
            "max_ring_payload_tokens=%s, prefix_tokens=%s; "
            "Q remains local and only compact latent KV is rotated",
            cp_rank,
            cp_size,
            logical_q_tokens,
            local_layout.token_count,
            max_rank_tokens,
            sum(int(x) for x in forward_batch.extend_prefix_lens_cpu),
        )
        _HCU_MLA_RING_TRACE_EMITTED = True

    # Bounded per-forward evidence: one line for the first few MLA layers on
    # each CP rank's attention-TP leader. This distinguishes GSM8K requests
    # from the startup warmup without printing prompts or output tokens.
    from sglang.srt.utils import get_bool_env_var

    if (
        get_bool_env_var("SGLANG_KIMI_CP_TRACE", default="false")
        and getattr(parallel, "attn_tp_rank", -1) == 0
    ):
        trace_key = (
            int(getattr(forward_batch, "batch_size", 0) or 0),
            int(getattr(forward_batch, "extend_num_tokens", 0) or 0),
            int(logical_q_tokens),
            int(local_layout.token_count),
        )
        if trace_key not in _HCU_MLA_RING_TRACE_KEYS and len(_HCU_MLA_RING_TRACE_KEYS) < 128:
            logger.info(
                "KIMI_MLA_PCP_RING_CALL: cp_rank=%s, layer_id=%s, "
                "batch_size=%s, extend_num_tokens=%s, local_q_tokens=%s, "
                "local_compact_kv_tokens=%s, max_ring_payload_tokens=%s, "
                "q_shape=%s, q_dtype=%s, compact_dtype=%s, local_k_shape=%s, "
                "local_v_shape=%s, ring_steps=%s, "
                "full_expanded_kv_materialized=False",
                cp_rank,
                getattr(layer, "layer_id", None),
                getattr(forward_batch, "batch_size", None),
                getattr(forward_batch, "extend_num_tokens", None),
                logical_q_tokens,
                local_layout.token_count,
                max_rank_tokens,
                tuple(q.shape),
                q.dtype,
                local_latent.dtype,
                tuple(local_k.shape),
                tuple(local_v.shape),
                cp_size,
            )
            _HCU_MLA_RING_TRACE_KEYS.add(trace_key)

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
        if ring_step == 0:
            source_k, source_v = local_k, local_v
        else:
            source_k, source_v = expand_compact(source_latent, source_rope)

        early_end = source_layout.early_token_count
        early_k, late_k = source_k[:early_end], source_k[early_end:]
        early_v, late_v = source_v[:early_end], source_v[early_end:]
        early_to_prev, early_to_next, late_to_next = (
            get_zigzag_mla_cp_ring_visibility(cp_rank, source_rank)
        )
        if early_to_prev:
            output_prev, lse_prev = accumulate_state(
                output_prev,
                lse_prev,
                q_prev,
                early_k,
                early_v,
                local_prev_lens,
                source_layout.early_lens,
                causal=source_rank == cp_rank,
            )
        if early_to_next:
            output_next, lse_next = accumulate_state(
                output_next,
                lse_next,
                q_next,
                early_k,
                early_v,
                local_next_lens,
                source_layout.early_lens,
                causal=False,
            )
        if late_to_next:
            output_next, lse_next = accumulate_state(
                output_next,
                lse_next,
                q_next,
                late_k,
                late_v,
                local_next_lens,
                source_layout.late_lens,
                causal=source_rank == cp_rank,
            )

        source_cache_locs = cache_locs_by_rank[source_rank]
        if source_cache_locs.shape[0] != source_layout.token_count:
            raise ValueError("HCU MLA CP ring cache/source token lengths differ.")
        token_to_kv_pool.set_mla_kv_buffer(
            layer, source_cache_locs, source_latent, source_rope
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
    for name in (
        "mla_cp_local_k",
        "mla_cp_local_k_rope",
        "mla_cp_prefix_k",
        "mla_cp_prefix_k_rope",
    ):
        if hasattr(forward_batch, name):
            delattr(forward_batch, name)
    return torch.cat((output_prev, output_next), dim=0)


__all__ = [
    "HCUMLACPRingSourceLayout",
    "build_hcu_mla_cp_ring_cache_locs",
    "build_hcu_mla_cp_ring_source_layouts",
    "get_zigzag_cp_rank_chunk_indices",
    "get_zigzag_mla_cp_ring_visibility",
    "hcu_mla_use_ring_prefill_cp",
    "run_hcu_mla_cp_ring",
    "select_mha_prefix_kv_indices",
]
