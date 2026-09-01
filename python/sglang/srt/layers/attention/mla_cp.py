"""HCU MLA helpers for true zigzag prefill context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HCUMLACPVarlenHalf:
    q_start: int
    q_end: int
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def build_hcu_mla_cp_varlen_halves(
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    forward_batch: Any,
    *,
    cp_size: int,
) -> tuple[HCUMLACPVarlenHalf, HCUMLACPVarlenHalf]:
    """Build prev/next KV-prefix views for two local-Q varlen calls."""
    cp_meta = forward_batch.attn_cp_metadata
    if cp_meta is None:
        raise ValueError("HCU MLA PCP requires zigzag CP metadata.")

    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix_lens is not None and any(int(length) != 0 for length in prefix_lens):
        raise NotImplementedError(
            "HCU MLA local-Q varlen PCP does not yet support prefix-cache hits."
        )

    bs = int(cp_meta.bs)
    blocks_per_request = 2 * cp_size
    full_lens = [
        sum(
            int(length)
            for length in cp_meta.split_list[
                request_id * blocks_per_request : (request_id + 1)
                * blocks_per_request
            ]
        )
        for request_id in range(bs)
    ]
    if sum(full_lens) != full_k.shape[0] or full_k.shape[0] != full_v.shape[0]:
        raise ValueError(
            "HCU MLA PCP natural KV geometry mismatch: "
            f"lengths={full_lens}, k={full_k.shape[0]}, v={full_v.shape[0]}."
        )

    k_by_request = torch.split(full_k, full_lens, dim=0)
    v_by_request = torch.split(full_v, full_lens, dim=0)
    half_kv_lens = (
        [int(length) for length in cp_meta.kv_len_prev_list],
        [int(length) for length in cp_meta.kv_len_next_list],
    )
    half_q_ranges = (
        (0, int(cp_meta.total_q_prev_tokens)),
        (
            int(cp_meta.total_q_prev_tokens),
            int(cp_meta.total_q_prev_tokens + cp_meta.total_q_next_tokens),
        ),
    )
    half_cu_q = (
        cp_meta.cu_seqlens_q_prev_tensor,
        cp_meta.cu_seqlens_q_next_tensor,
    )
    half_cu_k = (
        cp_meta.cu_seqlens_kv_prev_tensor,
        cp_meta.cu_seqlens_kv_next_tensor,
    )
    half_max_q = (cp_meta.max_seqlen_q_prev, cp_meta.max_seqlen_q_next)

    halves = []
    for half in range(2):
        k_prefixes = []
        v_prefixes = []
        for request_id, kv_len in enumerate(half_kv_lens[half]):
            if kv_len < 0 or kv_len > full_lens[request_id]:
                raise ValueError(
                    "HCU MLA PCP KV prefix is outside the natural request: "
                    f"request={request_id}, kv_len={kv_len}, "
                    f"full_len={full_lens[request_id]}."
                )
            k_prefixes.append(k_by_request[request_id][:kv_len])
            v_prefixes.append(v_by_request[request_id][:kv_len])
        # The serving hot path is bs=1; preserve a view of natural K/V instead
        # of copying a prompt-sized prefix.  Multi-request batches require one
        # packed tensor per half for FlashAttention varlen.
        packed_k = k_prefixes[0] if bs == 1 else torch.cat(k_prefixes, dim=0)
        packed_v = v_prefixes[0] if bs == 1 else torch.cat(v_prefixes, dim=0)
        halves.append(
            HCUMLACPVarlenHalf(
                q_start=half_q_ranges[half][0],
                q_end=half_q_ranges[half][1],
                k=packed_k,
                v=packed_v,
                cu_seqlens_q=half_cu_q[half],
                cu_seqlens_k=half_cu_k[half],
                max_seqlen_q=int(half_max_q[half]),
                max_seqlen_k=max(half_kv_lens[half], default=0),
            )
        )
    return halves[0], halves[1]


def pack_hcu_mla_cp_varlen_kv(
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    forward_batch: Any,
    *,
    cp_size: int,
):
    """Pack natural-order KV prefixes for rank-local zigzag MLA queries.

    For a query block at natural positions ``[start, end)``, FlashAttention's
    bottom-right causal alignment is exact with KV prefix ``[0, end)``.  Q stays
    rank-local while K/V are packed as ``2 * batch_size`` varlen sequences.

    Prefix-cache hits are rejected initially because cached MLA tokens remain
    latent and are not expanded through ``kv_b_proj`` for this path.
    """
    cp_meta = forward_batch.attn_cp_metadata
    halves = build_hcu_mla_cp_varlen_halves(
        full_k, full_v, forward_batch, cp_size=cp_size
    )
    kv_lens = [
        *[int(length) for length in cp_meta.kv_len_prev_list],
        *[int(length) for length in cp_meta.kv_len_next_list],
    ]

    return (
        torch.cat([half.k for half in halves], dim=0),
        torch.cat([half.v for half in halves], dim=0),
        cp_meta.cu_seqlens_q_combined_tensor,
        cp_meta.cu_seqlens_kv_combined_tensor,
        cp_meta.max_seqlen_q_combined,
        max(kv_lens, default=0),
    )


__all__ = [
    "HCUMLACPVarlenHalf",
    "build_hcu_mla_cp_varlen_halves",
    "pack_hcu_mla_cp_varlen_kv",
]
