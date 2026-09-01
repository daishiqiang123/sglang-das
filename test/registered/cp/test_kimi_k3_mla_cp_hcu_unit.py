import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.mla_cp import (
    build_hcu_mla_cp_varlen_halves,
    pack_hcu_mla_cp_varlen_kv,
)


def _forward_batch(prefix_len=0):
    metadata = SimpleNamespace(
        bs=1,
        # Natural blocks: rank0 owns 0/3, rank1 owns 1/2.
        split_list=[3, 2, 2, 3],
        kv_len_prev_list=[5],
        kv_len_next_list=[7],
        total_q_prev_tokens=2,
        total_q_next_tokens=2,
        cu_seqlens_q_prev_tensor=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_q_next_tensor=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_kv_prev_tensor=torch.tensor([0, 5], dtype=torch.int32),
        cu_seqlens_kv_next_tensor=torch.tensor([0, 7], dtype=torch.int32),
        max_seqlen_q_prev=2,
        max_seqlen_q_next=2,
        cu_seqlens_q_combined_tensor=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_kv_combined_tensor=torch.tensor(
            [0, 5, 12], dtype=torch.int32
        ),
        max_seqlen_q_combined=2,
    )
    return SimpleNamespace(
        attn_cp_metadata=metadata,
        extend_prefix_lens_cpu=[prefix_len],
    )


def test_pack_local_q_varlen_kv_prefixes_in_zigzag_order():
    full_k = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    full_v = full_k + 100

    packed_k, packed_v, cu_q, cu_k, max_q, max_k = pack_hcu_mla_cp_varlen_kv(
        full_k, full_v, _forward_batch(), cp_size=2
    )

    torch.testing.assert_close(
        packed_k.flatten(),
        torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], dtype=torch.float32),
    )
    torch.testing.assert_close(packed_v, packed_k + 100)
    torch.testing.assert_close(cu_q, torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(cu_k, torch.tensor([0, 5, 12], dtype=torch.int32))
    assert max_q == 2
    assert max_k == 7


def test_batch_one_halves_are_natural_kv_prefix_views():
    full_k = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    full_v = full_k + 100

    prev, next_ = build_hcu_mla_cp_varlen_halves(
        full_k, full_v, _forward_batch(), cp_size=2
    )

    assert (prev.q_start, prev.q_end) == (0, 2)
    assert (next_.q_start, next_.q_end) == (2, 4)
    assert prev.k.untyped_storage().data_ptr() == full_k.untyped_storage().data_ptr()
    assert next_.k.untyped_storage().data_ptr() == full_k.untyped_storage().data_ptr()
    assert prev.v.untyped_storage().data_ptr() == full_v.untyped_storage().data_ptr()
    assert next_.v.untyped_storage().data_ptr() == full_v.untyped_storage().data_ptr()
    torch.testing.assert_close(prev.k, full_k[:5])
    torch.testing.assert_close(next_.k, full_k[:7])


def test_local_q_varlen_rejects_prefix_cache_until_expansion_exists():
    full_k = torch.zeros(10, 1, 1)
    full_v = torch.zeros_like(full_k)
    with pytest.raises(NotImplementedError, match="prefix-cache"):
        pack_hcu_mla_cp_varlen_kv(
            full_k,
            full_v,
            _forward_batch(prefix_len=4),
            cp_size=2,
        )


def test_local_q_prefix_attention_matches_full_causal_attention():
    torch.manual_seed(7)
    full_q = torch.randn(10, 1, 4)
    full_k = torch.randn(10, 1, 4)
    full_v = torch.randn(10, 1, 3)
    # CP rank 1 owns natural blocks [3:5] and [5:7].
    local_q = torch.cat((full_q[3:5], full_q[5:7]), dim=0)

    packed_k, packed_v, cu_q, cu_k, _, _ = pack_hcu_mla_cp_varlen_kv(
        full_k, full_v, _forward_batch(), cp_size=2
    )

    local_outputs = []
    for segment in range(2):
        q_start, q_end = int(cu_q[segment]), int(cu_q[segment + 1])
        k_start, k_end = int(cu_k[segment]), int(cu_k[segment + 1])
        q_segment = local_q[q_start:q_end, 0]
        k_segment = packed_k[k_start:k_end, 0]
        v_segment = packed_v[k_start:k_end, 0]
        scores = q_segment @ k_segment.T / math.sqrt(q_segment.shape[-1])
        # FlashAttention aligns a shorter causal Q block to the bottom-right
        # of its KV prefix.  Query row i therefore sees keys through
        # ``kv_len - q_len + i``.
        q_len = q_end - q_start
        kv_len = k_end - k_start
        causal = torch.arange(kv_len).unsqueeze(0) <= (
            kv_len - q_len + torch.arange(q_len).unsqueeze(1)
        )
        probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
        local_outputs.append(probabilities @ v_segment)
    local_output = torch.cat(local_outputs, dim=0)

    full_scores = full_q[:, 0] @ full_k[:, 0].T / math.sqrt(full_q.shape[-1])
    full_mask = torch.arange(10).unsqueeze(0) <= torch.arange(10).unsqueeze(1)
    full_output = torch.softmax(
        full_scores.masked_fill(~full_mask, -torch.inf), dim=-1
    ) @ full_v[:, 0]
    torch.testing.assert_close(local_output, full_output[3:7], rtol=1e-5, atol=1e-6)


def test_pack_varlen_kv_keeps_request_order_for_batch_two():
    metadata = SimpleNamespace(
        bs=2,
        split_list=[3, 2, 2, 3, 2, 2, 2, 2],
        kv_len_prev_list=[5, 4],
        kv_len_next_list=[7, 6],
        total_q_prev_tokens=4,
        total_q_next_tokens=4,
        cu_seqlens_q_prev_tensor=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_q_next_tensor=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_kv_prev_tensor=torch.tensor([0, 5, 9], dtype=torch.int32),
        cu_seqlens_kv_next_tensor=torch.tensor([0, 7, 13], dtype=torch.int32),
        max_seqlen_q_prev=2,
        max_seqlen_q_next=2,
        cu_seqlens_q_combined_tensor=torch.tensor(
            [0, 2, 4, 6, 8], dtype=torch.int32
        ),
        cu_seqlens_kv_combined_tensor=torch.tensor(
            [0, 5, 9, 16, 22], dtype=torch.int32
        ),
        max_seqlen_q_combined=2,
    )
    forward_batch = SimpleNamespace(
        attn_cp_metadata=metadata,
        extend_prefix_lens_cpu=[0, 0],
    )
    full_k = torch.cat(
        (
            torch.arange(10, dtype=torch.float32),
            torch.arange(100, 108, dtype=torch.float32),
        )
    ).view(-1, 1, 1)
    full_v = full_k + 1000

    packed_k, packed_v, _, cu_k, _, max_k = pack_hcu_mla_cp_varlen_kv(
        full_k, full_v, forward_batch, cp_size=2
    )
    expected = torch.tensor(
        [
            *range(5),
            *range(100, 104),
            *range(7),
            *range(100, 106),
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(packed_k.flatten(), expected)
    torch.testing.assert_close(packed_v, packed_k + 1000)
    torch.testing.assert_close(
        cu_k, torch.tensor([0, 5, 9, 16, 22], dtype=torch.int32)
    )
    assert max_k == 7
