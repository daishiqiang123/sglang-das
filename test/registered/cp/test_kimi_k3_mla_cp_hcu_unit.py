from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.mla_cp import (
    build_hcu_mla_cp_ring_cache_locs,
    build_hcu_mla_cp_ring_source_layouts,
    get_zigzag_cp_rank_chunk_indices,
    get_zigzag_mla_cp_ring_visibility,
    select_mha_prefix_kv_indices,
)


def test_ring_source_layouts_follow_zigzag_natural_chunks():
    metadata = SimpleNamespace(bs=1, split_list=[3, 2, 2, 3])

    layouts = build_hcu_mla_cp_ring_source_layouts(metadata, cp_size=2)

    assert get_zigzag_cp_rank_chunk_indices(1, 2, 0) == [0, 3]
    assert get_zigzag_cp_rank_chunk_indices(1, 2, 1) == [1, 2]
    assert layouts[0].token_count == 6
    assert layouts[0].early_lens == (3,)
    assert layouts[0].late_lens == (3,)
    assert layouts[1].token_count == 4
    assert layouts[1].early_lens == (2,)
    assert layouts[1].late_lens == (2,)


def test_ring_cache_locations_restore_natural_global_positions():
    metadata = SimpleNamespace(bs=1, split_list=[3, 2, 2, 3])
    cache_locs = torch.arange(100, 110)

    by_rank = build_hcu_mla_cp_ring_cache_locs(
        cache_locs, metadata, cp_size=2
    )

    torch.testing.assert_close(
        by_rank[0], torch.tensor([100, 101, 102, 107, 108, 109])
    )
    torch.testing.assert_close(
        by_rank[1], torch.tensor([103, 104, 105, 106])
    )


def test_ring_visibility_matches_zigzag_causal_geometry():
    # rank0: source0 early is diagonal; source1 early is in the future.
    assert get_zigzag_mla_cp_ring_visibility(0, 0) == (True, True, True)
    assert get_zigzag_mla_cp_ring_visibility(0, 1) == (False, True, True)
    # rank1: source0 early is history; source0 late is in the future.
    assert get_zigzag_mla_cp_ring_visibility(1, 0) == (True, True, False)
    assert get_zigzag_mla_cp_ring_visibility(1, 1) == (True, True, True)


def test_select_mha_prefix_indices_keeps_each_request_prefix_only():
    indices = torch.tensor([10, 11, 12, 13, 20, 21, 22])

    selected = select_mha_prefix_kv_indices(
        indices,
        seq_lens=[4, 3],
        prefix_lens=[2, 1],
    )

    torch.testing.assert_close(selected, torch.tensor([10, 11, 20]))
