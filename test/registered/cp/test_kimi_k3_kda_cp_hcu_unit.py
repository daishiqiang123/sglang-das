import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


_KDA_CP_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/attention/linear/kda_cp.py"
)
_SPEC = importlib.util.spec_from_file_location("_sglang_kda_cp_under_test", _KDA_CP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_KDA_CP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KDA_CP)
_compose_affine_states = _KDA_CP._compose_affine_states
prepare_kda_cp_conv_states = _KDA_CP.prepare_kda_cp_conv_states


class _FakeGatherGroup:
    def __init__(self, all_rank_tensors, rank):
        self.all_rank_tensors = all_rank_tensors
        self.rank = rank

    def all_gather_into_tensor(self, output, input_tensor):
        torch.testing.assert_close(input_tensor, self.all_rank_tensors[self.rank])
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class _FakeTritonKernel:
    def __init__(self):
        self.launch_kwargs = None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launch_kwargs = kwargs

        return launch


class _ContextParallelExtendMode:
    @staticmethod
    def is_context_parallel_extend():
        return True

    @staticmethod
    def is_mixed():
        return False

    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_draft_extend_v2():
        return False


def _parallel(rank, all_rank_tensors):
    return SimpleNamespace(
        attn_cp_size=2,
        attn_cp_rank=rank,
        attn_cp_group=_FakeGatherGroup(all_rank_tensors, rank),
    )


def _metadata(rank):
    local_lens = ([3], [3]) if rank == 0 else ([2], [2])
    return SimpleNamespace(
        bs=1,
        split_list=[3, 2, 2, 3],
        cp_reverse_index=[0, 3, 1, 2],
        reverse_split_len=[3, 3, 2, 2],
        per_rank_actual_token=[6, 4],
        max_rank_len=[6],
        actual_seq_q_prev_list=local_lens[0],
        actual_seq_q_next_list=local_lens[1],
        cu_seqlens_q_combined_tensor=torch.tensor(
            [0, local_lens[0][0], sum(x[0] for x in local_lens)],
            dtype=torch.int32,
        ),
    )


def test_kda_prefill_cp_gate_is_off_when_cp_v2_is_inactive():
    forward_batch = SimpleNamespace(
        forward_mode=_ContextParallelExtendMode(),
        attn_cp_metadata=_metadata(0),
    )
    parallel = SimpleNamespace(attn_cp_size=2)
    with (
        patch.object(_KDA_CP, "get_parallel", return_value=parallel),
        patch.object(_KDA_CP, "is_hcu", return_value=True),
        patch.object(_KDA_CP, "is_cp_v2_active", return_value=False),
    ):
        assert not _KDA_CP.kda_use_prefill_cp(forward_batch)


def test_kda_prefill_cp_gate_rejects_empty_zigzag_segments():
    metadata = _metadata(0)
    metadata.split_list = [3, 0, 2, 3]
    forward_batch = SimpleNamespace(
        forward_mode=_ContextParallelExtendMode(),
        attn_cp_metadata=metadata,
    )
    parallel = SimpleNamespace(attn_cp_size=2)
    with (
        patch.object(_KDA_CP, "get_parallel", return_value=parallel),
        patch.object(_KDA_CP, "is_hcu", return_value=True),
        patch.object(_KDA_CP, "is_cp_v2_active", return_value=True),
    ):
        assert not _KDA_CP.kda_use_prefill_cp(forward_batch)


def test_kda_prefill_cp_gate_accepts_hcu_cp_v2_extend():
    forward_batch = SimpleNamespace(
        forward_mode=_ContextParallelExtendMode(),
        attn_cp_metadata=_metadata(0),
    )
    parallel = SimpleNamespace(attn_cp_size=2)
    with (
        patch.object(_KDA_CP, "get_parallel", return_value=parallel),
        patch.object(_KDA_CP, "is_hcu", return_value=True),
        patch.object(_KDA_CP, "is_cp_v2_active", return_value=True),
    ):
        assert _KDA_CP.kda_use_prefill_cp(forward_batch)


def test_zigzag_conv_initial_and_final_states():
    rank_tails = (
        torch.tensor([[[1.0], [2.0]], [[8.0], [9.0]]]),
        torch.tensor([[[3.0], [4.0]], [[5.0], [6.0]]]),
    )
    local_inputs = (
        torch.tensor([[0.0], [1.0], [2.0], [7.0], [8.0], [9.0]]),
        torch.tensor([[3.0], [4.0], [5.0], [6.0]]),
    )
    expected_initial = (
        torch.tensor([[[-2.0], [-1.0]], [[5.0], [6.0]]]),
        torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
    )

    for rank in range(2):
        forward_batch = SimpleNamespace(attn_cp_metadata=_metadata(rank))
        conv_pool = torch.tensor([[[-2.0], [-1.0]]])
        with patch.object(
            _KDA_CP, "get_parallel", return_value=_parallel(rank, rank_tails)
        ):
            initial, final, cu_seqlens, lens = prepare_kda_cp_conv_states(
                local_inputs[rank],
                conv_pool,
                torch.tensor([0], dtype=torch.int32),
                forward_batch,
            )

        torch.testing.assert_close(initial, expected_initial[rank])
        torch.testing.assert_close(final, torch.tensor([[[8.0], [9.0]]]))
        assert lens == ([3, 3] if rank == 0 else [2, 2])
        torch.testing.assert_close(
            cu_seqlens,
            torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32),
        )


def test_zigzag_affine_composition_order():
    # Natural segments implement state_out = constant + state_in * 2.
    rank_affine = (
        torch.tensor([[[[1.0], [2.0]]], [[[4.0], [2.0]]]]),
        torch.tensor([[[[2.0], [2.0]]], [[[3.0], [2.0]]]]),
    )
    expected_initial = (
        torch.tensor([10.0, 91.0]),
        torch.tensor([21.0, 44.0]),
    )

    for rank in range(2):
        with patch.object(
            _KDA_CP, "get_parallel", return_value=_parallel(rank, rank_affine)
        ):
            initial, final = _compose_affine_states(
                rank_affine[rank],
                torch.tensor([[[[10.0]]]]),
                SimpleNamespace(bs=1),
            )

        torch.testing.assert_close(initial[:, 0, 0, 0], expected_initial[rank])
        torch.testing.assert_close(final, torch.tensor([[[[186.0]]]]))


def test_zigzag_affine_composition_promotes_bf16_state():
    rank_affine = (
        torch.tensor([[[[1.0], [1.0]]], [[[2.0], [1.0]]]]),
        torch.tensor([[[[3.0], [1.0]]], [[[4.0], [1.0]]]]),
    )

    for rank in range(2):
        with patch.object(
            _KDA_CP, "get_parallel", return_value=_parallel(rank, rank_affine)
        ):
            local_initial, final = _compose_affine_states(
                rank_affine[rank],
                torch.tensor([[[[0.5]]]], dtype=torch.bfloat16),
                SimpleNamespace(bs=1),
            )

        assert local_initial.dtype == torch.float32
        assert final.dtype == torch.bfloat16


def test_affine_preprocess_does_not_materialize_chunk_states():
    from sglang.kernels.ops.attention.fla import chunk_delta_h

    fake_kernel = _FakeTritonKernel()
    k = torch.zeros(1, 2, 1, 2)
    w = torch.zeros_like(k)
    u = torch.zeros(1, 2, 1, 3)
    initial_state = torch.zeros(1, 1, 3, 2)
    state_indices = torch.tensor([0], dtype=torch.int32)

    with (
        patch.object(chunk_delta_h, "_USE_KDA_HCU_FROM_TRITON", False),
        patch.object(chunk_delta_h, "_USE_KDA_HCU_FROM_AITER", False),
        patch.object(
            chunk_delta_h,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            fake_kernel,
        ),
    ):
        h, v_new = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            initial_state=initial_state,
            initial_state_indices=state_indices,
            save_new_value=False,
            materialize_chunk_states=False,
        )

    assert h is None
    assert v_new is None
    assert fake_kernel.launch_kwargs["h"].numel() == 1
    assert fake_kernel.launch_kwargs["STORE_CHUNK_STATE"] is False
