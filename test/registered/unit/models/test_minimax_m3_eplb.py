from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerCore
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.models import minimax_m3
from sglang.srt.models.minimax_m3 import MiniMaxM3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def test_minimax_m3_marks_its_runner_for_cached_weight_layouts(monkeypatch):
    class FakeExperts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.moe_runner_config = MoeRunnerConfig()
            self.runner_core = AiterRunnerCore(self.moe_runner_config)

    parallel = SimpleNamespace(
        moe_ep_size=1, moe_tp_size=1, tp_size=1, moe_tp_rank=0, tp_rank=0
    )
    backend = SimpleNamespace(is_none=lambda: True, is_deepep=lambda: False)
    monkeypatch.setattr(minimax_m3, "get_parallel", lambda: parallel)
    monkeypatch.setattr(minimax_m3, "get_moe_a2a_backend", lambda: backend)
    monkeypatch.setattr(
        minimax_m3,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0)),
    )
    monkeypatch.setattr(minimax_m3, "is_shared_experts_fusion_disabled", lambda: True)
    monkeypatch.setattr(minimax_m3, "has_per_rank_fused_shared_slots", lambda _: False)
    monkeypatch.setattr(minimax_m3, "get_moe_impl_class", lambda _: FakeExperts)
    monkeypatch.setattr(minimax_m3, "TopK", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(
        minimax_m3,
        "ReplicatedLinear",
        lambda in_features, out_features, **kwargs: nn.Linear(
            in_features, out_features, bias=False
        ),
    )
    config = SimpleNamespace(
        num_local_experts=2,
        n_shared_experts=None,
        num_experts_per_tok=1,
        hidden_size=4,
        intermediate_size=8,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        scoring_func="softmax",
    )

    model = MiniMaxM3MoE(config, layer_id=0)

    assert model.experts.moe_runner_config.is_minimax_m3
    assert model.experts.runner_core.config is model.experts.moe_runner_config
    assert not MoeRunnerConfig().is_minimax_m3


def test_standard_ep_router_applies_expert_location_dispatch():
    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 128)
    topk_output = object()
    expected_output = torch.randn(2, 8)
    dispatch_info = object()
    moe = SimpleNamespace(
        layer_id=11,
        _compute_router_logits=Mock(return_value=router_logits),
        topk=Mock(return_value=topk_output),
        experts=Mock(return_value=expected_output),
    )

    with patch(
        "sglang.srt.models.minimax_m3.ExpertLocationDispatchInfo.init_new",
        return_value=dispatch_info,
    ) as init_dispatch:
        output = MiniMaxM3MoE._forward_router_experts(moe, hidden_states)

    init_dispatch.assert_called_once_with(layer_id=11)
    moe.topk.assert_called_once_with(
        hidden_states,
        router_logits,
        expert_location_dispatch_info=dispatch_info,
    )
    moe.experts.assert_called_once_with(hidden_states, topk_output)
    assert output is expected_output


@pytest.mark.parametrize("fused_shared", [0, 1])
@pytest.mark.parametrize("packed", [False, True])
def test_eplb_exports_live_routed_weights_and_packed_buffers(fused_shared, packed):
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_local_experts = 4
            self.w13_weight = nn.Parameter(torch.arange(8.0).reshape(4, 2))
            self.w2_weight = nn.Parameter(torch.ones(4, 2))
            self.w13_weight_scale = nn.Parameter(torch.ones(4, 1))
            self.correction_bias = nn.Parameter(torch.ones(4))
            self.scalar_metadata = nn.Parameter(torch.tensor(1.0))
            self.other_metadata = nn.Parameter(torch.ones(2))
            self.global_weight = nn.Parameter(torch.ones(4, 2))
            self.global_weight._sglang_require_global_experts = True
            self.register_buffer("unrelated_buffer", torch.ones(4, 2))
            if packed:
                self.register_buffer("w13_weight_deepgemm", torch.ones(4, 3))
                self.register_buffer("w2_weight_deepgemm_masked", torch.ones(4, 5))

    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.experts = Experts()
    moe.num_fused_shared_experts = fused_shared
    expected = [
        moe.experts.w13_weight,
        moe.experts.w2_weight,
        moe.experts.w13_weight_scale,
    ]
    if packed:
        expected += [
            moe.experts.w13_weight_deepgemm,
            moe.experts.w2_weight_deepgemm_masked,
        ]
    exported = moe.get_moe_weights()
    assert len(exported) == len(expected)
    for view, weight in zip(exported, expected):
        assert view.shape == weight[: 4 - fused_shared].shape
        assert view.data_ptr() == weight.data_ptr()
        assert view.stride() == weight.stride()
        before_shared = weight[-1].clone()
        view.add_(10)
        if fused_shared:
            assert torch.equal(weight[-1], before_shared)

    # Migration must resolve replacements made after quantization, not retain
    # stale Parameter references from model construction.
    previous = moe.experts.w13_weight
    moe.experts.w13_weight = nn.Parameter(torch.full((4, 2), 99.0))
    refreshed = moe.get_moe_weights()[0]
    assert refreshed.data_ptr() == moe.experts.w13_weight.data_ptr()
    assert refreshed.data_ptr() != previous.data_ptr()


def test_eplb_model_and_causal_lm_export_only_local_routed_layers():
    from sglang.srt.models.minimax_m3 import MiniMaxM3Model, MiniMaxM3SparseForCausalLM

    def routed_layer():
        moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
        nn.Module.__init__(moe)
        moe.experts = nn.Module()
        moe.experts.num_local_experts = 2
        moe.experts.w13_weight = nn.Parameter(torch.ones(2, 3))
        moe.num_fused_shared_experts = 0
        layer = nn.Module()
        layer.mlp = moe
        return layer

    model = MiniMaxM3Model.__new__(MiniMaxM3Model)
    nn.Module.__init__(model)
    dense = nn.Module()
    dense.mlp = nn.Linear(3, 3)
    model.layers = nn.ModuleList(
        [routed_layer(), routed_layer(), dense, routed_layer()]
    )
    model.start_layer, model.end_layer = 1, 4
    wrapper = MiniMaxM3SparseForCausalLM.__new__(MiniMaxM3SparseForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    exported = wrapper.routed_experts_weights_of_layer
    assert set(exported) == {1, 3}
    for layer_id in [1, 3]:
        assert (
            exported[layer_id][0].data_ptr()
            == model.layers[layer_id].mlp.experts.w13_weight.data_ptr()
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
