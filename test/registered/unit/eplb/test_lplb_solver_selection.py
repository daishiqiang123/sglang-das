"""CPU contracts for model-independent LPLB solver selection."""

import __future__
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")
REPO = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
SRT = REPO / "python/sglang/srt"


def load_function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("inherited_mode", ["megamoe", "invalid"])
@pytest.mark.parametrize("custom_solver", [False, True])
def test_model_hook_is_generic_and_reused_on_rebalance(
    monkeypatch, inherited_mode, custom_solver
):
    monkeypatch.setenv("SGLANG_MINIMAX_MOE_MODE", inherited_mode)
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_exec = lambda: SimpleNamespace(
        moe=SimpleNamespace(ep_dispatch_algorithm="lp")
    )
    runtime.get_parallel = lambda: SimpleNamespace(moe_ep_group=object())
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    loader = ModuleType("sglang.srt.model_loader.utils")
    factory = object()
    hook = Mock(return_value=factory)
    model_cls = type(
        "GenericModel",
        (),
        {"get_lplb_device_solver_factory": hook} if custom_solver else {},
    )
    loader.get_model_architecture = Mock(return_value=(model_cls, "GenericModel"))
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    solver = Mock()
    save = Mock()
    metadata = SimpleNamespace(
        num_layers=2,
        ep_size=2,
        physical_to_logical_map=[object(), object()],
        logical_to_all_physical_map=[object(), object()],
        logical_to_all_physical_map_num_valid=[object(), object()],
    )
    scope = {
        "envs": SimpleNamespace(
            SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS=SimpleNamespace(get=lambda: "")
        ),
        "assert_lplb_supported_model": Mock(),
        "get_global_expert_location_metadata": lambda: metadata,
        "clear_global_lplb_solvers": Mock(),
        "LPLBSolver": solver,
        "LoadAwareReplicaSolver": Mock(),
        "set_global_lplb_solver": save,
        "logger": Mock(),
    }
    initialize = load_function(
        SRT / "model_executor/model_runner_components/moe_ep_setup.py",
        "init_lplb_solvers",
        scope,
    )
    cfg = SimpleNamespace(hf_config=SimpleNamespace(architectures=["GenericModel"]))
    initialize(model_config=cfg)
    replacement = object()
    metadata.physical_to_logical_map[0] = replacement
    initialize(model_config=cfg)
    assert solver.call_count == 4 and save.call_count == 4
    assert solver.call_args_list[2].kwargs["phy2log"] is replacement
    assert hook.call_count == (2 if custom_solver else 0)
    for call in solver.call_args_list:
        if custom_solver:
            assert call.kwargs["device_solver_factory"] is factory
        else:
            assert "device_solver_factory" not in call.kwargs
    runtime.get_exec = lambda: SimpleNamespace(
        moe=SimpleNamespace(ep_dispatch_algorithm="load_aware")
    )
    loader.get_model_architecture.reset_mock()
    initialize(model_config=cfg)
    loader.get_model_architecture.assert_not_called()
    assert scope["LoadAwareReplicaSolver"].call_count == 2


def load_model_hook(filename, classname, scope):
    path = SRT / "models" / filename
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_lplb_device_solver_factory"
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize(
    "mode,dispatch,selected",
    [
        ("", "lp", False),
        ("megamoe", "static", False),
        ("megamoe", None, False),
        ("megamoe", "lp", True),
    ],
)
def test_minimax_selects_device_only_when_requested(
    monkeypatch, mode, dispatch, selected
):
    import os

    monkeypatch.setenv("SGLANG_MINIMAX_MOE_MODE", mode)
    module = ModuleType("sglang.srt.eplb.lplb_hip_waterfill")
    module.DeviceWaterfill = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    hook = load_model_hook(
        "minimax_m3.py",
        "MiniMaxM3SparseForCausalLM",
        {
            "os": os,
            "_is_hcu": True,
            "get_exec": lambda: SimpleNamespace(
                moe=SimpleNamespace(ep_dispatch_algorithm=dispatch)
            ),
            "get_moe_a2a_backend": lambda: SimpleNamespace(is_megamoe=lambda: True),
        },
    )
    assert hook() is (module.DeviceWaterfill if selected else None)


@pytest.mark.parametrize(
    "mode,backend,hcu",
    [
        ("megamoe_lplb", True, True),
        ("megamoe", False, True),
        ("megamoe", True, False),
    ],
)
def test_minimax_rejects_obsolete_or_incompatible_modes(
    monkeypatch, mode, backend, hcu
):
    import os

    monkeypatch.setenv("SGLANG_MINIMAX_MOE_MODE", mode)
    hook = load_model_hook(
        "minimax_m3.py",
        "MiniMaxM3SparseForCausalLM",
        {
            "os": os,
            "_is_hcu": hcu,
            "get_exec": lambda: SimpleNamespace(
                moe=SimpleNamespace(ep_dispatch_algorithm="lp")
            ),
            "get_moe_a2a_backend": lambda: SimpleNamespace(is_megamoe=lambda: backend),
        },
    )
    with pytest.raises(ValueError):
        hook()


def test_multimodal_wrapper_delegates_the_same_policy():
    factory = object()
    delegate = Mock(return_value=factory)
    hook = load_model_hook(
        "minimax_m3_vl.py",
        "MiniMaxM3SparseForConditionalGeneration",
        {
            "MiniMaxM3SparseForCausalLM": SimpleNamespace(
                get_lplb_device_solver_factory=delegate
            ),
        },
    )
    assert hook() is factory
    delegate.assert_called_once()


def solver_class():
    path = SRT / "eplb/lplb_solver.py"
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LPLBSolver"
    )
    scope = {"torch": torch, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["LPLBSolver"]


def test_device_solver_receives_compact_map_and_routes_counts(monkeypatch):
    device = SimpleNamespace(solve=Mock(return_value="solved"))
    factory = Mock(return_value=device)
    solver = solver_class()(
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([[0, 3, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1]]),
        2,
        device_solver_factory=factory,
    )
    assert factory.call_args.args[1].shape == (3, 2)
    assert solver.dispatch_map.dtype == torch.int32
    assert solver.solve(torch.tensor([[0, 2], [0, 1]])) == "solved"
    assert torch.equal(device.solve.call_args.args[0], torch.tensor([2.0, 1.0, 1.0]))


def test_unset_uses_original_warmup_even_on_hip(monkeypatch):
    module = ModuleType("sglang.kernels.ops.lplb.torch_solver")
    module.warmup = Mock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.version, "hip", "test")
    solver = solver_class()(
        torch.tensor([0, 1, 2, 0]), torch.tensor([[0, 3], [1, -1], [2, -1]]), 2
    )
    assert solver._device_solver is None and solver.dispatch_map is None
    module.warmup.assert_called_once()


def test_static_probabilities_cannot_silently_override_device_solver():
    with pytest.raises(ValueError, match="mutually exclusive"):
        solver_class()(
            torch.tensor([0, 1]),
            torch.tensor([[0], [1]]),
            2,
            static_probabilities=torch.ones(2, 1),
            device_solver_factory=Mock(),
        )
