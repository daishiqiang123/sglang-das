"""CPU control-flow contracts for MiniMax prefill shared overlap.

Load real method bodies without importing vendor modules that initialize GPUs.
Native stream/kernel performance is covered by the saved CP8 model profiles.
"""

import __future__
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")
REPO = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
SRT = REPO / "python/sglang/srt"


def load_function(path, name, scope, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("method", ["off", "stream", "split"])
def test_method_selector(method):
    scope = {
        "envs": SimpleNamespace(
            SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=SimpleNamespace(
                get=lambda: method
            )
        )
    }
    read = load_function(
        SRT / "models/minimax_m3.py", "_get_prefill_shared_overlap_method", scope
    )
    assert read() == method


def test_invalid_method_is_rejected():
    scope = {
        "envs": SimpleNamespace(
            SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=SimpleNamespace(
                get=lambda: "typo"
            )
        )
    }
    read = load_function(
        SRT / "models/minimax_m3.py", "_get_prefill_shared_overlap_method", scope
    )
    with pytest.raises(ValueError, match="off, stream, or split"):
        read()


def run_forward(method, rows=2, fail=False, invoke_hook=True, decode=False):
    events, handles = [], []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait", self.name, other.name))

    main, alt = Stream("main"), Stream("alt")

    @contextmanager
    def stream(value):
        events.append(("enter", value.name))
        yield

    class Handle:
        removed = False

        def remove(self):
            self.removed = True

    class Experts:
        callback = None

        def __init__(self):
            self.dispatcher = self

        def register_deepep_normal_combine_hook(self, callback):
            self.callback = callback
            handle = Handle()
            handles.append(handle)
            return [handle]

        def __call__(self, hidden, topk):
            events.append("routed")
            if self.callback and invoke_hook:
                events.append("combine")
                self.callback(self)
            if fail:
                raise RuntimeError("routed failure")
            return hidden * 5

    class TopK:
        def __call__(self, *args, **kwargs):
            return None

        def empty_topk_output(self, device, **kwargs):
            events.append(("empty", kwargs.get("layer_id")))
            return None

    def up(hidden):
        events.append("up")
        return hidden * 2, None

    def down(hidden):
        events.append("down")
        return hidden * 3

    def full(hidden):
        events.append("full")
        return hidden * 6

    mode = SimpleNamespace(is_normal=lambda: not decode, is_low_latency=lambda: decode)
    scope = {
        "torch": torch,
        "_is_npu": False,
        "get_is_extend_in_batch": lambda: not decode,
        "get_deepep_mode": lambda: SimpleNamespace(resolve=lambda _: mode),
        "ExpertLocationDispatchInfo": SimpleNamespace(init_new=lambda **_: None),
    }
    forward = load_function(
        SRT / "models/minimax_m3.py", "forward_deepep", scope, "MiniMaxM3MoE"
    )
    moe = SimpleNamespace(
        enable_deepep_shared_expert_overlap=method != "off",
        prefill_shared_overlap_method=method,
        split_shared_experts=decode,
        alt_stream=alt,
        shared_experts=SimpleNamespace(gate_up_proj=up, forward_down=down),
        _forward_shared_experts=full,
        _compute_router_logits=lambda x: x,
        topk=TopK(),
        experts=Experts(),
        use_mega_moe=False,
        layer_id=7,
        _run_shared_in_recv_windows=lambda x, _: (x * 5, x * 6),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: not decode),
        moe_num_token_non_padded=lambda: None,
    )
    hidden = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", stream),
    ):
        try:
            actual = forward(moe, hidden, batch)
        except RuntimeError:
            assert all(h.removed for h in handles)
            raise
    torch.testing.assert_close(actual, hidden * (5 if rows == 0 else 11))
    assert all(h.removed for h in handles)
    return events


@pytest.mark.parametrize("method", ["off", "stream", "split"])
def test_three_modes_preserve_output_and_schedule(method):
    events = run_forward(method)
    if method == "off":
        assert not any(isinstance(e, tuple) and e[0] == "enter" for e in events)
        assert events.index("full") < events.index("routed")
    elif method == "stream":
        assert ("enter", "alt") in events
        assert events.index("full") < events.index("routed")
        assert ("wait", "main", "alt") in events
    else:
        assert "full" not in events
        assert (
            events.index("up")
            < events.index("routed")
            < events.index("combine")
            < events.index("down")
        )
        assert ("wait", "main", "alt") in events


@pytest.mark.parametrize("method", ["off", "stream", "split"])
def test_empty_rank_skips_shared_work(method):
    events = run_forward(method, rows=0)
    assert any(isinstance(e, tuple) and e[0] == "empty" for e in events)
    assert not any(e in events for e in ["up", "down", "full", "combine"])


def test_split_cleanup_after_routed_failure():
    with pytest.raises(RuntimeError, match="routed failure"):
        run_forward("split", fail=True)


def test_missing_combine_hook_does_not_drop_shared_output():
    with pytest.raises(RuntimeError, match="did not run"):
        run_forward("split", invoke_hook=False)


@pytest.mark.parametrize("method", ["off", "stream", "split"])
def test_prefill_selector_does_not_change_decode_hooks(method):
    assert run_forward(method, decode=True) == []
