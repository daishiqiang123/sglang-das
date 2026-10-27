from types import SimpleNamespace

import pytest
import torch
from sglang.srt.eplb import expert_distribution as ed
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


@pytest.mark.parametrize("global_counts", [False, True])
@pytest.mark.parametrize(
    "original,padded,expected",
    [
        ([2, 3], [8, 8], [2, 3]),
        (None, [2, 3], [2, 3]),
        ([0, 0], [8, 8], [0, 0]),
        (None, None, None),
    ],
)
def test_per_pass_metadata_preserves_dispatch_counts(
    global_counts, original, padded, expected
):
    original = None if original is None else list(original)
    padded = None if padded is None else list(padded)
    gatherer = ed._LayerBasedGpuSinglePassGatherer.__new__(
        ed._LayerBasedGpuSinglePassGatherer
    )
    gatherer._expert_location_metadata = SimpleNamespace(
        num_local_physical_experts=2, num_physical_experts=4
    )
    gatherer._rank = 1
    gatherer._enable_global_physical_experts = global_counts
    gatherer._data = torch.tensor(
        [[3, 5, 7, 11]] if global_counts else [[7, 11]], dtype=torch.int32
    )
    before = gatherer.collect()["global_physical_count"].clone()
    batch = SimpleNamespace(
        original_global_num_tokens_cpu=original,
        global_num_tokens_cpu=padded,
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=2,
        input_ids=torch.arange(16),
        is_extend_in_batch=False,
    )
    gatherer.on_forward_pass_start(batch)
    result = gatherer.collect()
    assert torch.equal(result["global_physical_count"], before)
    assert result["forward_metadata"] == {
        "forward_mode": "TARGET_VERIFY",
        "batch_size": 2,
        "num_tokens": 16,
        "is_extend_in_batch": False,
        "global_num_tokens_cpu": expected,
    }
    # Metadata is a snapshot, independent of later scheduler list reuse.
    if original is not None:
        original[:] = [99, 99]
    if padded is not None:
        padded[:] = [99, 99]
    assert gatherer.collect()["forward_metadata"]["global_num_tokens_cpu"] == expected


def test_idle_forward_metadata_handles_absent_input_ids():
    gatherer = ed._LayerBasedGpuSinglePassGatherer.__new__(
        ed._LayerBasedGpuSinglePassGatherer
    )
    batch = SimpleNamespace(
        original_global_num_tokens_cpu=None,
        global_num_tokens_cpu=None,
        forward_mode=ForwardMode.IDLE,
        batch_size=0,
        input_ids=None,
        is_extend_in_batch=True,
    )
    gatherer.on_forward_pass_start(batch)
    assert gatherer._forward_metadata == {
        "forward_mode": "IDLE",
        "batch_size": 0,
        "num_tokens": 0,
        "is_extend_in_batch": True,
        "global_num_tokens_cpu": None,
    }
