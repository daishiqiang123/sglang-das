"""Validate MiniMax HiSparse logical addresses and changed graph inputs on gfx938."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=40, suite="stage-b-test-1-hcu-small")


class TestMiniMaxHiSparsePlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.version.hip or not torch.cuda.is_available():
            raise unittest.SkipTest("Requires an HCU device")
        if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx938"):
            raise unittest.SkipTest("Requires gfx938")

    @patch.dict(os.environ, {"M31_HISPARSE_HIP_STABLE_PLAN": "1"})
    def test_logical_address_oracle_and_graph_replay(self):
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_hisparse_plan import (
            pack_verify_plan,
        )

        torch.manual_seed(42)
        device = "cuda"
        width, heads, top, page = 8, 4, 16, 128
        slots, ctx, buffer_size = 12, 147712, 65536
        batch = 8
        real = torch.tensor([7], dtype=torch.int32, device=device)
        reqs = torch.arange(batch, device=device, dtype=torch.int64) + 1
        reqs[-1] = reqs[0]
        host = torch.arange(slots * ctx, device=device).reshape(slots, ctx) + page
        hot = (
            torch.arange(slots * (buffer_size + page), device=device).reshape(
                slots, buffer_size + page
            )
            + page
        )
        topk = torch.randint(
            0, 1151, (batch * width, heads, top), dtype=torch.int32, device=device
        )
        topk = topk.permute(1, 0, 2)
        ends = (
            (
                torch.full((batch, 1), 147452, device=device)
                + torch.arange(1, width + 1, device=device)
            )
            .flatten()
            .to(torch.int32)
        )
        topk[..., 0] = (ends - 1) // page
        topk[..., 1] = topk[..., 2]
        topk[..., 4] = -1
        base_tags = torch.full(
            (slots * 2, width * heads * top), -1, device=device, dtype=torch.long
        )
        stable_tags = base_tags.clone()
        page_slots = torch.full(
            (slots, ctx // page), -1, dtype=torch.int32, device=device
        )

        def baseline():
            return pack_verify_plan(
                reqs, real, topk, ends, host, hot, base_tags, buffer_size
            )

        def candidate():
            return pack_verify_plan(
                reqs, real, topk, ends, host, hot, stable_tags, buffer_size, page_slots
            )

        def decode_address(plan, tags):
            output = []
            for i in range(batch):
                idx = plan.indices[i].reshape(width, heads, top, page)
                resident = tags[reqs[i], :512]
                physical = hot[reqs[i], :buffer_size:page] // page
                lut = torch.full(
                    (slots * (buffer_size + page) // page + 1,),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                lut[physical] = resident
                for j in range(int(plan.miss_count[i])):
                    hp = int(plan.miss_src[i, j])
                    dp = int(plan.miss_dst[i, j])
                    lut[dp] = hp - (reqs[i] * ctx // page + 1)
                for j in range(int(plan.verify_count[i])):
                    dp = int(plan.verify_dst[i, j] // page)
                    scratch = int(plan.verify_src[i, j] - hot[reqs[i], buffer_size])
                    lut[dp] = (
                        int(ends[i * width + width - 1]) - width + scratch
                    ) // page
                output.append(
                    torch.where(
                        idx >= 0, lut[idx.clamp_min(0) // page] * page + idx % page, -1
                    )
                )
            return torch.stack(output)

        def check(a, b):
            expected = topk.reshape(heads, batch, width, top).permute(1, 2, 0, 3)
            expected = expected[..., None] * page + torch.arange(page, device=device)
            expected = torch.where(
                (expected >= 0)
                & (expected < ends.reshape(batch, width, 1, 1, 1))
                & (torch.arange(batch, device=device) < real[0]).reshape(
                    batch, 1, 1, 1, 1
                ),
                expected,
                -1,
            )
            assert torch.equal(decode_address(a, base_tags), expected)
            assert torch.equal(decode_address(b, stable_tags), expected)
            assert torch.equal(a.verify_count, b.verify_count)
            assert not a.miss_count[-1] and not b.miss_count[-1]
            for i in range(7):
                for p in (a, b):
                    assert (
                        p.miss_dst[i, : p.miss_count[i]].unique().numel()
                        == p.miss_count[i]
                    )
            return int(a.miss_count.sum()), int(b.miss_count.sum())

        check(baseline(), candidate())
        with torch.cuda.graph(g0 := torch.cuda.CUDAGraph()):
            a = baseline()
        with torch.cuda.graph(g1 := torch.cuda.CUDAGraph()):
            b = candidate()
        base_tags.fill_(-1)
        stable_tags.fill_(-1)
        page_slots.fill_(-1)
        g0.replay()
        g1.replay()
        print("cold", check(a, b), flush=True)
        for shift in (1, 3, 5):
            topk[..., 7:10].add_(shift).remainder_(1151)
            ends.add_(1)
            topk[..., 0] = (ends - 1) // page
            g0.replay()
            g1.replay()
            print("changed", shift, check(a, b), flush=True)


if __name__ == "__main__":
    unittest.main()
