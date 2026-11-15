import unittest

import torch

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")

from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (
    sparse_qk_index_gemma_rmsnorm_rope_cache as fn,
)


class TestMiniMaxMappedCache(unittest.TestCase):
    @unittest.skipUnless(
        torch.version.hip and torch.cuda.is_available(), "HCU required"
    )
    def test_mapping_and_changed_graph_inputs(self):
        torch.manual_seed(42)
        checks = [0]
        for n in (8, 64, 160):
            for neox in (False, True):
                inputs = [
                    torch.randn(n, h * 128, device="cuda", dtype=torch.bfloat16)
                    for h in (64, 4, 4, 4, 1)
                ]
                weights = [
                    torch.randn(128, device="cuda", dtype=torch.bfloat16)
                    for _ in range(4)
                ]
                pos = torch.arange(n, device="cuda", dtype=torch.int64)
                cos = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16)
                loc = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)
                loc[-1] = 0
                logical = 512
                physical = 256
                mapping = torch.zeros(logical, device="cuda", dtype=torch.int64)
                mapping[1 : n + 1] = torch.randperm(physical - 1, device="cuda")[:n] + 1

                def caches(slots):
                    return [
                        torch.zeros(
                            slots, h, 128, device="cuda", dtype=torch.float8_e4m3fn
                        )
                        for h in (4, 4, 1)
                    ]

                base = caches(logical)
                cand = caches(physical)
                cand[2] = torch.zeros_like(base[2])

                def call(cs, mp):
                    return fn(
                        *inputs,
                        *cs,
                        loc,
                        *weights,
                        pos,
                        cos,
                        1e-6,
                        128,
                        64,
                        neox,
                        fp8_cache=True,
                        main_cache_mapping=mp,
                    )

                def check(a, b):
                    for i in (0, 2):
                        assert torch.equal(
                            a[i].view(torch.uint8), b[i].view(torch.uint8)
                        )
                    ids = loc[:-1]
                    dest = mapping[ids]
                    for i in (0, 1):
                        assert torch.equal(
                            base[i].view(torch.uint8)[ids],
                            cand[i].view(torch.uint8)[dest],
                        )
                        untouched = torch.ones(
                            physical, dtype=torch.bool, device="cuda"
                        )
                        untouched[dest] = False
                        assert not cand[i].view(torch.uint8)[untouched].any()
                    assert torch.equal(
                        base[2].view(torch.uint8), cand[2].view(torch.uint8)
                    )
                    checks[0] += 1

                check(call(base, None), call(cand, mapping))
                with torch.cuda.graph(g0 := torch.cuda.CUDAGraph()):
                    a = call(base, None)
                with torch.cuda.graph(g1 := torch.cuda.CUDAGraph()):
                    b = call(cand, mapping)
                for shift in (0, 1, 3):
                    for x in base + cand:
                        x.zero_()
                    mapping[1 : n + 1] = (
                        torch.randperm(physical - 1, device="cuda")[:n] + 1
                    )
                    for x in inputs:
                        x.add_(0.25)
                    pos.add_(1)
                    g0.replay()
                    g1.replay()
                    check(a, b)
        self.assertEqual(checks[0], 24)


if __name__ == "__main__":
    unittest.main()
