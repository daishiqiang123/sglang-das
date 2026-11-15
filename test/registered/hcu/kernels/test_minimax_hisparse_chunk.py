"""Check exact query-subset paging with the real FlashMLA consumer."""

import unittest
import torch

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=45, suite="stage-b-test-1-hcu-small")


class TestMiniMaxHiSparseChunk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.version.hip or not torch.cuda.is_available():
            raise unittest.SkipTest("Requires an HCU device")
        if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx938"):
            raise unittest.SkipTest("Requires gfx938")

    def _check_shape(self, batch, tail):
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_hisparse import (
            flash_mla_chunked_verify,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_hisparse_plan import (
            pack_verify_plan,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
            flash_mla_sparse_decode_main,
            new_flash_mla_decode_metadata,
        )
        from sglang.kernels.ops.kvcache.hisparse import swap_mha_pages_tiled

        torch.manual_seed(43)
        width, ctx, page = 8, 147712, 128
        req = torch.arange(batch, dtype=torch.int64, device="cuda")
        real = torch.tensor([batch], dtype=torch.int32, device="cuda")
        host = torch.arange(page, page + batch * ctx, device="cuda").reshape(batch, ctx)
        ends = (
            147456 + tail + torch.arange(1, 9, device="cuda", dtype=torch.int32)
        ).repeat(batch)
        top = torch.randint(
            0, 1152, (4, batch * 8, 16), device="cuda", dtype=torch.int32
        )
        top[..., 0] = (ends - 1) // 128
        top[..., 3] = -1
        q = torch.randn(batch * 8, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        hk = torch.empty((page + batch * ctx, 4, 128), device="cuda", dtype=q.dtype)
        hv = torch.empty_like(hk)
        for t in [hk, hv]:
            raw = t.view(torch.uint8)
            raw.random_(0, 240)
            raw.add_((raw >= 120).to(torch.uint8) * 8)
        dummy = torch.zeros(batch, 1, device="cuda", dtype=torch.int32)
        variants = []
        for buffer in [65536, 30976]:
            dev = torch.arange(
                page, page + batch * (buffer + page), device="cuda"
            ).reshape(batch, buffer + page)
            k = torch.empty(
                (page + batch * (buffer + page), 4, 128), device="cuda", dtype=q.dtype
            )
            v = torch.empty_like(k)
            tags = torch.full((batch * 2, 512), -1, device="cuda", dtype=torch.int64)
            meta = new_flash_mla_decode_metadata()
            variants.append((buffer, dev, k, v, tags, meta))

        def run(which):
            buffer, dev, k, v, tags, meta = variants[which]
            src = host.gather(1, (ends.reshape(batch, 8) - 1).long()).reshape(-1)
            dst = dev[:, buffer : buffer + 8].flatten()
            for a, b in [(hk, k), (hv, v)]:
                b.view(torch.uint8).index_copy_(
                    0, dst, a.view(torch.uint8).index_select(0, src)
                )

            def swap(topk, lengths):
                p = pack_verify_plan(
                    req,
                    real,
                    topk,
                    lengths,
                    host,
                    dev,
                    tags,
                    buffer,
                    direct_verify_indices=True,
                )
                swap_mha_pages_tiled(
                    p.miss_src, p.miss_dst, p.miss_count, real, hk, hv, k, v, 256, 16
                )
                return p.indices

            kw = dict(
                q=q,
                k_cache=k,
                v_cache=v,
                topk_idx=top,
                page_table=dummy,
                seq_lens=ends,
                block_size_k=128,
                sched_meta=meta,
                verify_group_size=8,
            )
            if which == 0:
                return flash_mla_sparse_decode_main(**kw, hot_indices=swap(top, ends))
            return flash_mla_chunked_verify(
                query_chunk_size=4, hot_indices_fn=swap, **kw
            )

        for _ in range(2):
            a, b = run(0), run(1)
            torch.cuda.synchronize()
            assert torch.equal(a, b), (batch, tail, "eager", float((a - b).abs().max()))
        graphs = []
        for i in range(2):
            with torch.cuda.graph(g := torch.cuda.CUDAGraph()):
                out = run(i)
            graphs.append((g, out))
        for delta in [1, 3, 7]:
            ends.add_(delta)
            top[..., 4:11].add_(17).remainder_(1152)
            top[..., 0] = (ends - 1) // 128
            q.copy_(torch.randn(q.shape, device="cuda").to(q.dtype))
            for g, out in graphs:
                g.replay()
            torch.cuda.synchronize()
            assert torch.equal(graphs[0][1], graphs[1][1]), (
                batch,
                tail,
                "graph",
                delta,
            )

    def test_eager_and_changed_graph_inputs(self):
        for batch in (1, 2, 8):
            for tail in (29, 125, 127):
                with self.subTest(batch=batch, tail=tail):
                    self._check_shape(batch, tail)


if __name__ == "__main__":
    unittest.main()
