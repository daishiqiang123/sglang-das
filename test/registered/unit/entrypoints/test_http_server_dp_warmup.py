import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.api_contract import generate_contract_error
from sglang.srt.entrypoints.http_server import _send_dp_attention_warmup_requests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDPAttentionServerWarmup(unittest.TestCase):
    def test_concurrent_scalar_requests_obey_generate_schema(self):
        rank_count = 4
        barrier = threading.Barrier(rank_count, timeout=10)
        cases = [
            ("text", [f"warmup {rank}" for rank in range(rank_count)]),
            ("input_ids", [[10 + rank, 20] for rank in range(rank_count)]),
            ("input_embeds", [[[float(rank), 1.0]] for rank in range(rank_count)]),
        ]
        for key, inputs in cases:
            with self.subTest(input_key=key):
                body = {
                    key: inputs,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                }
                original = copy.deepcopy(body)
                seen = {}

                def post(url, *, json, **kwargs):
                    self.assertIsNone(generate_contract_error(json))
                    rank = json["routed_dp_rank"]
                    self.assertIs(type(rank), int)
                    self.assertEqual(json[key], inputs[rank])
                    self.assertEqual(url, "http://localhost/generate")
                    self.assertEqual(kwargs, {"timeout": 10, "verify": False})
                    seen[rank] = json
                    barrier.wait()
                    return SimpleNamespace(status_code=200, text="ok")

                with patch(
                    "sglang.srt.entrypoints.http_server.requests.post", side_effect=post
                ):
                    responses = _send_dp_attention_warmup_requests(
                        "http://localhost/generate",
                        body,
                        rank_count,
                        timeout=10,
                        verify=False,
                    )
                self.assertEqual(len(responses), rank_count)
                self.assertEqual(set(seen), set(range(rank_count)))
                self.assertEqual(body, original)

    def test_flat_input_ids_are_replicated(self):
        body = {"input_ids": [10, 11, 12]}
        with patch(
            "sglang.srt.entrypoints.http_server.requests.post",
            return_value=SimpleNamespace(status_code=200, text="ok"),
        ) as post:
            _send_dp_attention_warmup_requests("http://localhost/generate", body, 2)
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["json"]["input_ids"], body["input_ids"])

    def test_rank_failure_is_reported(self):
        with patch(
            "sglang.srt.entrypoints.http_server.requests.post",
            return_value=SimpleNamespace(status_code=400, text="rejected"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rank 0: 400 rejected"):
                _send_dp_attention_warmup_requests(
                    "http://localhost/generate", {"text": "warmup"}, 1
                )

    def test_mismatched_input_batch_is_rejected(self):
        with patch("sglang.srt.entrypoints.http_server.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "batch must match"):
                _send_dp_attention_warmup_requests(
                    "http://localhost/generate", {"text": ["a", "b"]}, 3
                )
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
