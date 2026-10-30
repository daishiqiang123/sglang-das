# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")


REPO_ROOT = Path(__file__).parents[4]
MOE_SOURCE = (
    REPO_ROOT / "python/sglang/srt/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_w8a8_fp8_moe.py"
)


def _load_helper():
    tree = ast.parse(MOE_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_should_use_hcu_fp8_w8a8_moe"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Optional": Optional}
    exec(compile(module, str(MOE_SOURCE), "exec"), namespace)
    return namespace["_should_use_hcu_fp8_w8a8_moe"]


class TestHcuSpeculativeW8A8MoeBackend(unittest.TestCase):
    def test_eagle_draft_keeps_hcu_fast_path(self):
        helper = _load_helper()
        self.assertTrue(
            helper(
                enabled=True,
                in_speculative_a2a_scope=True,
                speculative_algorithm="EAGLE",
                use_deepep=False,
            )
        )

    def test_standalone_dspark_draft_keeps_canonical_weights(self):
        helper = _load_helper()
        self.assertFalse(
            helper(
                enabled=True,
                in_speculative_a2a_scope=True,
                speculative_algorithm="DSPARK",
                use_deepep=False,
            )
        )

    def test_deepep_dspark_and_target_keep_hcu_fast_path(self):
        helper = _load_helper()
        for in_speculative_scope, use_deepep in ((True, True), (False, False)):
            with self.subTest(
                in_speculative_scope=in_speculative_scope,
                use_deepep=use_deepep,
            ):
                self.assertTrue(
                    helper(
                        enabled=True,
                        in_speculative_a2a_scope=in_speculative_scope,
                        speculative_algorithm="DSPARK",
                        use_deepep=use_deepep,
                    )
                )

    def test_explicit_aiter_keeps_its_weight_layout(self):
        helper = _load_helper()
        for use_deepep in (False, True):
            for in_scope in (False, True):
                for algorithm in (None, "DSPARK", "EAGLE"):
                    with self.subTest(
                        deepep=use_deepep, scope=in_scope, algo=algorithm
                    ):
                        self.assertFalse(
                            helper(
                                enabled=True,
                                in_speculative_a2a_scope=in_scope,
                                speculative_algorithm=algorithm,
                                use_deepep=use_deepep,
                                explicit_aiter_backend=True,
                            )
                        )

    def test_legacy_deepgemm_pack_excludes_explicit_aiter(self):
        tree = ast.parse(MOE_SOURCE.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "CompressedTensorsW8A8Fp8MoE"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "process_weights_after_loading"
        )
        branch = next(
            n
            for n in ast.walk(method)
            if isinstance(n, ast.If)
            and any(
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "_prepare_dsv4_channel_fp8_deepgemm_weights"
                for stmt in n.body
            )
        )
        condition = compile(ast.Expression(branch.test), str(MOE_SOURCE), "eval")
        namespace = {
            "self": SimpleNamespace(
                weight_quant=SimpleNamespace(strategy="channel"), use_deepep=True
            ),
            "QuantizationStrategy": SimpleNamespace(CHANNEL="channel"),
            "_use_deepgemm_moe": True,
            "_is_hcu": True,
        }
        for backend, expected in (
            ("aiter", False),
            ("deep_gemm", False),
            ("auto", True),
            ("triton", True),
        ):
            with self.subTest(backend=backend):
                namespace["get_moe_runner_backend"] = lambda b=backend: SimpleNamespace(
                    is_deep_gemm=lambda: b == "deep_gemm",
                    is_aiter=lambda: b == "aiter",
                )
                self.assertEqual(eval(condition, namespace), expected)

    def test_disabled_hcu_backend_stays_disabled(self):
        helper = _load_helper()
        self.assertFalse(
            helper(
                enabled=False,
                in_speculative_a2a_scope=True,
                speculative_algorithm="EAGLE",
                use_deepep=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
