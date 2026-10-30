# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for MiniMax pipeline-stage weight ownership."""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import get_layer_id
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MODEL_ROOT = Path(__file__).parents[4] / "python/sglang/srt/models"
LOGGER = logging.getLogger("minimax_pp_weight_loading_test")


def _load_function(filename, name, namespace):
    # Exercise the real loader without importing model kernels or constructing GPUs.
    tree = ast.parse((MODEL_ROOT / filename).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, function], type_ignores=[])
    )
    exec(compile(module, str(MODEL_ROOT / filename), "exec"), namespace)
    return namespace[name]


def _copy_weight(param, weight):
    with torch.no_grad():
        param.copy_(weight)


class TestMiniMaxM3PipelineWeightLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "nn": nn,
            "PPMissingLayer": PPMissingLayer,
            "FusedMoE": SimpleNamespace(make_expert_params_mapping=lambda **_: []),
            "get_layer_id": get_layer_id,
            "get_spec_layer_idx_from_weight_name": lambda *_: None,
            "maybe_remap_kv_scale_name": lambda name, _: name,
            "default_weight_loader": _copy_weight,
            "build_minimax_fused_qkv_index": lambda _: None,
            "logger": LOGGER,
        }
        cls.is_missing = staticmethod(
            _load_function(
                "minimax_m3.py", "_is_minimax_pp_missing_parameter", namespace
            )
        )
        cls.load_causal = staticmethod(
            _load_function("minimax_m3.py", "load_weights", namespace)
        )
        cls.load_vl = staticmethod(
            _load_function("minimax_m3_vl.py", "_load_llm_weight", namespace)
        )

    @staticmethod
    def _model(first_stage):
        model = nn.Module()
        model.model = nn.Module()
        model.model.embed_tokens = (
            nn.Embedding(2, 2) if first_stage else PPMissingLayer()
        )
        model.lm_head = PPMissingLayer() if first_stage else nn.Linear(2, 2, bias=False)
        model.model.layers = nn.ModuleList(
            [PPMissingLayer(), nn.Linear(2, 2, bias=False)]
        )
        model.model.start_layer = 1
        model.model.end_layer = 2
        model.config = SimpleNamespace(
            num_local_experts=0, text_config=SimpleNamespace(num_local_experts=0)
        )
        model.num_fused_shared_experts = 0
        return model

    def test_only_missing_pipeline_modules_are_skipped(self):
        model = self._model(first_stage=True)
        for name in ["lm_head.weight", "model.layers.0.self_attn.q_proj.weight"]:
            with self.subTest(name=name):
                self.assertTrue(self.is_missing(name, model))
        for name in [
            "model.embed_tokens.weight",
            "model.layers.1.weight",
            "model.unknown.weight",
            "model.layers.1.unknown.weight",
        ]:
            with self.subTest(name=name):
                self.assertFalse(self.is_missing(name, model))

    def test_causal_loader_skips_other_stages_and_loads_owned_weights(self):
        for first_stage in [True, False]:
            with self.subTest(first_stage=first_stage):
                model = self._model(first_stage)
                owned = "model.embed_tokens.weight" if first_stage else "lm_head.weight"
                missing = (
                    "lm_head.weight" if first_stage else "model.embed_tokens.weight"
                )
                weight = torch.full((2, 2), 7.0)
                with self.assertNoLogs(LOGGER, level="WARNING"):
                    loaded = self.load_causal(
                        model, [(missing, weight), (owned, weight)]
                    )
                self.assertEqual(loaded, {owned})
                torch.testing.assert_close(
                    dict(model.named_parameters())[owned], weight
                )

    def test_vl_loader_skips_other_stages_and_loads_owned_weights(self):
        for first_stage in [True, False]:
            with self.subTest(first_stage=first_stage):
                model = self._model(first_stage)
                owned = "model.embed_tokens.weight" if first_stage else "lm_head.weight"
                missing = (
                    "lm_head.weight" if first_stage else "model.embed_tokens.weight"
                )
                weight = torch.full((2, 2), 11.0)
                params = dict(model.named_parameters())
                with self.assertNoLogs(LOGGER, level="WARNING"):
                    self.load_vl(model, missing, weight, params, [], [])
                    self.load_vl(model, owned, weight, params, [], [])
                torch.testing.assert_close(params[owned], weight)

    def test_unknown_owned_parameter_still_warns(self):
        model = self._model(first_stage=True)
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            self.load_causal(model, [("model.unknown.weight", torch.ones(2, 2))])
        self.assertTrue(any("model.unknown.weight" in msg for msg in captured.output))


if __name__ == "__main__":
    unittest.main()
