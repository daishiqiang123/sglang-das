"""MiniMax M3.1's checkpoint-local DSpark draft.

The bundled export wraps each decoder in ``decoder_layer`` and uses Gemma
normalization and SwiGLU-OAI.  It must not inherit the Qwen draft's RMS/SiLU
defaults, even though the DSpark worker and context-injection protocol agree.
"""

import logging
from copy import copy
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding.base import _launch_rotary_embedding
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.models.dspark import DSparkConfidenceHead, DSparkDraftModel
from sglang.srt.models.minimax_m3 import MiniMaxM3MLP
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hcu

logger = logging.getLogger(__name__)


def normalize_minimax_dspark_weight_name(name: str) -> str:
    prefix = "language_model.model.dspark."
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected MiniMax DSpark checkpoint weight: {name!r}")
    name = name[len(prefix) :].replace(".decoder_layer.", ".")
    if name.startswith("final_norm."):
        name = "norm." + name[len("final_norm.") :]
    return name


class MiniMaxDSparkRotaryEmbedding(nn.Module):
    """Keep the shared FP32 QSA table while honoring the HCU kernel's dtype."""

    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary
        self._logged_q_heads = set()

    @property
    def cos_sin_cache(self):
        return self.rotary.cos_sin_cache

    @property
    def rotary_dim(self):
        return self.rotary.rotary_dim

    @property
    def head_size(self):
        return self.rotary.head_size

    @property
    def is_neox_style(self):
        return self.rotary.is_neox_style

    def forward(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
        if not is_hcu():
            return self.rotary(positions, query, key, offsets, fused_set_kv_buffer_arg)
        if fused_set_kv_buffer_arg is not None:
            raise ValueError("MiniMax DSpark uses a separate Gemma KV-cache write.")
        if offsets is not None:
            positions = positions + offsets
        # get_rope() may return the target's cached module. Never replace its
        # FP32 cache or forward method: the target QSA kernel requires FP32.
        cache = self.rotary.cos_sin_cache_as(query.dtype).to(device=query.device)
        if positions.numel():
            q_heads = query.shape[-1] // self.rotary.head_size
            if q_heads not in self._logged_q_heads:
                logger.debug(
                    "MiniMax DSpark HCU RoPE: q_dtype=%s k_dtype=%s "
                    "source_cache=%s kernel_cache=%s q_heads=%d q_stride=%s "
                    "k_stride=%s rotary_dim=%d.",
                    query.dtype,
                    key.dtype,
                    self.rotary.cos_sin_cache.dtype,
                    cache.dtype,
                    q_heads,
                    query.stride(),
                    key.stride(),
                    self.rotary.rotary_dim,
                )
                self._logged_q_heads.add(q_heads)
        _launch_rotary_embedding(
            self.rotary.fallback_rotary_embedding,
            positions,
            query,
            key,
            self.rotary.head_size,
            cache,
            self.rotary.is_neox_style,
        )
        return query, key


class MiniMaxDSparkAttention(DFlashAttention):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__(config, layer_id, quant_config, prefix)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # The generic table fusion multiplies by w; Gemma multiplies by 1+w.
        self.use_table_qk_norm_rope = False
        # self.rotary_emb = get_rope(...) from DFlash uses the generic HIP path.
        self.rotary_emb = MiniMaxDSparkRotaryEmbedding(self.rotary_emb)

    def kv_proj_only(self, hidden_states: torch.Tensor):
        """Skip unused Q outputs when materializing channel-INT8 context KV."""
        projection = self.qkv_proj
        weight = getattr(projection, "weight", None)
        if (
            weight is None
            or weight.dtype != torch.int8
            or not is_hcu()
            or not envs.SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY.get()
        ):
            return super().kv_proj_only(hidden_states)

        from compressed_tensors.quantization import QuantizationStrategy

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
            CompressedTensorsW8A8Int8,
        )

        scheme = getattr(projection, "scheme", None)
        output_size = self.q_size + 2 * self.kv_size
        scales = getattr(projection, "weight_scale", None)
        # Method 3 uses contiguous [N, K] on gfx936/gfx938. Other layouts,
        # tensor scales, static inputs and other quantization schemes retain
        # the generic path.
        if not (
            type(scheme) is CompressedTensorsW8A8Int8
            and scheme.w8a8_strategy == 3
            and scheme.strategy == QuantizationStrategy.CHANNEL
            and not scheme.is_static_input_scheme
            and scheme.input_symmetric
            and tuple(weight.shape) == (output_size, self.hidden_size)
            and weight.is_contiguous()
            and scales is not None
            and tuple(scales.shape) == (output_size, 1)
            and scales.dtype == torch.float32
        ):
            return super().kv_proj_only(hidden_states)

        # These are views of the current post-load parameters: no duplicate
        # weights and no cached pointers that could outlive a weight update.
        kv_rows = slice(self.q_size, output_size)
        kv_projection = SimpleNamespace(
            weight=weight[kv_rows], weight_scale=scales[kv_rows]
        )
        bias = getattr(projection, "bias", None)
        kv = scheme.apply_weights(
            kv_projection,
            hidden_states,
            bias=None if bias is None else bias[kv_rows],
        )
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        if self.v_scale is not None:
            v = v * self.v_scale
        return k, v


class MiniMaxDSparkDecoderLayer(DFlashDecoderLayer):
    """Reuse the residual protocol while constructing MiniMax submodules."""

    def __init__(
        self,
        config,
        layer_id,
        attention_conv=None,
        mlp_conv=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        if attention_conv is not None or mlp_conv is not None:
            raise ValueError("MiniMax DSpark does not export grouped convolutions.")
        prefix = f"{prefix}.decoder_layer" if prefix else "decoder_layer"
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = MiniMaxDSparkAttention(
            config, layer_id, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = MiniMaxM3MLP(
            config,
            quant_config,
            f"{prefix}.mlp",
            intermediate_size=config.intermediate_size,
        )
        self.attention_conv = None
        self.mlp_conv = None


class DSparkMiniMaxDraftModel(DSparkDraftModel):
    decoder_layer_cls = MiniMaxDSparkDecoderLayer
    supports_quantization = True

    def __init__(self, config, quant_config=None, prefix=""):
        # The export retains a VLM envelope; model construction receives that
        # envelope, while the trained draft network lives in text_config.
        config = getattr(config, "text_config", config)
        config = (
            PretrainedConfig.from_dict(config) if isinstance(config, dict) else copy(config)
        )
        if not getattr(config, "use_gemma_norm", False):
            raise ValueError("MiniMax DSpark requires use_gemma_norm=true.")
        if config.hidden_act != "swigluoai":
            raise ValueError("MiniMax DSpark requires the trained SwiGLU-OAI activation.")
        # Parsing understands dspark_* names, whereas the head constructors
        # consume these canonical names (the export stores them on the envelope).
        config.markov_rank = int(config.dspark_markov_rank)
        config.markov_head_type = config.dspark_markov_head_type
        config.intermediate_size = int(config.dspark_ffn_hidden_size)
        # Match the original ignore list, notably the unquantized fc projection.
        super().__init__(
            config,
            quant_config,
            prefix="language_model.model.dspark",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.confidence_head is None and config.enable_confidence_head:
            # The shared constructor omits this module for static verification.
            # Keep the trained MiniMax head so every exported tensor is loaded
            # and audited even when the planner uses a fixed verification width.
            self.confidence_head = DSparkConfidenceHead(
                hidden_size=config.hidden_size,
                markov_rank=config.markov_rank,
                with_markov=config.confidence_head_with_markov,
            )
        parallel = get_parallel()
        attention = self.layers[0].self_attn
        logger.info(
            "MiniMax DSpark placement: draft_tp=%d, draft_tp_rank=%d, "
            "draft_attention_dp=%d, local_q_heads=%d, local_kv_heads=%d, "
            "target_layer_ids=%s, context_features=%d.",
            parallel.tp_size,
            parallel.tp_rank,
            parallel.attn_dp_size,
            attention.num_heads,
            attention.num_kv_heads,
            config.dspark_target_layer_ids,
            self.num_context_features,
        )

    def _build_fused_kv_write_bundle(self, pool):
        # Generic context fusions implement RMS rather than Gemma normalization.
        return None

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        super().attach_shared_modules(embed_tokens=embed_tokens, lm_head=lm_head)
        logger.info(
            "MiniMax DSpark shared target modules: embedding=%s lm_head=%s "
            "org_vocab_size=%d.",
            tuple(embed_tokens.weight.shape), tuple(lm_head.weight.shape),
            lm_head.org_vocab_size,
        )

    def _stacked_ctx_kv_params(self):
        return None

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        normalized = []
        covered = set()
        mappings = (
            ("q_proj", "qkv_proj"),
            ("k_proj", "qkv_proj"),
            ("v_proj", "qkv_proj"),
            ("gate_proj", "gate_up_proj"),
            ("up_proj", "gate_up_proj"),
        )
        for name, tensor in weights:
            name = normalize_minimax_dspark_weight_name(name)
            param_name = name
            for source, target in mappings:
                if f".{source}." in name:
                    param_name = name.replace(f".{source}.", f".{target}.")
                    break
            if param_name not in params:
                raise ValueError(f"MiniMax DSpark weight has no parameter: {name!r}")
            covered.add(param_name)
            normalized.append((name, tensor))
        missing = set(params) - covered
        if missing:
            raise ValueError(f"MiniMax DSpark missing checkpoint parameters: {sorted(missing)}")
        super().load_weights(normalized)
        logger.info(
            "MiniMax DSpark loaded %d checkpoint tensors into %d parameters; "
            "fc=%s, qkv=%s, mlp=%s, norm=%s.",
            len(normalized),
            len(covered),
            type(self.fc.quant_method).__name__,
            type(self.layers[0].self_attn.qkv_proj.quant_method).__name__,
            type(self.layers[0].mlp.gate_up_proj.quant_method).__name__,
            type(self.norm).__name__,
        )
        return covered


EntryClass = [DSparkMiniMaxDraftModel]
