from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "triton", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if get_platform().is_hip:
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if get_platform().is_hip:
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse_feature_combos(server_args: ServerArgs) -> None:

    cfg = resolving_view(server_args)
    # A mixed batch cannot merge the HiSparse decode batch.
    if cfg.enable_mixed_chunk:
        raise ValueError("--enable-hisparse does not support --enable-mixed-chunk yet.")


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""

    cfg = resolving_view(server_args)
    if not cfg.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
        is_minimax_sparse,
    )

    hf_config = model_config_of(server_args).hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_m3_hisparse = is_minimax_sparse(hf_config)
    is_hip = get_platform().is_hip
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse or is_m3_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5), DeepSeek V4, and MiniMax M3 now. "
    )

    assert cfg.disable_radix_cache, (
        "Hierarchical sparse attention currently requires --disable-radix-cache."
    )
    validate_hisparse_feature_combos(server_args)

    enable_spec = (
        cfg.speculative_algorithm is not None
        and cfg.speculative_algorithm.upper() != "NONE"
    )
    if enable_spec and (cfg.speculative_eagle_topk or 1) != 1:
        raise ValueError(
            "The minimal HiSparse spec integration supports linear speculation "
            "only; use --speculative-eagle-topk 1."
        )
    if enable_spec and cfg.speculative_adaptive:
        raise ValueError(
            "HiSparse speculative decoding uses statically sized swap state; "
            "--speculative-adaptive is not supported."
        )

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    if is_m3_hisparse:
        from sglang.srt.environ import envs

        if envs.SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938.get():
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            from sglang.srt.utils import is_hcu

            config = parse_hisparse_config(config_json=cfg.hisparse_config or "{}")
            if not (
                is_hcu()
                and cfg.disaggregation_mode == "decode"
                and (
                    cfg.disaggregation_transfer_backend == "mooncake"
                    or (
                        cfg.disaggregation_transfer_backend == "fake"
                        and os.environ.get("M31_HISPARSE_FAKE_VALIDATE") == "1"
                    )
                )
                and cfg.pp_size == 1
                and cfg.attn_cp_size == 1
                and cfg.dcp_size == 1
                and not cfg.enable_hierarchical_cache
                and not cfg.disaggregation_decode_enable_radix_cache
                and cfg.disaggregation_decode_host_receive_threshold == 1
                and cfg.speculative_algorithm == "DSPARK"
                and cfg.speculative_num_draft_tokens == 8
                and read_ragged_verify_mode() is RaggedVerifyMode.STATIC
                and cfg.page_size == 128
            ):
                raise ValueError(
                    "MiniMax FlashMLA HiSparse requires HCU Mooncake PD decode, "
                    "PP/CP/DCP=1, static DSpark verify8, page size 128, "
                    "and no HiCache or decode radix cache."
                )
            size, ratio = config.device_buffer_size, config.host_to_device_ratio
            compact = (
                cfg.disaggregation_transfer_backend == "fake"
                and os.environ.get("M31_HISPARSE_AB_COMPACT_HOT") == "1"
            )
            query_chunk = int(os.environ.get("M31_HISPARSE_VERIFY_QUERY_CHUNK", "8"))
            if query_chunk not in (4, 8) or (query_chunk != 8 and not compact):
                raise ValueError(
                    "Split-query HiSparse requires the explicit fake-PD compact mode"
                )
            min_size = (query_chunk * 4 * 15 + 2) * 128 if compact else 65536
            if (
                type(size) is not int
                or size < min_size
                or size % 128
                or type(ratio) not in (int, float)
                or not math.isfinite(ratio)
                or ratio < 1
            ):
                raise ValueError(
                    "MiniMax FlashMLA HiSparse requires an aligned integer "
                    f"device_buffer_size >= {min_size} and finite host_to_device_ratio >= 1."
                )
        return

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
