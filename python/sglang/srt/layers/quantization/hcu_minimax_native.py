"""HCU weight-only execution of MiniMax MXFP8/NVFP4 source checkpoints.

Opt-in compatibility path: retain packed weights, decode to BF16 before MMAC.
Activations remain BF16; this does not emulate NVIDIA's FP4 activation recipe.
"""

import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)
from sglang.srt.utils import get_bool_env_var, set_weight_attrs

logger = logging.getLogger(__name__)


@triton.jit
def _dequant_mxfp8_kernel(
    W,
    S,
    O,
    N: tl.constexpr,
    K: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // K, i % K
    valid = row < N
    # gfx938 Triton cannot load/cast fp8e4nv directly. Decode unchanged bytes.
    # w = tl.load(W + row * SW0 + col * SW1, valid, other=0).to(tl.float32)
    raw = tl.load(W + row * SW0 + col * SW1, valid, other=0).to(tl.int32)
    exp, mant = (raw >> 3) & 15, raw & 7
    normal = (((exp + 120) << 23) | (mant << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exp == 0, mant.to(tl.float32) * 0.001953125, normal)
    w = tl.where((raw & 128) != 0, -magnitude, magnitude)
    w = tl.where((exp == 15) & (mant == 7), float("nan"), w)
    exponent = tl.load(S + row * SS0 + col // 32 * SS1, valid, other=127)
    # UE8M0 code 0 is 2**-127, rather than IEEE float32 zero.
    bits = tl.where(exponent == 0, 0x00400000, exponent.to(tl.int32) << 23)
    scale = bits.to(tl.float32, bitcast=True)
    tl.store(O + i, w * scale, valid)


def dequant_mxfp8_weight(weight, scales, dtype=torch.bfloat16):
    n, k = weight.shape
    if k % 32 or scales.shape != (n, k // 32):
        raise ValueError(f"Invalid MXFP8 shapes: {weight.shape}, {scales.shape}")
    if scales.dtype != torch.uint8:
        raise TypeError("MXFP8 source scales must be raw UE8M0 bytes")
    out = torch.empty((n, k), device=weight.device, dtype=dtype)
    _dequant_mxfp8_kernel[(triton.cdiv(n * k, 1024),)](
        weight.view(torch.uint8),
        scales,
        out,
        n,
        k,
        *weight.stride(),
        *scales.stride(),
        BLOCK=1024,
    )
    return out


class HcuMxfp8LinearMethod(LinearMethodBase):
    # Skip MXFP8-specific projection/activation fusions. Those require native
    # scaled-matmul kernels, whereas this path consumes BF16 activations.
    use_mxfp8 = False

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        Fp8LinearMethod.create_fp8_weight_(
            layer,
            block_quant=True,
            quant_config=self.quant_config,
            use_mxfp8=True,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            is_checkpoint_fp8_serialized=True,
            skip_block_quant_check=extra_weight_attrs.get(
                "skip_block_quant_check", False
            ),
        )

    def process_weights_after_loading(self, layer):
        if torch.any(layer.weight_scale_inv == 255).item():
            raise ValueError("Reserved UE8M0 scale code 255 in source checkpoint")

    def apply(self, layer, x, bias=None):
        weight = dequant_mxfp8_weight(layer.weight, layer.weight_scale_inv, x.dtype)
        return F.linear(x, weight, bias)


class HcuNvfp4MoEMethod(FusedMoEMethodBase):
    """Packed E2M1 + group-16 E4M3 scales, decoded inside Triton GEMM."""

    def __init__(self):
        super().__init__()
        self._loaded = set()
        self.compact_scales = get_bool_env_var("SGLANG_HCU_M3_COMPACT_SCALES", "false")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        with_bias=False,
        **extra_weight_attrs,
    ):
        if with_bias or hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError("HCU native NVFP4 requires bias-free, group-16 shards")
        h, n, e = hidden_size, intermediate_size_per_partition, layer.num_local_experts
        self.num_local_experts = e
        self.intermediate_size = n
        original_loader = extra_weight_attrs["weight_loader"]

        def checked_loader(param, loaded_weight, *args, **kwargs):
            original_loader(param, loaded_weight, *args, **kwargs)
            expert = kwargs.get("expert_id")
            shard = kwargs.get("shard_id")
            if expert is not None:
                local = layer._map_global_expert_id_to_local_expert_id(expert)
                if 0 <= local < e:
                    self._loaded.add((param._hcu_native_name, local, shard))

        def register(name, shape, dtype, quant_method=None):
            fill = 0 if dtype == torch.uint8 else float("nan")
            if dtype == torch.float8_e4m3fn:
                data = torch.empty(shape, dtype=dtype)
                data.view(torch.uint8).fill_(127)  # E4M3 NaN sentinel.
            else:
                data = torch.full(shape, fill, dtype=dtype)
            param = torch.nn.Parameter(data, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(
                param, {**extra_weight_attrs, "weight_loader": checked_loader}
            )
            param._hcu_native_name = name
            if quant_method is not None:
                param.quant_method = quant_method

        register("w13_weight", (e, 2 * n, h // 2), torch.uint8)
        register("w2_weight", (e, h, n // 2), torch.uint8)
        scale_dtype = torch.float8_e4m3fn if self.compact_scales else torch.float32
        register("w13_weight_scale", (e, 2 * n, h // 16), scale_dtype, "group")
        register("w2_weight_scale", (e, h, n // 16), scale_dtype, "group")
        register("w13_weight_scale_2", (e, 2), torch.float32, "tensor")
        register("w2_weight_scale_2", (e,), torch.float32, "tensor")

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        if not get_moe_runner_backend().is_triton():
            raise ValueError(
                "HCU source NVFP4 compatibility path requires --moe-runner-backend triton"
            )
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def process_weights_after_loading(self, layer):
        # Nine source tensors per expert: three packed matrices, three group
        # scales and three distinct global scales (gate and up may differ).
        if len(self._loaded) != self.num_local_experts * 9:
            raise ValueError(
                f"Incomplete native NVFP4 load: {len(self._loaded)} / {self.num_local_experts * 9}"
            )
        for global_scale in (layer.w13_weight_scale_2, layer.w2_weight_scale_2):
            if not torch.all(torch.isfinite(global_scale) & (global_scale > 0)).item():
                raise ValueError("Invalid native NVFP4 weight_global_scale")
        if self.compact_scales:
            for scales in (layer.w13_weight_scale, layer.w2_weight_scale):
                if not torch.all(torch.isfinite(scales.float())).item():
                    raise ValueError("Non-finite native NVFP4 source group scales")
            return
        n = self.intermediate_size
        # Source convention is E2M1 * E4M3_scale / weight_global_scale.
        # Folding into FP32 scales changes no packed codes or group granularity.
        layer.w13_weight_scale.data[:, :n].div_(
            layer.w13_weight_scale_2[:, 0, None, None]
        )
        layer.w13_weight_scale.data[:, n:].div_(
            layer.w13_weight_scale_2[:, 1, None, None]
        )
        layer.w2_weight_scale.data.div_(layer.w2_weight_scale_2[:, None, None])
        for scales in (layer.w13_weight_scale, layer.w2_weight_scale):
            if not torch.all(torch.isfinite(scales)).item():
                raise ValueError("Non-finite native NVFP4 group scales")

    def get_triton_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        # The shared E2M1 W4A16 kernel accepts either MXFP4 UE8M0/g32 or
        # native NVFP4's exact FP32 group scales/g16 on the dedicated branch.
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_scale=(
                layer.w13_weight_scale.view(torch.uint8)
                if self.compact_scales
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale.view(torch.uint8)
                if self.compact_scales
                else layer.w2_weight_scale
            ),
            w13_global_scale=layer.w13_weight_scale_2 if self.compact_scales else None,
            w2_global_scale=layer.w2_weight_scale_2 if self.compact_scales else None,
            use_mxfp4_w4a16=True,
            block_shape=[0, 16],
        )

    def apply(self, layer, dispatch_output):
        return self.runner.run(dispatch_output, self.get_triton_quant_info(layer))


def remap_native_minimax_weights(weights):
    """Translate source names without applying the ordinary MXFP8 scale rename."""
    for name, weight in weights:
        if ".block_sparse_moe.experts." in name:
            if name.endswith(".input_global_scale"):
                # Weight-only BF16-activation execution does not consume A4 scales.
                continue
            if name.endswith(".weight_packed"):
                name = name.removesuffix("_packed")
            elif name.endswith(".weight_global_scale"):
                name = name.removesuffix(".weight_global_scale") + ".weight_scale_2"
        elif name.endswith(".weight_scale"):
            name += "_inv"
        yield name, weight
