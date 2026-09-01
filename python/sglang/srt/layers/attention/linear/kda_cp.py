"""Kimi K3 zigzag prefill context parallel helpers.

The KDA recurrence is affine in its incoming recurrent state.  Each local
zigzag segment is summarized as ``state_out = C + state_in @ M``.  CP ranks
all-gather only ``[C | M]``, compose the transforms in natural token order,
and then execute their local segments with the correct incoming states.

This implementation intentionally reuses SGLang's existing Triton/boltops FLA
building blocks.  It is the HCU bring-up counterpart of the NPU affine-state
implementation in sgl-project/sglang#35226; no new compiled operator is needed
for the first performance measurement.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

_CHUNK_SIZE = 64


def get_parallel():
    """Lazy runtime lookup so metadata-only tests never initialize HCU kernels."""
    from sglang.srt.runtime_context import get_parallel as runtime_get_parallel

    return runtime_get_parallel()


def is_hcu() -> bool:
    from sglang.srt.utils import is_hcu as runtime_is_hcu

    return runtime_is_hcu()


def is_cp_v2_active(forward_batch: Any) -> bool:
    from sglang.srt.layers.cp.utils import is_cp_v2_active as runtime_cp_v2_active

    return runtime_cp_v2_active(forward_batch)


def _use_kda_hcu_op() -> bool:
    from sglang.srt.utils import get_bool_env_var

    return is_hcu() and get_bool_env_var("SGLANG_KDA_USE_HCU_OP")


def kda_use_prefill_cp(forward_batch: Any) -> bool:
    """Whether this forward must use the KDA affine-state PCP path."""
    parallel = get_parallel()
    mode = forward_batch.forward_mode
    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    split_list = getattr(metadata, "split_list", None)
    return bool(
        is_hcu()
        and parallel.attn_cp_size > 1
        and is_cp_v2_active(forward_batch)
        and metadata is not None
        # The affine preprocessing kernel needs a real segment for every
        # natural zigzag block.  This matches the final NPU PCP path rather
        # than entering the kernel with identity-only empty blocks.
        and split_list
        and min(split_list) > 0
        and mode.is_context_parallel_extend()
        and not mode.is_mixed()
        and not mode.is_target_verify()
        and not mode.is_draft_extend_v2()
    )


def _validate_zigzag_metadata(metadata: Any, cp_size: int) -> None:
    required = (
        "split_list",
        "cp_reverse_index",
        "reverse_split_len",
        "per_rank_actual_token",
        "max_rank_len",
        "actual_seq_q_prev_list",
        "actual_seq_q_next_list",
        "cu_seqlens_q_combined_tensor",
    )
    missing = [name for name in required if getattr(metadata, name, None) is None]
    if missing:
        raise NotImplementedError(
            "KDA prefill CP currently requires zigzag CP metadata; "
            f"missing {missing}."
        )
    if len(metadata.per_rank_actual_token) != cp_size:
        raise ValueError(
            "KDA CP metadata/group size mismatch: "
            f"metadata={len(metadata.per_rank_actual_token)}, cp_size={cp_size}."
        )
    expected_segments = int(metadata.bs) * 2 * cp_size
    if len(metadata.split_list) != expected_segments:
        raise ValueError(
            "KDA CP zigzag segment count mismatch: "
            f"segments={len(metadata.split_list)}, expected={expected_segments}."
        )
    if min(metadata.split_list) <= 0:
        raise NotImplementedError(
            "KDA affine PCP currently requires every zigzag block to be non-empty."
        )


def _natural_segment_owner(segment: int, cp_size: int) -> tuple[int, int]:
    """Return ``(cp_rank, local_half)`` for a natural zigzag segment."""
    if segment < cp_size:
        return segment, 0
    return 2 * cp_size - 1 - segment, 1


def _all_gather_cp(x: torch.Tensor) -> torch.Tensor:
    parallel = get_parallel()
    gathered = x.new_empty((parallel.attn_cp_size * x.shape[0], *x.shape[1:]))
    parallel.attn_cp_group.all_gather_into_tensor(gathered, x.contiguous())
    return gathered


def prepare_kda_cp_conv_states(
    mixed_qkv: torch.Tensor,
    conv_state_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    forward_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build exact causal-convolution inputs for this rank's two segments.

    Only the last ``kernel_size - 1`` raw rows of every segment are gathered.
    This keeps the convolution communication independent of prompt length.

    Args:
        mixed_qkv: rank-local logical zigzag rows, ``[T_local, channels]``.
        conv_state_pool: persistent raw-input windows, ``[slots, W, channels]``.
        cache_indices: one persistent slot per request.

    Returns:
        Local segment initial windows in ``[2*bs, W, channels]`` order,
        the globally final window per request, local cumulative sequence
        lengths, and the corresponding CPU lengths.
    """
    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    cp_rank = parallel.attn_cp_rank
    metadata = forward_batch.attn_cp_metadata
    _validate_zigzag_metadata(metadata, cp_size)

    bs = int(metadata.bs)
    local_lens = [
        *[int(x) for x in metadata.actual_seq_q_prev_list],
        *[int(x) for x in metadata.actual_seq_q_next_list],
    ]
    logical_tokens = sum(local_lens)
    if mixed_qkv.shape[0] != logical_tokens:
        raise ValueError(
            "KDA CP logical row mismatch: "
            f"tensor={mixed_qkv.shape[0]}, metadata={logical_tokens}."
        )
    if cache_indices.numel() != bs or bool((cache_indices < 0).any()):
        raise ValueError(
            "KDA CP requires one valid recurrent-state slot per request: "
            f"slots={cache_indices.tolist()}, bs={bs}."
        )

    window = int(conv_state_pool.shape[1])
    channels = int(mixed_qkv.shape[-1])
    if conv_state_pool.shape[-1] != channels:
        raise ValueError(
            "KDA CP convolution state width mismatch: "
            f"pool={tuple(conv_state_pool.shape)}, input={tuple(mixed_qkv.shape)}."
        )

    local_tails = mixed_qkv.new_zeros((2 * bs, window, channels))
    offset = 0
    for index, length in enumerate(local_lens):
        take = min(length, window)
        if take:
            local_tails[index, -take:].copy_(
                mixed_qkv[offset + length - take : offset + length]
            )
        offset += length

    gathered = _all_gather_cp(local_tails).view(
        cp_size, 2, bs, window, channels
    )
    persistent = conv_state_pool.index_select(0, cache_indices.to(torch.long)).clone()
    local_initial = mixed_qkv.new_empty((2, bs, window, channels))
    final_states = mixed_qkv.new_empty((bs, window, channels))

    segment_count = 2 * cp_size
    for batch_id in range(bs):
        rolling = persistent[batch_id]
        for segment in range(segment_count):
            owner, half = _natural_segment_owner(segment, cp_size)
            if owner == cp_rank:
                local_initial[half, batch_id].copy_(rolling)
            length = int(metadata.split_list[batch_id * segment_count + segment])
            take = min(length, window)
            if take:
                tail = gathered[owner, half, batch_id, -take:]
                rolling = torch.cat((rolling, tail), dim=0)[-window:]
        final_states[batch_id].copy_(rolling)

    cu_seqlens = metadata.cu_seqlens_q_combined_tensor.to(
        device=mixed_qkv.device, dtype=torch.int32
    )
    return local_initial.reshape(2 * bs, window, channels), final_states, cu_seqlens, local_lens


def _compose_affine_states(
    local_affine: torch.Tensor,
    initial_state: torch.Tensor,
    metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather affine maps and return local segment inputs plus global final state."""
    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    cp_rank = parallel.attn_cp_rank
    bs = int(metadata.bs)
    value_dim = int(initial_state.shape[-2])
    key_dim = int(initial_state.shape[-1])

    gathered = _all_gather_cp(local_affine).view(
        cp_size,
        2,
        bs,
        *local_affine.shape[1:],
    )
    # Keep the composed maps and segment boundary states in fp32.  The
    # recurrent kernel itself accumulates in fp32, so retaining this precision
    # across CP segments avoids an extra BF16 round-trip at every boundary.
    affine_dtype = local_affine.dtype
    local_inputs = torch.empty(
        (2, bs, *initial_state.shape[1:]),
        device=local_affine.device,
        dtype=affine_dtype,
    )
    state = initial_state.to(device=local_affine.device, dtype=affine_dtype)

    for segment in range(2 * cp_size):
        owner, half = _natural_segment_owner(segment, cp_size)
        if owner == cp_rank:
            local_inputs[half].copy_(state)
        transform = gathered[owner, half]
        constant = transform[..., :value_dim, :]
        matrix = transform[..., value_dim : value_dim + key_dim, :]
        state = constant + torch.matmul(state, matrix)

    return local_inputs.reshape(2 * bs, *initial_state.shape[1:]), state.to(
        initial_state.dtype
    )


def run_kda_affine_prefill_cp(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    forward_batch: Any,
) -> torch.Tensor:
    """Run KDA PCP using Triton affine-state preprocessing and composition."""
    # Keep kernel imports off the module import path.  CP-off and pure metadata
    # tests must not initialize HCU/Triton merely because kda_backend imports
    # this helper module.
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.kda import (
        RCP_LN2,
        chunk_gla_fwd_o_gk,
        kda_gate_chunk_cumsum,
    )
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

    metadata = forward_batch.attn_cp_metadata
    cp_size = get_parallel().attn_cp_size
    _validate_zigzag_metadata(metadata, cp_size)
    if q.shape != k.shape or q.shape[-2] != v.shape[-2]:
        raise NotImplementedError(
            "KDA affine PCP currently requires equal Q/K/V head counts; "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )

    cu_seqlens = metadata.cu_seqlens_q_combined_tensor.to(
        device=q.device, dtype=torch.int32
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    q = l2norm_fwd(q.contiguous())
    k = l2norm_fwd(k.contiguous())
    v = v.contiguous()
    beta = beta.contiguous()
    if _use_kda_hcu_op():
        from boltops.fla.kda.triton import fused_kda_gate_chunk_cumsum

        g = fused_kda_gate_chunk_cumsum(
            g.contiguous(),
            beta,
            A_log=A_log,
            g_bias=dt_bias,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=_CHUNK_SIZE,
        )[0]
    else:
        g = kda_gate_chunk_cumsum(
            g.contiguous(),
            A_log=A_log,
            chunk_size=_CHUNK_SIZE,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )

    num_chunks = int(chunk_indices.shape[0])
    small_grid = q.shape[0] * num_chunks * q.shape[-2] <= 256
    w, u, _, gated_k, query_key, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=k.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=_CHUNK_SIZE,
        chunk_indices=chunk_indices,
        fuse_diagonal=small_grid,
        fuse_recompute=small_grid,
    )

    bs = int(metadata.bs)
    num_segments = 2 * bs
    heads = int(v.shape[-2])
    value_dim = int(v.shape[-1])
    key_dim = int(k.shape[-1])
    if value_dim != key_dim:
        raise NotImplementedError(
            "The first HCU affine PCP implementation requires K == V; "
            f"got K={key_dim}, V={value_dim}."
        )

    # Affine maps are an intermediate numerical representation. Keep them in
    # fp32 even when the persistent recurrent-state pool is BF16.
    affine_states = torch.zeros(
        (num_segments, heads, value_dim + key_dim, key_dim),
        device=q.device,
        dtype=torch.float32,
    )
    identity = torch.eye(key_dim, dtype=affine_states.dtype, device=q.device)
    affine_states[..., value_dim:, :].copy_(
        identity.view(1, 1, key_dim, key_dim)
    )
    affine_u = torch.cat((u, torch.zeros_like(u)), dim=-1)
    segment_indices = torch.arange(
        num_segments, device=q.device, dtype=cache_indices.dtype
    )
    chunk_gated_delta_rule_fwd_h(
        k=gated_k,
        w=w,
        u=affine_u,
        gk=g,
        initial_state=affine_states,
        initial_state_indices=segment_indices,
        save_new_value=False,
        materialize_chunk_states=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )

    persistent = ssm_states.index_select(0, cache_indices.to(torch.long))
    local_initial, final_state = _compose_affine_states(
        affine_states, persistent, metadata
    )

    chunk_states, new_values = chunk_gated_delta_rule_fwd_h(
        k=gated_k,
        w=w,
        u=u,
        gk=g,
        initial_state=local_initial,
        initial_state_indices=segment_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    output = chunk_gla_fwd_o_gk(
        q=q,
        v=new_values,
        g=g,
        A=query_key,
        h=chunk_states,
        o=v,
        scale=key_dim**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=_CHUNK_SIZE,
        chunk_indices=chunk_indices,
    )
    ssm_states.index_copy_(0, cache_indices.to(torch.long), final_state)
    return output


def forward_kda_affine_prefill_cp(
    backend: Any,
    layer: Any,
    forward_batch: Any,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    causal_conv_fn: Any,
) -> torch.Tensor:
    """Full KDA CP forward, including CP-aware short convolution state."""
    metadata = backend.forward_metadata
    if metadata.has_mamba_track_mask:
        raise NotImplementedError(
            "KDA affine PCP radix checkpoints are not enabled in the first HCU "
            "version. Run the initial validation with --disable-radix-cache."
        )

    cache_indices = metadata.mamba_cache_indices
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_state_pool = cache.conv[0]
    ssm_states = cache.temporal

    # CP-v2 pads the local shard for collectives.  KDA's varlen geometry is
    # logical and must never expose those rows to the convolution or FLA
    # kernels.  The model layer normally trims before entering the backend;
    # keep this boundary check here as well because DP-attention and eager
    # runners can pass the physical buffer directly.
    metadata_cp = forward_batch.attn_cp_metadata
    local_seq_lens_cpu = [
        *[int(x) for x in metadata_cp.actual_seq_q_prev_list],
        *[int(x) for x in metadata_cp.actual_seq_q_next_list],
    ]
    logical_tokens = sum(local_seq_lens_cpu)
    if mixed_qkv.shape[0] != logical_tokens:
        if mixed_qkv.shape[0] < logical_tokens:
            raise ValueError(
                "KDA CP input is shorter than its logical zigzag shard: "
                f"tensor={mixed_qkv.shape[0]}, metadata={logical_tokens}."
            )
        mixed_qkv = mixed_qkv[:logical_tokens]
        a = a[:, :logical_tokens]
        b = b[:, :logical_tokens]

    (
        local_conv_states,
        final_conv_states,
        local_cu_seqlens,
        prepared_seq_lens_cpu,
    ) = prepare_kda_cp_conv_states(
        mixed_qkv,
        conv_state_pool,
        cache_indices,
        forward_batch,
    )
    local_seq_lens_cpu = prepared_seq_lens_cpu

    # causal_conv1d_fn consumes channel-first states [N, channels, window].
    conv_states = local_conv_states.transpose(-1, -2).contiguous()
    local_cache_indices = torch.arange(
        len(local_seq_lens_cpu),
        device=mixed_qkv.device,
        dtype=cache_indices.dtype,
    )
    has_initial_state = torch.ones(
        len(local_seq_lens_cpu), dtype=torch.bool, device=mixed_qkv.device
    )

    splits = [layer.q_dim, layer.k_dim, layer.v_dim]
    q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
    q_weight, k_weight, v_weight = layer.conv_weights.split(splits, dim=0)
    q_state, k_state, v_state = conv_states.split(splits, dim=-2)
    if layer.bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)

    def run_conv(x, weight, bias, state):
        return causal_conv_fn(
            x,
            weight,
            bias,
            activation="silu",
            conv_states=state,
            has_initial_state=has_initial_state,
            cache_indices=local_cache_indices,
            query_start_loc=local_cu_seqlens,
            seq_lens_cpu=local_seq_lens_cpu,
        ).transpose(0, 1)

    q = run_conv(q, q_weight, q_bias, q_state)
    k = run_conv(k, k_weight, k_bias, k_state)
    v = run_conv(v, v_weight, v_bias, v_state)

    q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
    output = run_kda_affine_prefill_cp(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=layer.lower_bound,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        forward_batch=forward_batch,
    )
    conv_state_pool.index_copy_(
        0, cache_indices.to(torch.long), final_conv_states.to(conv_state_pool.dtype)
    )
    return output


__all__ = [
    "forward_kda_affine_prefill_cp",
    "kda_use_prefill_cp",
    "prepare_kda_cp_conv_states",
    "run_kda_affine_prefill_cp",
]
