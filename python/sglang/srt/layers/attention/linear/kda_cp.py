"""Kimi K3 zigzag prefill context parallel helpers.

The KDA recurrence is affine in its incoming recurrent state.  Each local
zigzag segment is summarized as ``state_out = C + state_in @ M``.  CP ranks
all-gather only ``[C | M]``, compose the transforms in natural token order,
and then execute their local segments with the correct incoming states.

This is the HCU counterpart of the NPU affine-state implementation in
sgl-project/sglang#35226.  It provides HCU affine preprocess/merge kernels and
keeps the generic Triton formulation as a correctness fallback oracle.
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


def _use_kda_hcu_affine() -> bool:
    """Use the PR #745-style compact HCU affine prepass by default on HCU."""
    from sglang.srt.utils import get_bool_env_var

    return is_hcu() and get_bool_env_var(
        "SGLANG_KDA_USE_HCU_AFFINE", default="true"
    )


def kda_use_prefill_cp(forward_batch: Any) -> bool:
    """Return whether KDA PCP owns this forward, failing closed once active."""
    parallel = get_parallel()
    mode = forward_batch.forward_mode
    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    split_list = getattr(metadata, "split_list", None)
    if (
        not is_hcu()
        or parallel.attn_cp_size <= 1
        or not is_cp_v2_active(forward_batch)
    ):
        return False

    supported = bool(
        metadata is not None
        and split_list
        and min(int(length) for length in split_list) > 0
        and mode is not None
        and mode.is_context_parallel_extend()
        and not mode.is_mixed()
        and not mode.is_target_verify()
        and not mode.is_draft_extend_v2()
    )
    if not supported:
        raise NotImplementedError(
            "Active KDA prefill CP cannot fall back to the ordinary local-shard "
            "KDA path. It requires non-empty zigzag metadata and a non-mixed "
            "context-parallel extend batch."
        )
    return True


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
    bs = int(metadata.bs)
    if len(metadata.actual_seq_q_prev_list) != bs or len(
        metadata.actual_seq_q_next_list
    ) != bs:
        raise ValueError(
            "KDA CP actual sequence metadata/batch mismatch: "
            f"bs={bs}, prev={len(metadata.actual_seq_q_prev_list)}, "
            f"next={len(metadata.actual_seq_q_next_list)}."
        )
    if len(metadata.cu_seqlens_q_combined_tensor) != 2 * bs + 1:
        raise ValueError(
            "KDA CP cu_seqlens/metadata mismatch: "
            f"cu={len(metadata.cu_seqlens_q_combined_tensor)}, bs={bs}."
        )
    expected_segments = bs * 2 * cp_size
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
    if cache_indices.numel() != bs:
        raise ValueError(
            "KDA CP requires one valid recurrent-state slot per request: "
            f"slots={cache_indices.tolist()}, bs={bs}."
        )
    # ``cache_indices`` is shared by all 69 KDA layers in one ForwardBatch.
    # Checking a device tensor with bool(any()) in every layer forces a
    # device-to-host synchronization and serializes the CP collectives.  Keep
    # the defensive validation, but perform it only once per forward.
    if not forward_batch.kda_cp_cache_indices_validated:
        if bool((cache_indices < 0).any()):
            raise ValueError(
                "KDA CP requires one valid recurrent-state slot per request: "
                f"slots={cache_indices.tolist()}, bs={bs}."
            )
        forward_batch.kda_cp_cache_indices_validated = True

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

    communicated = local_affine.to(torch.float32)
    gathered = _all_gather_cp(communicated).to(local_affine.dtype).view(
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


def _compose_affine_states_key_major(
    local_affine: torch.Tensor,
    initial_state: torch.Tensor,
    metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose PR #745's ``[K, V + K]`` affine maps in FP32.

    The existing fallback stores ``[V + K, K]`` (value-major) maps.  The HCU
    kernels intentionally retain the NPU contract, so transpose the recurrent
    state only at this boundary and return the same ``[V, K]`` state layout to
    the existing output kernels.
    """
    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    cp_rank = parallel.attn_cp_rank
    bs = int(metadata.bs)
    key_dim = int(local_affine.shape[-2])
    affine_dim = int(local_affine.shape[-1])
    value_dim = affine_dim - key_dim
    if initial_state.shape[-2:] != (value_dim, key_dim):
        raise ValueError(
            "KDA HCU affine state shape mismatch: "
            f"affine={tuple(local_affine.shape)}, "
            f"initial={tuple(initial_state.shape)}"
        )

    communicated = local_affine.to(torch.float32)
    gathered = _all_gather_cp(communicated).to(torch.float32).view(
        cp_size,
        2,
        bs,
        *local_affine.shape[1:],
    )
    state = initial_state.to(device=local_affine.device, dtype=torch.float32)
    state = state.transpose(-1, -2).contiguous()
    local_inputs = torch.empty(
        (2, bs, *state.shape[1:]),
        device=local_affine.device,
        dtype=torch.float32,
    )

    from sglang.srt.utils import get_bool_env_var

    # Use the fused HCU merge for the validated batched K=128 path.
    use_fused_merge = get_bool_env_var(
        "SGLANG_KDA_CP_HCU_AFFINE_MERGE", default="true"
    )
    if use_fused_merge and key_dim == 128 and cp_size > 1:
        from sglang.kernels.ops.attention.fla.kda_affine_hcu import (
            merge_kda_cp_affine_states,
        )

        fused_local = torch.empty(
            (2, *state.shape),
            device=state.device,
            dtype=torch.float32,
        )
        fused_final = torch.empty_like(state)
        merge_kda_cp_affine_states(
            gathered=gathered,
            initial_state=state,
            local_initial=fused_local,
            final_state=fused_final,
            cp_rank=cp_rank,
            track_step=-1,
        )
        local_inputs.copy_(fused_local)
        state = fused_final
    else:
        for segment in range(2 * cp_size):
            owner, half = _natural_segment_owner(segment, cp_size)
            if owner == cp_rank:
                local_inputs[half].copy_(state)
            transform = gathered[owner, half]
            constant = transform[..., :value_dim]
            matrix = transform[..., value_dim : value_dim + key_dim]
            state = torch.matmul(matrix, state) + constant

    # FLA consumes one varlen sequence per row, so flatten the two local
    # zigzag halves and batch dimensions exactly like the value-major path:
    # [2, bs, H, V, K] -> [2*bs, H, V, K].
    local_inputs = local_inputs.transpose(-1, -2).reshape(
        2 * bs, *initial_state.shape[1:]
    ).contiguous()
    return local_inputs, state.transpose(-1, -2).to(initial_state.dtype).contiguous()


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
    output_kernel: Optional[Any] = None,
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
    metadata_tokens = sum(
        int(x) for x in metadata.actual_seq_q_prev_list
    ) + sum(int(x) for x in metadata.actual_seq_q_next_list)
    if int(q.shape[1]) != metadata_tokens:
        raise ValueError(
            "KDA CP local token mismatch before FLA: "
            f"q={int(q.shape[1])}, metadata={metadata_tokens}."
        )

    cu_seqlens = metadata.cu_seqlens_q_combined_tensor.to(
        device=q.device, dtype=torch.int32
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    # The affine pre-pass needs normalized Q/K and activated gates, while
    # FlashKDA fuses those transforms internally. Preserve the raw local
    # segment tensors so the optional fast output pass does not normalize or
    # activate them twice.
    raw_q = q.contiguous()
    raw_k = k.contiguous()
    raw_v = v.contiguous()
    raw_g = g.contiguous()
    raw_beta = beta.contiguous()

    q = l2norm_fwd(raw_q)
    k = l2norm_fwd(raw_k)
    v = raw_v
    beta = raw_beta
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

    segment_indices = torch.arange(
        num_segments, device=q.device, dtype=cache_indices.dtype
    )
    use_hcu_affine = _use_kda_hcu_affine() and key_dim == 128
    if use_hcu_affine:
        # PR #745-style HCU affine producer: emit compact FP32 maps in the
        # NPU-compatible [segments, heads, K, V+K] layout.
        from sglang.kernels.ops.attention.fla.kda_affine_hcu import (
            chunk_gated_delta_rule_fwd_affine_hcu,
        )

        affine_states = chunk_gated_delta_rule_fwd_affine_hcu(
            k=gated_k,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
        )
    else:
        # Generic fallback retained as the numerical oracle.  It stores the
        # equivalent transform in value-major [V+K, K] layout.
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
    # HCU Kimi temporal pools may keep a singleton state-group axis:
    # [slots, 1, H, V, K]. FLA expects [batch, H, V, K], while the pool
    # write-back must retain its original layout. Normalize only at this
    # kernel boundary and restore the axis after affine composition.
    if persistent.ndim == 5 and persistent.shape[1] == 1:
        persistent_for_kernel = persistent[:, 0]
        restore_state_group_axis = True
    elif persistent.ndim == 4:
        persistent_for_kernel = persistent
        restore_state_group_axis = False
    else:
        raise ValueError(
            "KDA CP temporal state must be [batch, H, V, K] or "
            f"[batch, 1, H, V, K], got {tuple(persistent.shape)}."
        )
    if use_hcu_affine:
        local_initial, final_state = _compose_affine_states_key_major(
            affine_states, persistent_for_kernel, metadata
        )
    else:
        local_initial, final_state = _compose_affine_states(
            affine_states, persistent_for_kernel, metadata
        )
    if restore_state_group_axis:
        final_state = final_state.unsqueeze(1)

    if output_kernel is not None:
        # FlashKDA accepts one initial state per varlen sequence, including
        # FP32 states. The two local zigzag halves are represented as 2*bs
        # independent sequences here; the affine composition above supplies
        # the exact incoming state for each half. Its local final states are
        # scratch only--the globally composed final_state remains canonical.
        output = output_kernel.extend(
            raw_q,
            raw_k,
            raw_v,
            raw_g,
            raw_beta,
            ssm_states=local_initial,
            cache_indices=segment_indices,
            query_start_loc=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            extend_seq_lens_cpu=[
                *[int(x) for x in metadata.actual_seq_q_prev_list],
                *[int(x) for x in metadata.actual_seq_q_next_list],
            ],
        )
    else:
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
    from sglang.srt.utils import get_bool_env_var

    if metadata.has_mamba_track_mask:
        raise NotImplementedError(
            "KDA affine PCP radix checkpoints are not enabled in the first HCU "
            "version. Run the initial validation with --disable-radix-cache."
        )

    # CP-v2 may expose the static recurrent-state index buffer as [bs, 1].
    # The KDA state/conv pools and FLA kernels use one flat slot per request;
    # preserve the values while removing only this metadata-only singleton axis.
    raw_cache_indices = metadata.mamba_cache_indices
    expected_bs = int(forward_batch.attn_cp_metadata.bs)
    if raw_cache_indices.numel() != expected_bs:
        raise ValueError(
            "KDA CP requires one recurrent-state slot per request: "
            f"index_shape={tuple(raw_cache_indices.shape)}, bs={expected_bs}."
        )
    cache_indices = raw_cache_indices.reshape(-1)
    if not forward_batch.kda_cp_cache_indices_validated:
        if bool((cache_indices < 0).any()):
            raise ValueError(
                "KDA CP cannot process padding/idle rows with negative mamba slots: "
                f"indices={cache_indices.tolist()}."
            )
        forward_batch.kda_cp_cache_indices_validated = True
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
    output_kernel = None
    if get_bool_env_var("SGLANG_KDA_CP_FLASHKDA_OUTPUT", default="true"):
        from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
            FlashKDAKernel,
        )

        candidate = backend.kernel_dispatcher.extend_kernel
        # FlashKDA is an output-pass optimization, not a PCP correctness
        # requirement. Other prefill backends retain the true-PCP Triton output
        # pass after the same affine-state composition.
        if isinstance(candidate, FlashKDAKernel):
            output_kernel = candidate

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
        output_kernel=output_kernel,
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
