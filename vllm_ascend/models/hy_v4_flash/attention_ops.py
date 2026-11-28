# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash attention ops."""

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950

from .config import uses_prefill_ops

ACL_FORMAT_ND = 2
BF16_QUERY_PROJECTION_LAYER = 1
QUERY_CHUNK = 16
C8_PREFILL_MAX_HEADS = 16
TRANSPOSE_BMM_MAX_TOKENS = 65536
# Larger gathers benefit from the bulk CANN copy/cast path.
MAX_FUSED_C8_READ_ROWS = 4096
# The bulk CANN writer has lower latency for large sparse prefill batches.
MAX_FUSED_C8_WRITE_TOKENS = 4096
# 512 captured decode tokens with 16 index heads fit in this bound.
MAX_FUSED_INDEXPOOL_ROWS = 8192
# Bound the vector kernel to the decode tensor sizes measured on 950DT.
MAX_FUSED_OUTPUT_GATE_ELEMENTS = 96 * 2048
# Equal-width CANN shapes verified on 950DT at TP8.
BATCH_VERIFY_MAX_QUERIES = 6
BATCH_VERIFY_SWA_SHAPE = (8, 512)
BATCH_VERIFY_SPARSE_SHAPE = (4, 2048)
# Queries, index heads, compression ratio, and complete groups recalled.
PAGED_POOL_DECODE_SHAPE = (6, 16, 4, 511)
# Store three int64 MRoPE coordinates in the BF16 sidecar by bit reinterpretation.
POSITION_LANES = 3 * torch.int64.itemsize // torch.bfloat16.itemsize


def _head_bmm(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # [tokens, heads, in] x [heads, in, out] -> [tokens, heads, out].
    if (
        tokens.device.type == "npu"
        and tokens.dtype == weight.dtype == torch.bfloat16
        and 0 < tokens.shape[0] < TRANSPOSE_BMM_MAX_TOKENS
        and hasattr(torch_npu, "npu_transpose_batchmatmul")
    ):
        return torch_npu.npu_transpose_batchmatmul(tokens, weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
    return torch.bmm(tokens.transpose(0, 1).contiguous(), weight).transpose(0, 1)


GATE_BLOCK_SIZE = 1024


@triton.jit
def _output_gate_kernel(value, logits, output, count: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < count
    x = tl.load(value + index, live, 0).to(tl.float32)
    gate = tl.load(logits + index, live, 0).to(tl.float32)
    # Preserve the BF16 sigmoid intermediate used by unquantized projection.
    probability = tl.sigmoid(gate).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + index, x * probability, live)


def fused_output_gate(value, logits):
    output = torch.empty_like(value)
    _output_gate_kernel[(triton.cdiv(value.numel(), GATE_BLOCK_SIZE),)](
        value,
        logits,
        output,
        value.numel(),
        GATE_BLOCK_SIZE,
        enable_fp_fusion=False,
    )
    return output


_LOGIT_TILE = 512


@triton.jit(do_not_specialize=["rows", "width", "candidate_row_stride"])
def _build_mla_logits(
    scores,
    candidates,
    sink,
    logits,
    rows,
    width,
    candidate_row_stride,
    CANDIDATE_COL_STRIDE: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    SHORT_ROW: tl.constexpr,
):
    start = tl.program_id(0) * TILE
    offsets = start + tl.arange(0, TILE)
    row_width = width + 1
    if SHORT_ROW:
        row = offsets // row_width
        column = offsets % row_width
    else:
        # A tile spans at most two rows. Divide once per program.
        column = start % row_width + tl.arange(0, TILE)
        crosses_row = column >= row_width
        row = start // row_width + crosses_row.to(tl.int32)
        column = tl.where(crosses_row, column - row_width, column)
    live = row < rows
    is_sink = column == width
    is_candidate = live & ~is_sink

    candidate_offset = row // HEADS * candidate_row_stride + column * CANDIDATE_COL_STRIDE
    candidate = tl.load(candidates + candidate_offset, is_candidate, -1)
    score = tl.load(scores + offsets - row, is_candidate, 0.0)
    sink_score = tl.load(sink + row % HEADS * SINK_STRIDE, live & is_sink, 0.0)

    logit = tl.where(candidate >= 0, score * SCALE, float("-inf"))
    tl.store(logits + offsets, tl.where(is_sink, sink_score, logit), live)


def build_mla_logits(scores, candidates, sink, scale):
    """Build contiguous FP32 logits with the learned sink in the final column."""
    queries, heads, width = scores.shape
    logits = torch.empty((queries, heads, width + 1), dtype=torch.float32, device=scores.device)
    _build_mla_logits[(triton.cdiv(logits.numel(), _LOGIT_TILE),)](
        scores,
        candidates,
        sink,
        logits,
        queries * heads,
        width,
        *candidates.stride(),
        sink.stride(0),
        heads,
        scale,
        _LOGIT_TILE,
        width + 1 < _LOGIT_TILE,
        num_warps=1,
        multibuffer=False,
    )
    return logits


_MX_GROUP_SIZE = 32
_MX_SCALE_PAIR = 2
_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
_E4M3_MAX_EXPONENT = 8


@triton.jit(do_not_specialize=["num_groups"])
def _gate_dynamic_mx_quant(
    value,
    gate_logits,
    quantized,
    scales,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    E4M3_MAX: tl.constexpr,
    E4M3_MAX_EXPONENT: tl.constexpr,
):
    group = tl.program_id(0) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    valid = group < num_groups
    offset = group[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    value_f32 = tl.load(value + offset, valid[:, None], 0).to(tl.float32)
    logits = tl.load(gate_logits + offset, valid[:, None], 0).to(tl.float32)
    # Preserve the two BF16 outputs of the unfused sigmoid and multiplication.
    gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16)
    product = (value_f32 * gate.to(tl.float32)).to(tl.bfloat16)

    bits = product.to(tl.int16, bitcast=True).to(tl.int32)
    exponent = (bits & 0x7F80) >> 7
    group_exponent = tl.max(exponent, axis=1)
    nonfinite = group_exponent == 255
    scale = tl.maximum(group_exponent - E4M3_MAX_EXPONENT, 0)
    reciprocal = ((254 - scale) << 23).to(tl.float32, bitcast=True)
    scaled = product.to(tl.float32) * reciprocal[:, None]
    clipped = tl.minimum(tl.maximum(scaled, -E4M3_MAX), E4M3_MAX)
    fp8 = tl.cast(clipped, tl.float8e4nv, fp_downcast_rounding="rtne")
    # CANN preserves signed zero for tiny groups, and emits all-zero payload
    # with an E8M0 NaN scale when a group contains any nonfinite value.
    signed_zero = (bits >> 8) & 0x80
    codes = tl.where(scale[:, None] == 0, signed_zero, fp8.to(tl.uint8, bitcast=True))
    tl.store(quantized + offset, tl.where(nonfinite[:, None], 0, codes), valid[:, None])
    tl.store(scales + group, tl.where(nonfinite, 255, scale), valid)


def gate_dynamic_mx_quant(value: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return OCP MXFP8 activation bytes for contiguous BF16 attention outputs."""
    rows, width = value.shape
    num_groups = rows * width // _MX_GROUP_SIZE
    groups_per_program = 16 if num_groups <= 512 else 32
    quantized = torch.empty_like(value, dtype=torch.uint8)
    scales = torch.empty(
        (rows, width // (_MX_GROUP_SIZE * _MX_SCALE_PAIR), _MX_SCALE_PAIR), dtype=torch.uint8, device=value.device
    )
    _gate_dynamic_mx_quant[(triton.cdiv(num_groups, groups_per_program),)](
        value,
        gate_logits,
        quantized,
        scales,
        num_groups,
        _MX_GROUP_SIZE,
        groups_per_program,
        _E4M3_MAX,
        _E4M3_MAX_EXPONENT,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return quantized.view(torch.float8_e4m3fn), scales


_POOL_RATIO = 4
_POOL_BUDGET = 511
_CANDIDATES = (_POOL_BUDGET + 1) * _POOL_RATIO
_EXPAND_TILE = 512


@triton.jit(do_not_specialize=["rows"])
def _expand_decode_candidates(
    groups,
    visible,
    output,
    rows,
    TILE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = index // CANDIDATES
    column = index % CANDIDATES
    group = column // RATIO
    lane = column % RATIO
    live = row < rows
    position = tl.load(visible + row, live, other=-1).to(tl.int64)
    cutoff = position >> 2
    selected = tl.load(groups + row * BUDGET + group, live & (group < BUDGET), other=-1).to(tl.int64)
    history = tl.where((selected >= 0) & (selected < cutoff), selected * RATIO + lane, -1)
    tail = cutoff * RATIO + lane
    candidate = tl.where(group < BUDGET, history, tl.where(tail <= position, tail, -1))
    tl.store(output + index, candidate, live)


def expand_decode_candidates(groups, visible):
    """Keep the sorted LI order and return the original INT64 candidate contract."""
    rows = groups.shape[0]
    output = torch.empty((rows, _CANDIDATES), dtype=torch.int64, device=groups.device)
    if rows:
        _expand_decode_candidates[(triton.cdiv(rows * _CANDIDATES, _EXPAND_TILE),)](
            groups.contiguous(),
            visible.contiguous(),
            output,
            rows,
            _EXPAND_TILE,
            _POOL_RATIO,
            _POOL_BUDGET,
            _CANDIDATES,
        )
    return output


@triton.jit
def _sink_softmax_pv(
    scores,
    values,
    selected,
    sink,
    output,
    heads: tl.constexpr,
    candidates: tl.constexpr,
    value_width: tl.constexpr,
    value_stride: tl.constexpr,
    selected_stride: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INVALID_MASK: tl.constexpr,
):
    row_head = tl.program_id(0)
    row = row_head // heads
    head = row_head % heads
    k = tl.arange(0, BLOCK_K)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_k = k < candidates
    ids = tl.load(selected + row * selected_stride + k, valid_k, 1 if INVALID_MASK else -1)
    valid_ids = ids == 0 if INVALID_MASK else ids >= 0
    logits = tl.load(scores + row_head * candidates + k, valid_k, 0) * scale
    logits = tl.where(valid_k & valid_ids, logits, float("-inf"))
    sink_logit = tl.load(sink + head).to(tl.float32)
    maximum = tl.maximum(tl.max(logits, 0), sink_logit)
    p = tl.exp(logits - maximum)
    denominator = tl.sum(p, 0) + tl.exp(sink_logit - maximum)
    result = tl.zeros((BLOCK_D,), tl.float32)
    for start in range(tl.cdiv(candidates, 128)):
        kk = start * 128 + tl.arange(0, 128)
        index = tl.load(selected + row * selected_stride + kk, kk < candidates, 1 if INVALID_MASK else -1)
        valid_index = index == 0 if INVALID_MASK else index >= 0
        s = tl.load(scores + row_head * candidates + kk, kk < candidates, 0) * scale
        probability = tl.exp(tl.where((kk < candidates) & valid_index, s, float("-inf")) - maximum) / denominator
        value = tl.load(
            values + row * candidates * value_stride + kk[:, None] * value_stride + d[None, :],
            ((kk < candidates) & valid_index)[:, None] & (d[None, :] < value_width),
            0,
        ).to(tl.float32)
        result += tl.sum(probability[:, None] * value, 0)
    tl.store(output + row_head * value_width + d, result, d < value_width)


def fused_fp32_mla_context(query, selected, kv, sink, scale, *, invalid_mask=None, scores=None, output_dtype=None):
    rows, heads, _ = query.shape
    candidates = selected.shape[-1]
    values = kv.contiguous()
    if scores is None:
        scores = torch.bmm(query.float(), values.transpose(1, 2))
    scores = scores.contiguous()
    indices = selected if invalid_mask is None else invalid_mask.reshape(rows, candidates)
    output = torch.empty((rows, heads, 512), device=query.device, dtype=output_dtype or query.dtype)
    if rows == 0 or candidates == 0:
        return output.zero_()
    block_d = 64
    _sink_softmax_pv[(rows * heads, triton.cdiv(512, block_d))](
        scores,
        values,
        indices,
        sink,
        output,
        heads,
        candidates,
        512,
        values.shape[-1],
        indices.stride(0),
        scale,
        triton.next_power_of_2(candidates),
        block_d,
        invalid_mask is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def uses_decode_fusions(config):
    """Decode fusion is part of the 950 runtime, independent of configuration."""
    return not uses_prefill_ops(config) and is_950()


def is_decode_batch(batch):
    return batch is not None and bool(batch.prefill_reqs) and not any(batch.prefill_reqs)


def native_rms_norm(value, norm):
    """Retain nn.RMSNorm's FP32 arithmetic and final output cast."""
    if value.device.type != "npu" or not is_950():
        return norm(value)
    dtype = value.dtype
    # torch.rms_norm uses the accumulation dtype for its implicit epsilon.
    # BF16/FP16 inputs accumulate in FP32 in this PyTorch runtime.
    eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps
    gamma = getattr(norm, "_hy4_decode_fp32_weight", None)
    if gamma is None:
        gamma = norm.weight.detach().float()
        norm._hy4_decode_fp32_weight = gamma
    return torch_npu.npu_rms_norm(value, gamma, eps)[0].to(dtype)


def _fia_kv_inputs(query, selected, kv):
    """Adapt legacy and stable-packed C8 layouts to the original FIA ABI."""
    from .fia_lengths import CompactFIAKV, PackedC8KV, compact_fia_rows

    if isinstance(kv, PackedC8KV):
        keys = kv.values.to(query.dtype)
        return (
            keys[..., :512].unsqueeze(2).contiguous(),
            keys[..., 512:].unsqueeze(2).contiguous(),
        )
    if isinstance(kv, tuple):
        nope, rope, _ = kv
        nope, rope = nope.to(query.dtype), rope.to(query.dtype)
    else:
        keys = kv.to(query.dtype).unsqueeze(2)
        nope = keys[..., :512].contiguous()
        rope = keys[..., 512:].contiguous()
    if not isinstance(kv, CompactFIAKV):
        nope, rope = compact_fia_rows(selected, nope, rope)
    return nope.contiguous(), rope.contiguous()


def native_mla_context(
    query, selected, kv, sink, scale, *, length_provider=None, metadata=None, scores=None, query_fp32=None
):
    """Run CANN FIA; retain fused sink correction and updatable graph tasks."""
    if query.device.type != "npu":
        return _reference_mla_context(
            query,
            selected,
            kv,
            sink,
            scale,
            length_provider=length_provider,
            metadata=metadata,
            scores=scores,
            query_fp32=query_fp32,
        )
    nope, rope = _fia_kv_inputs(query, selected, kv)
    kwargs = dict(
        query=query[..., :512].unsqueeze(1).contiguous(),
        key=nope,
        value=nope,
        query_rope=query[..., 512:].unsqueeze(1).contiguous(),
        key_rope=rope,
        num_query_heads=query.shape[1],
        num_key_value_heads=1,
        input_layout="BSND",
        sparse_mode=0,
        actual_seq_qlen=[1] * query.shape[0],
        softmax_scale=scale,
        return_softmax_lse=True,
    )
    kwargs.update(length_provider.resolve(metadata))
    if query.device.type == "npu" and torch.npu.is_current_stream_capturing():
        from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task

        workspace_kwargs = {**kwargs, "actual_seq_kvlen": [selected.shape[1]] * query.shape[0]}
        # Sequential FIA calls on one stream can reuse scratch storage. The
        # provider is call-specific metadata, not a workspace shape identity.
        workspace_key = (
            "hy4.mla.fia.workspace",
            tuple(kwargs["query"].shape),
            tuple(nope.shape),
            tuple(rope.shape),
            query.dtype,
            torch.npu.current_stream().npu_stream,
        )
        workspace = get_capture_resource(
            workspace_key,
            lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**workspace_kwargs),
        )
        output = torch.empty((query.shape[0], 1, query.shape[1], 512), dtype=query.dtype, device=query.device)
        lse = torch.empty((query.shape[0], query.shape[1], 1, 1), dtype=torch.float32, device=query.device)
        register_task(
            torch_npu.npu_fused_infer_attention_score_v2.out,
            {**kwargs, "workspace": workspace, "out": [output, lse]},
            length_provider,
        )
    else:
        output, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
    return fused_sink_context(output.squeeze(1), lse, sink, selected)


def _reference_mla_context(
    query, selected, kv, sink, scale, *, length_provider, metadata, scores=None, query_fp32=None
):
    """CPU reference only; the NPU runtime uses CANN FIA."""
    from .fia_lengths import CompactFIAKV, PackedC8KV

    invalid_mask = None
    if isinstance(kv, PackedC8KV):
        values, invalid_mask = kv
        if scores is not None:
            order = torch.argsort((selected < 0).to(torch.int32), dim=-1, stable=True)
            scores = scores.gather(-1, order[:, None].expand(-1, scores.shape[1], -1))
    elif isinstance(kv, tuple):
        nope, rope, mask = kv
        values = torch.cat((nope[:, :, 0].float(), rope[:, :, 0].float()), dim=-1)
        invalid_mask = mask
        if isinstance(kv, CompactFIAKV) and scores is not None:
            order = torch.argsort((selected < 0).to(torch.int32), dim=-1, stable=True)
            scores = scores.gather(-1, order[:, None].expand(-1, scores.shape[1], -1))
    else:
        values = kv.float()
    attention_query = query if query_fp32 is None else query_fp32
    # CPU execution is for reference tests; no CANN/Triton launch is possible.
    if scores is None:
        scores = torch.bmm(attention_query.float(), values.transpose(1, 2))
    invalid = selected < 0 if invalid_mask is None else invalid_mask.reshape_as(selected)
    logits = (scores * scale).masked_fill(invalid[:, None], -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(query.shape[0], -1, 1)), -1)
    probabilities = logits.softmax(-1)[..., :-1]
    return torch.bmm(probabilities, values[..., :512]).to(query.dtype)


def native_indexpool_rope(nope, even, odd, cos, sin, dtype, *, fuse_layout=True):
    """Map HY4's even/odd input to CANN half layout and retain its output order."""
    layout = None
    if fuse_layout:
        from .rope_layout import pack_rope_inputs

        layout = pack_rope_inputs(even, odd, cos, sin)
    if layout is None:
        packed = torch.cat((even, odd), -1)
        rows = packed.shape[0]
        width = packed.shape[-1]
        packed_cos = torch.cat((cos, cos), -1).reshape(rows, 1, 1, width)
        packed_sin = torch.cat((sin, sin), -1).reshape(rows, 1, 1, width)
    else:
        packed, packed_cos, packed_sin = layout
        rows, width = packed.shape[0], packed.shape[-1]
    heads = packed.shape[1] if packed.ndim == 3 else 1
    rotated = torch_npu.npu_rotary_mul(
        packed.reshape(rows, 1, heads, width),
        packed_cos,
        packed_sin,
        rotary_mode="half",
    ).reshape_as(packed)
    return torch.cat((nope, rotated.to(dtype)), -1)


def prepare_indexpool_selection(visible, query_rows, pool_rows):
    """Prepare layer-independent selection metadata once per request/forward."""
    import cann_ops_transformer  # noqa: F401 - registers the CANN 9.3 operators

    ratio, budget = 4, 511
    cutoff = visible // ratio
    used = cutoff[-1:].to(torch.int32).clamp_min(budget + 1)
    residual = (visible[-1:] % ratio).to(torch.int32)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        16,
        1,
        128,
        budget,
        seqused_k=used,
        cmp_residual_k=residual,
        batch_size=1,
        max_seqlen_q=query_rows,
        max_seqlen_k=pool_rows,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    offsets = torch.arange(ratio, device=visible.device)
    return used, residual, metadata, cutoff, offsets


def native_indexpool_select(query, weights, pool, visible, selection_state=None, *, fuse_postprocess=False):
    """Score layer-specific keys using the shared selection geometry."""
    ratio, budget = 4, 511
    if selection_state is None:
        selection_state = prepare_indexpool_selection(visible, query.shape[0], pool.shape[0])
    used, residual, metadata, cutoff, offsets = selection_state
    groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query.to(torch.bfloat16).unsqueeze(0),
        pool.to(torch.bfloat16).view(1, -1, 1, 128),
        weights.unsqueeze(0),
        budget,
        seqused_k=used,
        cmp_residual_k=residual,
        metadata=metadata,
        max_seqlen_q=query.shape[0],
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
        return_value=0,
    )
    groups = groups.reshape(query.shape[0], budget).long().sort(-1).values
    if fuse_postprocess:
        return expand_decode_candidates(groups, visible)
    history = (
        (groups[..., None] * ratio + offsets)
        .masked_fill(((groups < 0) | (groups >= cutoff[:, None]))[..., None], -1)
        .flatten(1)
    )
    tail = cutoff[:, None] * ratio + offsets
    return torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), -1)


def reuse_prefill_metadata(device, key, create):
    if device.type != "npu" or not is_forward_context_available():
        return create()
    context = get_forward_context()
    if context.cudagraph_runtime_mode is not CUDAGraphMode.NONE:
        return create()
    stream = torch.npu.current_stream(device).npu_stream
    cache = context.additional_kwargs.setdefault("hy4_prefill_metadata", {})
    key = (device, stream, *key)
    if key not in cache:
        cache[key] = create()
    return cache[key]


def logical_positions(first, end, device):
    return reuse_prefill_metadata(
        device, ("logical_positions", first, end), lambda: torch.arange(first, end, device=device)
    )


def _make_window_candidates(first, queries, width, device):
    starts = torch.arange(first, first + queries, dtype=torch.int32, device=device)
    offsets = torch.arange(width, dtype=torch.int32, device=device)
    return starts[:, None] + offsets


def window_indices(first, queries, width, device):
    return reuse_prefill_metadata(
        device,
        ("c8_window_indices", first, queries, width),
        lambda: _make_window_candidates(first, queries, width, device),
    )


def _make_sparse_lengths(num_pages, queries, length, device):
    table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    query_ends = torch.full((1,), queries, dtype=torch.int32, device=device)
    kv_lengths = torch.full((1,), length, dtype=torch.int32, device=device)
    return table, query_ends, kv_lengths


def sparse_lengths(num_pages, queries, length, device):
    return reuse_prefill_metadata(
        device,
        ("c8_sparse_lengths", num_pages, queries, length),
        lambda: _make_sparse_lengths(num_pages, queries, length, device),
    )


HEADS_PER_PROGRAM = 8


@triton.jit
def _sink_context_kernel(
    context,
    lse,
    sink,
    selected,
    output,
    heads: tl.constexpr,
    width: tl.constexpr,
    candidates: tl.constexpr,
    candidate_block: tl.constexpr,
    head_block: tl.constexpr,
    value_block: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * head_block + tl.arange(0, head_block)
    col = tl.arange(0, value_block)
    valid_head = head < heads
    index = tl.arange(0, candidate_block)
    position = tl.load(selected + row * candidates + index, index < candidates, -1)
    valid_row = tl.sum((position >= 0).to(tl.int32), 0) > 0
    normalization = tl.load(lse + row * heads + head, valid_head, 0).to(tl.float32)
    bias = tl.load(sink + head, valid_head, 0).to(tl.float32)
    ratio = tl.sigmoid(normalization - bias)
    offset = (row * heads + head[:, None]) * width + col[None, :]
    live = valid_head[:, None] & (col[None, :] < width)
    value = tl.load(context + offset, live, 0).to(tl.float32)
    corrected = tl.where(valid_row, value * ratio[:, None], 0.0)
    tl.store(output + offset, corrected, live)


def fused_sink_context(context, lse, sink, selected):
    rows, heads, width = context.shape
    output = torch.empty(context.shape, device=context.device, dtype=context.dtype)
    _sink_context_kernel[(rows, triton.cdiv(heads, HEADS_PER_PROGRAM))](
        context.contiguous(),
        lse.reshape(rows, heads).contiguous(),
        sink.contiguous(),
        selected.contiguous(),
        output,
        heads,
        width,
        selected.shape[-1],
        triton.next_power_of_2(selected.shape[-1]),
        HEADS_PER_PROGRAM,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output
