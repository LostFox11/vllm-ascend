# SPDX-License-Identifier: Apache-2.0
"""Pure layout packing for the existing FP32 rotary operator."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_rope_inputs(
    even,
    odd,
    cos,
    sin,
    packed,
    packed_cos,
    packed_sin,
    count,
    heads: tl.constexpr,
    half: tl.constexpr,
    even_row: tl.constexpr,
    even_head: tl.constexpr,
    even_col: tl.constexpr,
    odd_row: tl.constexpr,
    odd_head: tl.constexpr,
    odd_col: tl.constexpr,
    cos_row: tl.constexpr,
    cos_col: tl.constexpr,
    sin_row: tl.constexpr,
    sin_col: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    live = i < count
    col = i % (2 * half)
    head = (i // (2 * half)) % heads
    row = i // (heads * 2 * half)
    lane = col % half
    a = tl.load(even + row * even_row + head * even_head + lane * even_col, live & (col < half), 0)
    b = tl.load(odd + row * odd_row + head * odd_head + lane * odd_col, live & (col >= half), 0)
    tl.store(packed + i, tl.where(col < half, a, b), live)
    c = tl.load(cos + row * cos_row + lane * cos_col, live & (head == 0), 0)
    s = tl.load(sin + row * sin_row + lane * sin_col, live & (head == 0), 0)
    tl.store(packed_cos + row * (2 * half) + col, c, live & (head == 0))
    tl.store(packed_sin + row * (2 * half) + col, s, live & (head == 0))


def pack_rope_inputs(even, odd, cos, sin):
    """Return the same three FP32 concatenations, or None for fallback."""
    if (
        even.device.type != "npu"
        or even.ndim not in (2, 3)
        or odd.shape != even.shape
        or not even.shape[-1]
        or any(t.dtype != torch.float32 or t.device != even.device for t in (even, odd, cos, sin))
    ):
        return None
    rows, half = even.shape[0], even.shape[-1]
    heads = even.shape[1] if even.ndim == 3 else 1
    if not heads:
        return None
    factor_shape = (rows, 1, half) if even.ndim == 3 else (rows, half)
    if cos.shape != factor_shape or sin.shape != factor_shape:
        return None
    packed = torch.empty((*even.shape[:-1], 2 * half), dtype=even.dtype, device=even.device)
    packed_cos = torch.empty((rows, 1, 1, 2 * half), dtype=cos.dtype, device=cos.device)
    packed_sin = torch.empty_like(packed_cos)
    if packed.numel():
        _pack_rope_inputs[(triton.cdiv(packed.numel(), 256),)](
            even,
            odd,
            cos,
            sin,
            packed,
            packed_cos,
            packed_sin,
            packed.numel(),
            heads,
            half,
            even.stride(0),
            even.stride(1) if even.ndim == 3 else 0,
            even.stride(-1),
            odd.stride(0),
            odd.stride(1) if odd.ndim == 3 else 0,
            odd.stride(-1),
            cos.stride(0),
            cos.stride(-1),
            sin.stride(0),
            sin.stride(-1),
            256,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return packed, packed_cos, packed_sin
