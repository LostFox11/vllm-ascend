# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for packed C8 and layout-only fusion fallbacks."""

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention, c8_read
from vllm_ascend.models.hy_v4_flash.decode_fusions import native_mla_context
from vllm_ascend.models.hy_v4_flash.fia_lengths import CompactFIAKV, PackedC8KV
from vllm_ascend.models.hy_v4_flash.rope_layout import pack_rope_inputs


@pytest.mark.parametrize("with_scores", [False, True])
@pytest.mark.parametrize("empty_row", [False, True])
def test_packed_c8_keeps_mask_and_precomputed_score_permutation(with_scores, empty_row):
    generator = torch.Generator().manual_seed(1024)
    selected = torch.tensor([[-1, 3, -1, 7]])
    if empty_row:
        selected.fill_(-1)
    values = torch.randn(1, 4, 576, generator=generator)
    mask = torch.tensor([[[[False, False, True, True]]]])
    if empty_row:
        mask.fill_(True)
    values.masked_fill_(mask.reshape(1, 4, 1), 0)
    old = CompactFIAKV(values[:, :, None, :512].contiguous(), values[:, :, None, 512:].contiguous(), mask)
    packed = PackedC8KV(values, mask)
    query = torch.randn(1, 2, 576, generator=generator)
    sink = torch.randn(2, generator=generator)
    scores = torch.randn(1, 2, 4, generator=generator) if with_scores else None
    kwargs = dict(length_provider=None, metadata=None, scores=scores)
    expected = native_mla_context(query, selected, old, sink, 576**-0.5, **kwargs)
    actual = native_mla_context(query, selected, packed, sink, 576**-0.5, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_reader_selection_keeps_original_fallback(enabled):
    layer = attention.AscendHYV4FlashAttention.__new__(attention.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.fuse_decode_c8_layout = enabled
    allocate, read = layer._decode_c8_reader()
    assert allocate is (c8_read.empty_packed_c8_rows if enabled else c8_read.empty_compact_c8_fia_rows)
    assert read is (c8_read.gather_compact_c8_packed_rows if enabled else c8_read.gather_compact_c8_fia_rows)


def test_packed_c8_rejects_invalid_output_without_launching():
    cache = torch.empty(1, 656, dtype=torch.uint8)
    table = torch.zeros(1, dtype=torch.int32)
    selected = torch.zeros(1, 8, dtype=torch.int32)
    output = (torch.empty(1, 8, 576, dtype=torch.float64), torch.empty(1, 1, 1, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="contiguous BF16 KV"):
        c8_read.gather_compact_c8_packed_rows(cache, table, 64, selected, output=output)


def test_rope_cpu_inputs_select_original_cat_path():
    value = torch.zeros(6, 16, 64)
    phase = torch.zeros(6, 1, 32)
    assert pack_rope_inputs(value[..., ::2], value[..., 1::2], phase, phase) is None
