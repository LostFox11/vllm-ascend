# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for packed C8 and layout-only fusion fallbacks."""

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention
from vllm_ascend.models.hy_v4_flash import cache as cache_ops
from vllm_ascend.models.hy_v4_flash.attention_ops import native_mla_context
from vllm_ascend.models.hy_v4_flash.fia_lengths import CompactFIAKV, PackedC8KV
from vllm_ascend.models.hy_v4_flash.rope_layout import pack_rope_inputs


@pytest.mark.parametrize("layout", ["dense", "compact", "packed"])
def test_native_fia_adapter_accepts_shared_compaction_layout(layout):
    from vllm_ascend.models.hy_v4_flash.attention_ops import _fia_kv_inputs
    from vllm_ascend.models.hy_v4_flash.cache import CompactKV, compact_kv_rows

    selected = torch.tensor([[-1, 3, -1, 7]])
    values = torch.arange(4 * 576).reshape(1, 4, 576).float()
    query = torch.zeros(1, 2, 576, dtype=torch.bfloat16)
    nope, rope = compact_kv_rows(selected, values[:, :, None, :512], values[:, :, None, 512:])
    mask = torch.tensor([[[[False, False, True, True]]]])
    if layout == "compact":
        rows = CompactKV(nope, rope, mask)
    elif layout == "packed":
        rows = PackedC8KV(torch.cat((nope[:, :, 0], rope[:, :, 0]), -1).bfloat16(), mask)
    else:
        rows = values
    actual_nope, actual_rope = _fia_kv_inputs(query, selected, rows)
    torch.testing.assert_close(actual_nope, nope.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(actual_rope, rope.bfloat16(), rtol=0, atol=0)
    assert actual_nope.is_contiguous() and actual_rope.is_contiguous()


def test_restored_native_attention_dispatches_to_cann_with_host_lengths(monkeypatch):
    from types import SimpleNamespace

    from vllm_ascend.models.hy_v4_flash import attention_ops
    from vllm_ascend.models.hy_v4_flash.fia_lengths import FIALengthProvider

    tensor = torch.zeros(1, 2, 576, dtype=torch.bfloat16)

    class NativeQuery:
        device = SimpleNamespace(type="npu")
        dtype = tensor.dtype
        shape = tensor.shape

        def __getitem__(self, index):
            return tensor[index]

    selected = torch.tensor([[-1, 0, 1, 2]])
    kv = torch.zeros(1, 4, 576, dtype=torch.float32)
    metadata = SimpleNamespace(seq_lens=[3], query_start_loc=[0, 1])
    provider = FIALengthProvider("layer", 0, 1, 4, 512)
    calls = []

    def native_op(**kwargs):
        calls.append(kwargs)
        return torch.ones(1, 1, 2, 512, dtype=torch.bfloat16), torch.zeros(1, 2, 1, 1)

    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(attention_ops.torch_npu, "npu_fused_infer_attention_score_v2", native_op)
    monkeypatch.setattr(attention_ops, "fused_sink_context", lambda context, *args: context)
    result = native_mla_context(
        NativeQuery(), selected, kv, torch.zeros(2), 1.0, length_provider=provider, metadata=metadata
    )
    assert result.shape == (1, 2, 512)
    assert len(calls) == 1
    assert calls[0]["actual_seq_kvlen"] == [3]
    assert calls[0]["key"].dtype == torch.bfloat16
    assert calls[0]["return_softmax_lse"] is True


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
    assert allocate is (cache_ops.empty_packed_c8_rows if enabled else cache_ops.empty_compact_c8_rows)
    assert read is (cache_ops.gather_compact_c8_packed_rows if enabled else cache_ops.gather_compact_c8_rows)


def test_packed_c8_rejects_invalid_output_without_launching():
    cache = torch.empty(1, 656, dtype=torch.uint8)
    table = torch.zeros(1, dtype=torch.int32)
    selected = torch.zeros(1, 8, dtype=torch.int32)
    output = (torch.empty(1, 8, 576, dtype=torch.float64), torch.empty(1, 1, 1, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="contiguous BF16 KV"):
        cache_ops.gather_compact_c8_packed_rows(cache, table, 64, selected, output=output)


def test_rope_cpu_inputs_select_original_cat_path():
    value = torch.zeros(6, 16, 64)
    phase = torch.zeros(6, 1, 32)
    assert pack_rope_inputs(value[..., ::2], value[..., 1::2], phase, phase) is None
