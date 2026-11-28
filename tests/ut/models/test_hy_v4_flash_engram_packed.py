# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection layout/reload contracts. NPU numerical and performance acceptance are separate."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash.engram import FlashEngram


def make_engram(tp=1, dtype=torch.bfloat16):
    layer = FlashEngram.__new__(FlashEngram)
    torch.nn.Module.__init__(layer)
    layer.hidden_size, layer.total_width, layer.hc_mult = 8, 16, 4
    layer.start, layer.stop, layer.rank = 0, 16, 0
    layer.multi_head_embedding = SimpleNamespace(tp_size=tp)
    layer.use_packed_decode_projection = True
    layer.register_buffer("_decode_kv_weight", None, persistent=False)
    layer.register_buffer("_decode_kv_bias", None, persistent=False)
    layer._decode_pack_constants = None
    layer.key_projs = torch.nn.ModuleList([torch.nn.Linear(16, 8, dtype=dtype) for _ in range(layer.hc_mult)])
    layer.value_proj = torch.nn.Linear(16, 8, dtype=dtype)
    with torch.no_grad():
        for channel, projection in enumerate((*layer.key_projs, layer.value_proj)):
            # Exact, bounded arithmetic isolates packing/layout from GEMM rounding.
            projection.weight.copy_(torch.arange(128).reshape(8, 16).remainder(3) - 1)
            projection.bias.fill_(channel)
    return layer


@pytest.mark.parametrize("rows", [1, 2, 6, 18, 192, 768])
def test_packed_projection_preserves_key_value_order_and_strided_views(rows):
    layer = make_engram()
    parameter_ids = [id(p) for p in layer.parameters()]
    layer.pack_decode_weights()
    x = torch.arange(rows * 16).reshape(rows, 16).remainder(3).bfloat16()
    parts = layer._project_decode_kv(x)
    assert len(parts) == layer.hc_mult + 1
    for actual, projection in zip(parts, (*layer.key_projs, layer.value_proj), strict=True):
        torch.testing.assert_close(actual, projection(x), rtol=0, atol=0)
        assert actual.stride() == ((layer.hc_mult + 1) * layer.hidden_size, 1)
    assert [id(p) for p in layer.parameters()] == parameter_ids
    assert not any("_decode_kv" in name for name in layer.state_dict())


@pytest.mark.parametrize(
    "tp,dtype,enabled", [(2, torch.bfloat16, True), (1, torch.float32, True), (1, torch.bfloat16, False)]
)
def test_unsupported_or_disabled_packing_preserves_fallback(tp, dtype, enabled):
    layer = make_engram(tp=tp, dtype=dtype)
    layer.use_packed_decode_projection = enabled
    layer.pack_decode_weights()
    assert layer._decode_kv_weight is None
    assert layer._project_decode_kv(torch.zeros(6, 16, dtype=dtype)) is None


@pytest.mark.parametrize("case", ["empty", "dtype", "strided", "disabled"])
def test_runtime_input_guards(case):
    layer = make_engram()
    layer.pack_decode_weights()
    x = torch.zeros(6, 16, dtype=torch.bfloat16)
    if case == "empty":
        x = x[:0]
    elif case == "dtype":
        x = x.float()
    elif case == "strided":
        x = torch.zeros(6, 32, dtype=torch.bfloat16)[:, ::2]
    else:
        layer.use_packed_decode_projection = False
    assert layer._project_decode_kv(x) is None


@pytest.mark.parametrize("reload_bias", [False, True])
def test_weight_reload_invalidates_pack_and_repacking_uses_new_values(reload_bias):
    layer = make_engram()
    layer.pack_decode_weights()
    layer._decode_pack_constants = object()
    projection = layer.key_projs[0]
    with torch.no_grad():
        if reload_bias:
            layer._load_bias(projection.bias, torch.full_like(projection.bias, 7))
        else:
            layer._load_projection(projection.weight, torch.zeros_like(projection.weight))
    assert layer._decode_kv_weight is None
    assert layer._decode_kv_bias is None
    assert layer._decode_pack_constants is None
    layer.pack_decode_weights()
    x = torch.ones(6, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(layer._project_decode_kv(x)[0], projection(x), rtol=0, atol=0)
