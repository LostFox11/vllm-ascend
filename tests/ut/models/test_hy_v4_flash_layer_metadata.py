# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer ownership regressions; no NPU service is needed for these cases."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention, engram, ihc, model, moe


def metadata(prefill=False, starts=(0, 2)):
    return attention.FlashAttentionMetadata(
        starts,
        (starts[-1],),
        torch.empty(0),
        torch.empty(0),
        precise_decode_reqs=(not prefill,),
        prefill_reqs=(prefill,),
    )


@pytest.fixture
def layer_context(monkeypatch):
    own = metadata()
    other = metadata(prefill=True, starts=(0, 7))
    context = SimpleNamespace(attn_metadata={"other.attn": other, "layers.8.attn": own})
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention, "is_forward_context_available", lambda: True)
    return context, own, other


@pytest.mark.parametrize("first", ["other", "unrelated", "own"])
def test_explicit_metadata_ignores_dictionary_order(layer_context, first):
    context, own, other = layer_context
    entries = {"other.attn": other, "unrelated.attn": SimpleNamespace(), "layers.8.attn": own}
    first_key = {"other": "other.attn", "unrelated": "unrelated.attn", "own": "layers.8.attn"}[first]
    context.attn_metadata = {first_key: entries.pop(first_key), **entries}
    assert attention.get_flash_attention_metadata("layers.8.attn") is own
    assert attention.get_flash_attention_metadata("absent.attn") is None
    assert attention.get_flash_attention_metadata("unrelated.attn") is None


def test_native_decode_dispatch_uses_own_layer(layer_context):
    layer = attention.AscendHYV4FlashAttention.__new__(attention.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "layers.8.attn"
    layer.native_decode_attention = True
    layer._forward_native_decode = Mock(return_value="decode")
    layer._forward_existing = Mock(side_effect=AssertionError("wrong layer selected Prefill"))
    assert layer(torch.empty(2), torch.empty(2, 4)) == "decode"


def test_native_context_receives_own_cache_group_metadata(layer_context, monkeypatch):
    _, own, _ = layer_context
    layer = attention.AscendHYV4FlashAttention.__new__(attention.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "layers.8.attn"
    layer.native_decode_attention = True
    layer.learnable_sink_param = torch.empty(1)
    layer.scale = 1.0
    layer._fia_length_provider = lambda *args: "provider"
    result = torch.ones(2, 4)
    native = Mock(return_value=result)
    monkeypatch.setattr(attention, "native_mla_context", native)
    monkeypatch.setattr(attention, "_head_bmm", lambda context, weights: context)
    actual = layer._attend_decoded(torch.empty(2, 1, 4), torch.zeros(2, 3), torch.empty(0), None)
    assert actual is result
    assert native.call_args.kwargs["metadata"] is own
    assert native.call_args.kwargs["length_provider"] == "provider"


def test_indexpool_encoding_selects_own_layer(layer_context, monkeypatch):
    from vllm_ascend.models.hy_v4_flash import indexer

    pool = attention.FlashIndexPool.__new__(attention.FlashIndexPool)
    torch.nn.Module.__init__(pool)
    pool.metadata_layer_name = "layers.8.attn"
    encode = Mock(return_value="decode-fp8")
    monkeypatch.setattr(indexer, "encode_indexpool_fp8_fused", encode)
    vectors = torch.zeros(2, 128)
    assert pool._encode_fp8(vectors, full_graph=True) == "decode-fp8"
    encode.assert_called_once_with(vectors)


def test_engram_rms_receives_own_layer(layer_context, monkeypatch):
    _, _, other = layer_context
    value, weight = torch.ones(2, 4), torch.ones(4)
    inverse = torch.full((2, 1), 3.0)
    op = Mock(return_value=(None, inverse))
    monkeypatch.setattr(engram.torch_npu, "npu_rms_norm", op)
    assert engram._engram_inverse_rms(value, weight, 1e-6, False, layer_name="layers.8.attn") is inverse
    op.assert_called_once()
    op.reset_mock()
    actual = engram._engram_inverse_rms(value, weight, 1e-6, False, layer_name="other.attn")
    torch.testing.assert_close(actual, (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt())
    assert other.prefill_reqs == (True,)
    op.assert_not_called()


def test_ihc_precise_reduction_uses_own_partition(layer_context, monkeypatch):
    lookup = Mock(wraps=ihc.get_flash_attention_metadata)
    monkeypatch.setattr(ihc, "get_flash_attention_metadata", lookup)
    values = torch.arange(8).reshape(2, 4).float()
    actual = ihc._inverse_rms(values, 1e-6, layer_name="layers.8.attn")
    expected = (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    lookup.assert_called_once_with("layers.8.attn")


def test_moe_prefill_does_not_follow_other_decode_layer(layer_context, monkeypatch):
    context, own, other = layer_context
    context.attn_metadata = {"other.attn": own, "layers.8.attn": other}
    layer = moe.AscendFlashMoE.__new__(moe.AscendFlashMoE)
    torch.nn.Module.__init__(layer)
    layer.metadata_layer_name = "layers.8.attn"
    monkeypatch.setattr(moe.HYV4FlashMoE, "forward", lambda self, x: "prefill")
    assert layer(torch.empty(2, 4)) == "prefill"


def test_model_binds_each_module_to_owner_and_head_to_last_layer():
    layers = []
    for index in (0, 8):
        expert = moe.AscendFlashMoE.__new__(moe.AscendFlashMoE)
        torch.nn.Module.__init__(expert)
        expert.gate = torch.nn.Identity()
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(layer_name=f"model.layers.{index}.self_attn.attn"),
                hc_attn_layer=SimpleNamespace(enable_ihc=True, hc_pre=SimpleNamespace()),
                hc_mlp_layer=SimpleNamespace(enable_ihc=False),
                mlp=expert,
                engram_layer=SimpleNamespace(),
            )
        )
    head = ihc.FlashHCHeadLayer.__new__(ihc.FlashHCHeadLayer)
    torch.nn.Module.__init__(head)
    target = SimpleNamespace(layers=layers, hc_head=head)
    model.AscendHYV4FlashModel._bind_layer_metadata(target)
    for layer in layers:
        expected = layer.self_attn.layer_name
        assert layer.hc_attn_layer.metadata_layer_name == expected
        assert layer.hc_attn_layer.hc_pre.metadata_layer_name == expected
        assert layer.hc_mlp_layer.metadata_layer_name == expected
        assert layer.mlp.metadata_layer_name == layer.mlp.gate.metadata_layer_name == expected
        assert layer.engram_layer.metadata_layer_name == expected
    assert head.metadata_layer_name == layers[-1].self_attn.layer_name
