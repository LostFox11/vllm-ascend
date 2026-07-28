# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.worker.v2 import attn_utils


def test_get_kv_cache_spec_preserves_non_mla_attention_layers(monkeypatch):
    vllm_config = Mock()
    gdn_spec = Mock()
    gdn_layer = Mock()
    gdn_layer.kv_sharing_target_layer_name = None
    gdn_layer.get_kv_cache_spec.return_value = gdn_spec
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {
            "language_model.model.layers.0.linear_attn": gdn_layer
        },
    )

    specs = attn_utils.get_kv_cache_spec(vllm_config)

    assert specs == {
        "language_model.model.layers.0.linear_attn": gdn_spec,
    }


def test_mamba_kv_cache_allocation_and_reshape(monkeypatch):
    layer_name = "language_model.model.layers.0.linear_attn"
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2,), (3,)),
        dtypes=(torch.float32, torch.float32),
    )
    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mamba_spec.page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=mamba_spec,
            )
        ],
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kv_transfer_config=None),
    )

    raw_caches = attn_utils._allocate_kv_cache(
        kv_cache_config,
        shared_layers={},
        device=torch.device("cpu"),
    )
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=mamba_spec,
                layer_names=[layer_name],
            )
        ],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[mamba_spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )

    assert [state.shape for state in caches[layer_name]] == [
        (num_blocks, 2),
        (num_blocks, 3),
    ]
