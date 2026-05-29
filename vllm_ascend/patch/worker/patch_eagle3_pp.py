#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch: Eagle3 draft model pipeline parallelism support.
# On non-last PP ranks the draft model is replaced with a minimal stub;
# full weights are only loaded on the last PP stage.
#

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
    LlamaModel,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Stub helpers for non-last PP ranks
# ---------------------------------------------------------------------------
def _make_stub_empty_tensors(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    return IntermediateTensors({})


def _stub_combine_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states


# ---------------------------------------------------------------------------
# Save originals
# ---------------------------------------------------------------------------
_original_eagle3_init = Eagle3LlamaForCausalLM.__init__
_original_eagle3_forward = Eagle3LlamaForCausalLM.forward


# ---------------------------------------------------------------------------
# Patched Eagle3LlamaForCausalLM.__init__
# ---------------------------------------------------------------------------
def _patched_eagle3_init(self, *, vllm_config, prefix: str = ""):
    pp_group = get_pp_group()

    if pp_group.world_size > 1 and not pp_group.is_last_rank:
        # Non-last PP rank: create a minimal stub to avoid loading
        # unused weights and to satisfy PP interface checks.
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        self.model = PPMissingLayer()
        self.lm_head = PPMissingLayer()
        self.logits_processor = None
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )
        self.use_parallel_drafting = (
            vllm_config.speculative_config.parallel_drafting
        )
        self.make_empty_intermediate_tensors = _make_stub_empty_tensors
        self.combine_hidden_states = _stub_combine_hidden_states
        return

    # Last PP rank or PP=1: full initialization.
    # The original init uses get_num_layers() which returns the LOCAL PP rank
    # layer count. Under PP>1 this causes the draft model's start_layer_id to
    # overlap with the target model's layer indices on rank > 0 (duplicate
    # layer prefix). Work around this by temporarily forcing get_num_layers
    # to return the total number of layers.
    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    _original_get_num_layers = vllm_config.model_config.get_num_layers
    vllm_config.model_config.get_num_layers = (
        lambda parallel_config: total_layers
    )
    try:
        _original_eagle3_init(self, vllm_config=vllm_config, prefix=prefix)
    finally:
        vllm_config.model_config.get_num_layers = _original_get_num_layers

    self.make_empty_intermediate_tensors = (
        self.model.make_empty_intermediate_tensors
    )


# ---------------------------------------------------------------------------
# Patched Eagle3LlamaForCausalLM.forward
# ---------------------------------------------------------------------------
def _patched_eagle3_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return self.model(input_ids, positions, hidden_states, inputs_embeds)


# ---------------------------------------------------------------------------
# Add make_empty_intermediate_tensors to the inner LlamaModel (eagle3 variant)
# ---------------------------------------------------------------------------
def _make_empty_intermediate_tensors(
    self,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    return IntermediateTensors({
        "hidden_states": torch.zeros(
            (batch_size, self.config.hidden_size), dtype=dtype, device=device
        ),
        "residual": torch.zeros(
            (batch_size, self.config.hidden_size), dtype=dtype, device=device
        ),
    })


# ---------------------------------------------------------------------------
# Patch AscendEagleProposer.load_model to skip non-last PP ranks
# ---------------------------------------------------------------------------
def _patch_proposer_load_model():
    """Wrap AscendEagleProposer.load_model so that on non-last PP ranks only
    the draft model stub is created and all attention-layer discovery /
    kernel setup is skipped."""
    try:
        from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
    except ImportError:
        return

    _original_load_model = AscendEagleProposer.load_model

    def _patched_load_model(self, model: nn.Module) -> None:
        pp_group = get_pp_group()

        if pp_group.world_size > 1 and not pp_group.is_last_rank:
            # Non-last PP rank: only create the draft model stub, skip
            # attention-layer discovery, kernel config, and weight sharing.
            with self.maybe_eager_context:
                self.model = self._get_model()
            self._draft_attn_layer_names = set()
            self.attn_layer_names = []
            self.piece_all_attn_layer_name = []
            # 不 return——下面的 broadcast 需要所有 rank 参与
        else:
            # Last PP rank or PP=1: full load_model.
            _original_load_model(self, model)

        # [FIX] PP > 1: send embed_tokens from rank 0 to last rank (per TP shard).
        # PP 下 _maybe_share_embeddings 被跳过, 但 Eagle3 checkpoint 通常不带
        # 独立 embed_tokens 权重, drafter 的 embed_tokens 随机初始化.
        # autoregressive 的 2nd/3rd draft token 用随机 embedding → 低质量预测.
        # 用 send/recv 而不是 broadcast, 因为每个 TP rank 的 shard 不同,
        # broadcast 会把同一个 shard 发给所有 rank 导致 TP shard 错乱.
        if pp_group.world_size > 1:
            # PP group has 2 members per TP rank: [PP0_TPk, PP1_TPk]
            # pp_group.ranks[0] = PP0's global rank for this TP rank
            # pp_group.ranks[1] = PP1's global rank for this TP rank
            if pp_group.is_first_rank:
                # PP0: send my embed_tokens shard to PP1
                if (hasattr(model.model, "embed_tokens")
                        and hasattr(model.model.embed_tokens, "weight")):
                    torch.distributed.send(
                        model.model.embed_tokens.weight.data.contiguous(),
                        dst=pp_group.ranks[1],
                        group=pp_group.device_group,
                    )
            elif pp_group.is_last_rank:
                # PP1: receive my shard from PP0
                if (hasattr(self, "model") and hasattr(self.model, "model")
                        and hasattr(self.model.model, "embed_tokens")
                        and hasattr(self.model.model.embed_tokens, "weight")):
                    torch.distributed.recv(
                        self.model.model.embed_tokens.weight.data,
                        src=pp_group.ranks[0],
                        group=pp_group.device_group,
                    )
                    logger.info("PP recv embed_tokens for drafter (shape=%s)",
                                list(self.model.model.embed_tokens.weight.data.shape))

    AscendEagleProposer.load_model = _patched_load_model


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
Eagle3LlamaForCausalLM.__init__ = _patched_eagle3_init
Eagle3LlamaForCausalLM.forward = _patched_eagle3_forward
Eagle3LlamaForCausalLM.supports_pp = True
LlamaModel.make_empty_intermediate_tensors = _make_empty_intermediate_tensors
_patch_proposer_load_model()
