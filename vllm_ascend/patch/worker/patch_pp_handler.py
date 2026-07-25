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

import numpy as np
import torch
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler, PendingRecv, compute_need_sampled_mask


def _pad_sampled_tokens(self: PPHandler, sampled_token_ids: torch.Tensor, num_reqs: int) -> torch.Tensor:
    if sampled_token_ids.dim() == 1:
        sampled_token_ids = sampled_token_ids.view(-1, 1)
    assert sampled_token_ids.shape[0] == num_reqs
    sampled_tokens = sampled_token_ids.new_zeros(
        num_reqs,
        self.max_sample_len,
    )
    copy_len = min(sampled_token_ids.shape[1], self.max_sample_len)
    sampled_tokens[:, :copy_len].copy_(sampled_token_ids[:, :copy_len])
    return sampled_tokens.contiguous()


def _receive(self: PPHandler, input_batch: InputBatch) -> bool:
    """Always join the PP sampled-token broadcast for stream safety.

    EAGLE/rejection sampling can make local request state differ temporarily
    across PP ranks. If one rank skips the sampled-token broadcast while another
    joins it, HCCL reports an asynchronous broadcast error later. Always
    participate in the fixed-shape collective and only enqueue results when the
    local rank actually needs deferred postprocess data.
    """
    assert not self.is_last_rank
    need_sampled_mask = compute_need_sampled_mask(input_batch)
    gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

    num_reqs = input_batch.num_reqs
    with torch.cuda.stream(self.broadcast_stream):
        self.broadcast_stream.wait_stream(self.main_stream)
        sampled_tokens = torch.empty(
            num_reqs,
            self.max_sample_len,
            dtype=torch.int64,
            device=self.device,
        )
        combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            sampled_tokens,
            src=self.last_rank,
            group=self.broadcast_group,
        )
        torch.distributed.broadcast(
            combined,
            src=self.last_rank,
            group=self.broadcast_group,
        )
        event = self.broadcast_stream.record_event()
        num_sampled, num_rejected = combined.unbind(dim=0)
        sampled_tokens.record_stream(self.main_stream)
        combined.record_stream(self.main_stream)

    if need_sampled_mask is None:
        # The collective was only needed to keep PP ranks in lockstep.
        return False

    self.queue[-1] = PendingRecv(
        event,
        sampled_tokens,
        num_sampled,
        num_rejected,
        input_batch.idx_mapping,
        input_batch.idx_mapping_np,
        need_sampled_mask,
        gen_at_receive_np,
    )
    return bool(need_sampled_mask.all())


def _get_prev_sampled_outputs(self: PPHandler) -> dict[str, torch.Tensor] | None:
    if not self.queue:
        return None
    slot = self.queue.popleft()
    # Reserve this step's slot; `receive` overwrites it if applicable.
    self.queue.append(None)
    if slot is None:
        return None

    # Skip requests which did not need sampled output and/or those already
    # finished. The post_update kernel skips the -1 entries.
    freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
    exclude_mask = freed | ~slot.need_sampled_mask
    idx_mapping = slot.idx_mapping
    if exclude_mask.any():
        if exclude_mask.all():
            return None
        idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

    self.main_stream.wait_event(slot.event)
    return dict(
        sampled_tokens=slot.sampled_tokens,
        num_sampled=slot.num_sampled,
        num_rejected=slot.num_rejected,
        idx_mapping=idx_mapping,
    )


def _broadcast(
    self: PPHandler,
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    input_batch: InputBatch,
) -> None:
    assert self.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    num_reqs = input_batch.num_reqs
    with torch.cuda.stream(self.broadcast_stream):
        self.broadcast_stream.wait_stream(self.main_stream)
        sampled_tokens = _pad_sampled_tokens(self, sampled_token_ids, num_reqs)
        torch.distributed.broadcast(
            sampled_tokens,
            src=self.last_rank,
            group=self.broadcast_group,
        )
        combined = torch.stack((num_sampled, num_rejected), dim=0)
        torch.distributed.broadcast(
            combined,
            src=self.last_rank,
            group=self.broadcast_group,
        )
        for tensor in (
            sampled_tokens,
            sampled_token_ids,
            num_sampled,
            num_rejected,
            combined,
        ):
            tensor.record_stream(self.broadcast_stream)


PPHandler.receive = _receive
PPHandler.get_prev_sampled_outputs = _get_prev_sampled_outputs
PPHandler.broadcast = _broadcast
