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

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler, compute_need_sampled_mask


def _broadcast(
    self: PPHandler,
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    input_batch: InputBatch,
) -> None:
    assert self.is_last_rank
    if compute_need_sampled_mask(input_batch) is None:
        # No request needs sampled outputs for a subsequent decode step.
        return

    assert sampled_token_ids.dtype == torch.int64
    with torch.cuda.stream(self.broadcast_stream):
        self.broadcast_stream.wait_stream(self.main_stream)
        sampled_tokens = sampled_token_ids.contiguous()
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


PPHandler.broadcast = _broadcast
