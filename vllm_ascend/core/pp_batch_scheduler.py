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

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request


def get_pp_batch_request_limit(num_requests: int, pp_size: int) -> int:
    """Return ceil(num_requests / pp_size)."""
    assert num_requests >= 0
    assert pp_size > 0
    return (num_requests + pp_size - 1) // pp_size


class PPBatchAsyncScheduler(AsyncScheduler):
    """Dynamically spread available requests across PP scheduler batches.

    The batch request limit is ``ceil(min(max_num_seqs, unfinished) / pp_size)``
    while the global RUNNING request limit remains ``max_num_seqs``. Before
    invoking the upstream scheduler, this class conservatively reserves
    admission slots based on how many RUNNING requests are already eligible in
    that PP cadence step.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pp_batch_request_limit = 0

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        effective_num_requests = min(
            self.max_num_running_reqs,
            self.get_num_unfinished_requests(),
        )
        batch_request_limit = get_pp_batch_request_limit(
            effective_num_requests,
            self.pp_size,
        )
        self.pp_batch_request_limit = batch_request_limit

        # Scheduler.schedule increments current_step before scanning RUNNING
        # requests. Count requests whose PP decode fence permits that next step.
        next_step = self.current_step + 1
        num_eligible_running = 0
        deferred_eligibility: list[tuple[Request, int]] = []
        for request in self.running:
            if request.next_decode_eligible_step > next_step:
                continue
            if num_eligible_running < batch_request_limit:
                num_eligible_running += 1
                continue

            # When the dynamic limit shrinks, an existing PP cohort can be
            # larger than the new limit. Defer its excess requests by one step
            # so the current SchedulerOutput still respects the hard cap.
            deferred_eligibility.append(
                (request, request.next_decode_eligible_step)
            )
            request.next_decode_eligible_step = next_step + 1

        remaining_batch_slots = max(
            0,
            batch_request_limit - num_eligible_running,
        )

        # The upstream WAITING loop admits requests until
        # len(running) reaches max_num_running_reqs. Temporarily lower that
        # ceiling so this step can admit at most the remaining batch slots.
        original_max_num_running_reqs = self.max_num_running_reqs
        self.max_num_running_reqs = min(
            original_max_num_running_reqs,
            len(self.running) + remaining_batch_slots,
        )
        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = original_max_num_running_reqs
            for request, next_decode_eligible_step in deferred_eligibility:
                request.next_decode_eligible_step = next_decode_eligible_step

        assert len(scheduler_output.num_scheduled_tokens) <= batch_request_limit
        return scheduler_output
