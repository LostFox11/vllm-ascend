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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_ascend.core.pp_batch_scheduler import (
    PPBatchAsyncScheduler,
    get_pp_batch_request_limit,
)


@pytest.mark.parametrize(
    ("max_num_seqs", "pp_size", "expected"),
    [
        (25, 2, 13),
        (25, 4, 7),
        (32, 4, 8),
        (2, 4, 1),
        (0, 4, 0),
    ],
)
def test_get_pp_batch_request_limit(
    max_num_seqs: int,
    pp_size: int,
    expected: int,
) -> None:
    assert get_pp_batch_request_limit(max_num_seqs, pp_size) == expected


def _make_scheduler(
    *,
    current_step: int,
    running_eligible_steps: list[int],
    unfinished_requests: int,
) -> PPBatchAsyncScheduler:
    scheduler = object.__new__(PPBatchAsyncScheduler)
    scheduler.current_step = current_step
    scheduler.max_num_running_reqs = 25
    scheduler.pp_size = 2
    scheduler.pp_batch_request_limit = 0
    scheduler.running = [
        SimpleNamespace(next_decode_eligible_step=eligible_step)
        for eligible_step in running_eligible_steps
    ]
    scheduler.get_num_unfinished_requests = lambda: unfinished_requests
    return scheduler


@pytest.mark.parametrize(
    (
        "current_step",
        "running_eligible_steps",
        "unfinished_requests",
        "expected_batch_limit",
        "expected_temporary_limit",
    ),
    [
        (0, [], 50, 13, 13),
        (0, [], 10, 5, 5),
        (1, [3] * 5, 10, 5, 10),
        (1, [3] * 13, 50, 13, 25),
        (2, [3] * 13 + [4] * 12, 50, 13, 25),
    ],
)
def test_schedule_limits_waiting_admission(
    current_step: int,
    running_eligible_steps: list[int],
    unfinished_requests: int,
    expected_batch_limit: int,
    expected_temporary_limit: int,
) -> None:
    scheduler = _make_scheduler(
        current_step=current_step,
        running_eligible_steps=running_eligible_steps,
        unfinished_requests=unfinished_requests,
    )

    def fake_schedule(self, throttle_prefills=False):
        assert throttle_prefills
        assert self.max_num_running_reqs == expected_temporary_limit
        return SimpleNamespace(num_scheduled_tokens={})

    with patch.object(AsyncScheduler, "schedule", fake_schedule):
        scheduler.schedule(throttle_prefills=True)

    assert scheduler.max_num_running_reqs == 25
    assert scheduler.pp_batch_request_limit == expected_batch_limit


def test_schedule_rejects_oversized_batch() -> None:
    scheduler = _make_scheduler(
        current_step=0,
        running_eligible_steps=[],
        unfinished_requests=10,
    )

    def fake_schedule(self, throttle_prefills=False):
        return SimpleNamespace(
            num_scheduled_tokens={f"request-{index}": 1 for index in range(6)}
        )

    with (
        patch.object(AsyncScheduler, "schedule", fake_schedule),
        pytest.raises(AssertionError),
    ):
        scheduler.schedule()


def test_schedule_defers_running_requests_above_dynamic_limit() -> None:
    scheduler = _make_scheduler(
        current_step=4,
        running_eligible_steps=[0] * 12,
        unfinished_requests=17,
    )

    def fake_schedule(self, throttle_prefills=False):
        # ceil(17 / 2) is 9, so the last three requests are fenced for
        # Scheduler step 5 and become eligible again in step 6.
        assert [req.next_decode_eligible_step for req in self.running] == (
            [0] * 9 + [6] * 3
        )
        return SimpleNamespace(
            num_scheduled_tokens={
                f"request-{index}": 1 for index in range(9)
            }
        )

    with patch.object(AsyncScheduler, "schedule", fake_schedule):
        scheduler.schedule()

    assert scheduler.pp_batch_request_limit == 9
    assert all(
        request.next_decode_eligible_step == 0
        for request in scheduler.running
    )
