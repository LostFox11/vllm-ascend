# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
    RecomputeReqInfo,
    RecomputeScheduler,
)


def test_add_request_does_not_inject_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == []
    assert request.num_tokens_with_spec == request.num_tokens


def test_recompute_notification_precedes_regular_output():
    scheduler_output = SimpleNamespace(
        recomputed_reqs=[
            RecomputeReqInfo(
                request_id="recomputed-request",
                output_token_ids=[],
                client_index=0,
            )
        ]
    )
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    RecomputeScheduler._add_recomputed_outputs(scheduler_output, outputs)
    outputs[0].append(
        EngineCoreOutput(
            request_id="regular-request",
            new_token_ids=[1],
        )
    )

    output = outputs[0][0]
    assert output.request_id == "recomputed-request"
    assert output.finish_reason == FinishReason.STOP
    assert output.stop_reason == "recomputed"
    assert outputs[0][1].request_id == "regular-request"


def test_finish_recomputed_request_uses_normal_abort_cleanup():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = Request(
        request_id="fallback-recomputed-request",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING

    # The fallback victim has already been popped from the running queue.
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = MagicMock()
    scheduler.skipped_waiting = MagicMock()
    scheduler._inflight_prefills = {request}
    scheduler._connector_finished = MagicMock(return_value=(False, None))
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.ec_connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_request_blocks = MagicMock()

    recomputed_reqs: list[RecomputeReqInfo] = []
    scheduler._finish_recomputed_request(request, recomputed_reqs)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request not in scheduler._inflight_prefills
    assert request.request_id not in scheduler.requests
    assert request.request_id in scheduler.finished_req_ids
    scheduler._connector_finished.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert recomputed_reqs == [
        RecomputeReqInfo(
            request_id=request.request_id,
            output_token_ids=request.output_token_ids,
            client_index=request.client_index,
        )
    ]


def test_async_recompute_scheduler_inherits_ppv2_scheduling() -> None:
    mro = AsyncRecomputeScheduler.__mro__
    assert mro.index(AsyncScheduler) < mro.index(RecomputeScheduler)
    assert AsyncRecomputeScheduler.schedule is RecomputeScheduler.schedule
    assert (
        AsyncRecomputeScheduler._update_after_schedule
        is AsyncScheduler._update_after_schedule
    )


def test_recompute_scheduler_applies_dynamic_pp_batch_limit() -> None:
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.current_step = 0
    scheduler.max_num_running_reqs = 25
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    scheduler.use_v2_model_runner = True
    scheduler.running = []
    scheduler.get_num_unfinished_requests = MagicMock(return_value=10)

    def fake_recompute_schedule(self, throttle_prefills=False):
        assert throttle_prefills
        assert self.max_num_running_reqs == 5
        return SimpleNamespace(
            num_scheduled_tokens={
                f"request-{index}": 1 for index in range(5)
            }
        )

    with patch.object(
        RecomputeScheduler,
        "_schedule",
        fake_recompute_schedule,
    ):
        output = scheduler.schedule(throttle_prefills=True)

    assert len(output.num_scheduled_tokens) == 5
    assert scheduler.max_num_running_reqs == 25


def test_recompute_scheduler_pp_batch_limit_rounds_up() -> None:
    get_limit = RecomputeScheduler._get_pp_batch_request_limit
    assert get_limit(25, 2) == 13
    assert get_limit(10, 2) == 5
    assert get_limit(5, 2) == 3
    assert get_limit(25, 4) == 7
    assert get_limit(0, 4) == 0


def test_recompute_scheduler_defers_dynamic_batch_excess() -> None:
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.current_step = 4
    scheduler.max_num_running_reqs = 25
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    scheduler.use_v2_model_runner = True
    scheduler.running = [
        SimpleNamespace(next_decode_eligible_step=0) for _ in range(12)
    ]
    scheduler.get_num_unfinished_requests = MagicMock(return_value=17)

    def fake_recompute_schedule(self, throttle_prefills=False):
        assert [req.next_decode_eligible_step for req in self.running] == (
            [0] * 9 + [6] * 3
        )
        return SimpleNamespace(
            num_scheduled_tokens={
                f"request-{index}": 1 for index in range(9)
            }
        )

    with patch.object(
        RecomputeScheduler,
        "_schedule",
        fake_recompute_schedule,
    ):
        scheduler.schedule()

    assert all(
        request.next_decode_eligible_step == 0
        for request in scheduler.running
    )
    assert scheduler.max_num_running_reqs == 25


def test_recompute_scheduler_bypasses_limit_without_pp() -> None:
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.max_num_running_reqs = 25
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    scheduler.use_v2_model_runner = True

    def fake_recompute_schedule(self, throttle_prefills=False):
        assert self.max_num_running_reqs == 25
        return SimpleNamespace(
            num_scheduled_tokens={
                f"request-{index}": 1 for index in range(10)
            }
        )

    with patch.object(
        RecomputeScheduler,
        "_schedule",
        fake_recompute_schedule,
    ):
        output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 10


def test_recompute_scheduler_bypasses_limit_for_model_runner_v1() -> None:
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.max_num_running_reqs = 25
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    scheduler.use_v2_model_runner = False

    def fake_recompute_schedule(self, throttle_prefills=False):
        assert self.max_num_running_reqs == 25
        return SimpleNamespace(
            num_scheduled_tokens={
                f"request-{index}": 1 for index in range(10)
            }
        )

    with patch.object(
        RecomputeScheduler,
        "_schedule",
        fake_recompute_schedule,
    ):
        output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 10
