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
"""Avoid rescheduling PP+MTP decode requests with unresolved async outputs."""

from __future__ import annotations

from collections import Counter
from functools import wraps
from typing import Iterable

from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

_PATCHED = False
_INFLIGHT_ATTR = "_vllm_ascend_pp_mtp_inflight_reqs"
_ENABLED_LOG_ATTR = "_vllm_ascend_pp_mtp_inflight_logged"


def _get_inflight_counter(scheduler: Scheduler) -> Counter[str]:
    counter = getattr(scheduler, _INFLIGHT_ATTR, None)
    if counter is None:
        counter = Counter()
        setattr(scheduler, _INFLIGHT_ATTR, counter)
    return counter


def _is_mtp_spec_decode(scheduler: Scheduler) -> bool:
    speculative_config = getattr(scheduler.vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False

    method = getattr(speculative_config, "method", None)
    if method == "mtp":
        return True

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    if not isinstance(model_type, str):
        return False
    return model_type == "mtp" or model_type.endswith("_mtp")


def _guard_enabled(scheduler: Scheduler) -> bool:
    additional_config = getattr(scheduler.vllm_config, "additional_config", None) or {}
    explicit = additional_config.get("enable_pp_mtp_inflight_guard")
    if explicit is not None and not bool(explicit):
        return False

    parallel_config = scheduler.vllm_config.parallel_config
    scheduler_config = scheduler.vllm_config.scheduler_config
    can_overlap_batches = (
        getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        or bool(getattr(scheduler_config, "async_scheduling", False))
    )
    enabled = _is_mtp_spec_decode(scheduler) and can_overlap_batches
    if explicit is not None:
        enabled = bool(explicit) and enabled

    if enabled and not getattr(scheduler, _ENABLED_LOG_ATTR, False):
        logger.info(
            "PP+MTP in-flight scheduler guard is enabled. "
            "Set additional_config.enable_pp_mtp_inflight_guard=false to disable it."
        )
        setattr(scheduler, _ENABLED_LOG_ATTR, True)
    return enabled


def _is_live_request(scheduler: Scheduler, request: Request) -> bool:
    return request.request_id in scheduler.requests and not request.is_finished()


def _should_defer_request(scheduler: Scheduler, request: Request, inflight: Counter[str]) -> bool:
    if inflight.get(request.request_id, 0) <= 0:
        return False
    if not _is_live_request(scheduler, request):
        return False
    # Prefill chunks do not depend on the previous sampled token, and are the
    # primary source of work used to fill PP bubbles. Only guard decode/spec
    # steps whose next input depends on the previous output being accepted.
    return request.num_computed_tokens >= request.num_prompt_tokens


def _restore_running_requests(
    scheduler: Scheduler,
    original_running: list[Request],
    runnable_running: list[Request],
    deferred_running: list[Request],
    current_running: list[Request],
) -> list[Request]:
    current_by_obj = {id(req): req for req in current_running}
    runnable_ids = {id(req) for req in runnable_running}
    deferred_ids = {id(req) for req in deferred_running}

    restored: list[Request] = []
    restored_ids: set[int] = set()
    for request in original_running:
        obj_id = id(request)
        if obj_id in deferred_ids:
            if _is_live_request(scheduler, request):
                restored.append(request)
                restored_ids.add(obj_id)
            continue
        current_request = current_by_obj.get(obj_id)
        if current_request is not None and _is_live_request(scheduler, current_request):
            restored.append(current_request)
            restored_ids.add(obj_id)

    for request in current_running:
        obj_id = id(request)
        if obj_id in restored_ids or obj_id in runnable_ids:
            continue
        if _is_live_request(scheduler, request):
            restored.append(request)

    return restored


def _iter_decode_req_ids(scheduler: Scheduler, scheduler_output: SchedulerOutput) -> Iterable[str]:
    for req_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished():
            continue
        if not request.is_prefill_chunk:
            yield req_id


def _patch_schedule(scheduler_cls: type[Scheduler]) -> None:
    original_schedule = scheduler_cls.schedule
    if getattr(original_schedule, "_vllm_ascend_pp_mtp_scheduler_patched", False):
        return

    @wraps(original_schedule)
    def _patched_schedule(self: Scheduler) -> SchedulerOutput:
        if not _guard_enabled(self):
            return original_schedule(self)

        inflight = _get_inflight_counter(self)
        if not inflight:
            return original_schedule(self)

        runnable_running: list[Request] = []
        deferred_running: list[Request] = []
        for request in self.running:
            if _should_defer_request(self, request, inflight):
                deferred_running.append(request)
            else:
                runnable_running.append(request)

        if not deferred_running:
            return original_schedule(self)

        original_running = self.running
        original_max_running = self.max_num_running_reqs
        self.running = runnable_running
        self.max_num_running_reqs = max(0, original_max_running - len(deferred_running))
        try:
            scheduler_output = original_schedule(self)
            current_running = self.running
        finally:
            self.max_num_running_reqs = original_max_running
        self.running = _restore_running_requests(
            self,
            original_running,
            runnable_running,
            deferred_running,
            current_running,
        )
        return scheduler_output

    _patched_schedule._vllm_ascend_pp_mtp_scheduler_patched = True  # type: ignore[attr-defined]
    scheduler_cls.schedule = _patched_schedule


def _patch_scheduler() -> None:
    original_update_after_schedule = Scheduler._update_after_schedule
    original_update_from_output = Scheduler.update_from_output

    @wraps(original_update_after_schedule)
    def _patched_update_after_schedule(self: Scheduler, scheduler_output: SchedulerOutput) -> None:
        original_update_after_schedule(self, scheduler_output)
        if not _guard_enabled(self):
            return
        inflight = _get_inflight_counter(self)
        for req_id in _iter_decode_req_ids(self, scheduler_output):
            inflight[req_id] += 1

    @wraps(original_update_from_output)
    def _patched_update_from_output(
        self: Scheduler,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ):
        scheduled_req_ids = tuple(scheduler_output.num_scheduled_tokens)
        try:
            return original_update_from_output(self, scheduler_output, model_runner_output)
        finally:
            if _guard_enabled(self):
                inflight = _get_inflight_counter(self)
                for req_id in scheduled_req_ids:
                    if req_id not in inflight:
                        continue
                    if inflight.get(req_id, 0) <= 1:
                        inflight.pop(req_id, None)
                    else:
                        inflight[req_id] -= 1

    _patch_schedule(Scheduler)
    try:
        from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
    except ImportError:
        pass
    else:
        _patch_schedule(BalanceScheduler)

    Scheduler._update_after_schedule = _patched_update_after_schedule
    Scheduler.update_from_output = _patched_update_from_output


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_scheduler()


_apply_patch()
