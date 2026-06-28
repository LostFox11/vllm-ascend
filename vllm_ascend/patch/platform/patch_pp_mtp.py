#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Backport vLLM PP + MTP runtime support.

The local Eagle/MTP drafter returns the draft tokens that belong to the model
output being processed. With PP batch_queue, EngineCore schedules a newer batch
before consuming the older output, so updating ``request.spec_token_ids`` from
``post_step`` observes live Request state from the newer schedule step.
"""

from __future__ import annotations

import copy
from functools import wraps

from vllm.logger import logger

_PATCHED = False


def _patch_model_runner_output() -> None:
    from vllm.v1 import outputs as outputs_mod

    model_runner_output_cls = outputs_mod.ModelRunnerOutput
    fields = getattr(model_runner_output_cls, "__dataclass_fields__", {})
    if "spec_token_ids" not in fields:
        model_runner_output_cls.spec_token_ids = None
        original_init = model_runner_output_cls.__init__
        if getattr(original_init, "_vllm_ascend_pp_mtp_patched", False):
            return

        @wraps(original_init)
        def _patched_init(self, *args, spec_token_ids=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.spec_token_ids = spec_token_ids

        _patched_init._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
        model_runner_output_cls.__init__ = _patched_init

    empty_output = outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
    if not hasattr(empty_output, "spec_token_ids"):
        empty_output.spec_token_ids = None


def _patch_engine_core() -> None:
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.post_step, "_vllm_ascend_pp_mtp_patched", False):
        return

    original_post_step = EngineCore.post_step

    @wraps(original_post_step)
    def _patched_post_step(self, model_executed: bool) -> None:
        if (
            getattr(self, "batch_queue", None) is not None
            and not getattr(self, "async_scheduling", False)
            and getattr(self, "use_spec_decode", False)
            and model_executed
        ):
            return
        return original_post_step(self, model_executed)

    _patched_post_step._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    EngineCore.post_step = _patched_post_step


def _patch_scheduler_update_from_output() -> None:
    """Patch Scheduler.update_from_output to consume spec_token_ids from
    ModelRunnerOutput. When max_concurrent_batches > 1 in sync PP,
    post_step is skipped (to avoid stale request state from a newer
    batch).  Instead, draft tokens are carried on ModelRunnerOutput and
    consumed here.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.update_from_output
    if getattr(original, "_vllm_ascend_pp_mtp_uof_patched", False):
        return

    @wraps(original)
    def _patched_update_from_output(
        self,
        scheduler_output,
        model_runner_output,
    ):
        # PP+MTP can produce 0-token entries in num_scheduled_tokens
        # (pipeline bubbles).  The base Scheduler.update_from_output
        # asserts num > 0 for every entry.  Temporarily hide 0-token
        # entries from the original, then restore them afterwards.
        num_sched = scheduler_output.num_scheduled_tokens
        zero_req_ids = [r for r, n in num_sched.items() if n == 0]
        backup = {r: num_sched[r] for r in zero_req_ids}
        for r in zero_req_ids:
            del num_sched[r]
        try:
            result = original(self, scheduler_output, model_runner_output)
        finally:
            num_sched.update(backup)

        output_spec_token_ids = getattr(
            model_runner_output, "spec_token_ids", None
        )
        if not output_spec_token_ids:
            return result

        sampled_token_ids = getattr(
            model_runner_output, "sampled_token_ids", None
        )
        for req_id in num_sched:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = model_runner_output.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )
            if not generated:
                request.spec_token_ids = []
                continue
            next_spec_token_ids = output_spec_token_ids[req_index]
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                next_spec_token_ids = metadata.grammar.validate_tokens(
                    next_spec_token_ids
                )
            request.spec_token_ids = next_spec_token_ids

        return result

    _patched_update_from_output._vllm_ascend_pp_mtp_uof_patched = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = _patched_update_from_output


def _patch_model_config_validation() -> None:
    from typing import get_args

    from vllm.config.model import ModelConfig
    from vllm.config.speculative import MTPModelTypes

    original_verify = ModelConfig.verify_with_parallel_config
    if getattr(original_verify, "_vllm_ascend_pp_mtp_patched", False):
        return

    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
        model_type = getattr(hf_config, "model_type", None)
        is_eagle_drafter = (model_type == "eagle" or model_type == "speculators") and any(
            arch.startswith("Eagle") or arch.endswith("Eagle3") for arch in getattr(self, "architectures", ())
        )
        is_mtp_drafter = model_type in mtp_model_types
        if (
            getattr(self, "runner", None) == "draft"
            and (is_eagle_drafter or is_mtp_drafter)
            and getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            # Local Eagle/MTP drafters are loaded on the last PP stage rather
            # than partitioned across all PP stages. Keep normal target-model
            # validation intact, but validate these draft models as PP=1.
            logger.warning(
                "Validating local Eagle/MTP drafter with pipeline_parallel_size=1 "
                "because it is loaded locally on the last pipeline stage."
            )
            patched_config = copy.copy(parallel_config)
            patched_config.pipeline_parallel_size = 1
            return original_verify(self, patched_config)
        return original_verify(self, parallel_config)

    _patched_verify_with_parallel_config._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    ModelConfig.verify_with_parallel_config = _patched_verify_with_parallel_config


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_model_runner_output()
    _patch_engine_core()
    _patch_scheduler_update_from_output()
    _patch_model_config_validation()


_apply_patch()
