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
# Patch: Mark DeepSeekMTP as PP-compatible so that config validation during
# API server startup (verify_with_parallel_config) passes before worker patches
# are loaded.
#

from __future__ import annotations

from functools import wraps

import torch
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.sequence import IntermediateTensors

DeepSeekMTP.supports_pp = True


def _mtp_make_empty_intermediate_tensors(
    self,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    return IntermediateTensors({})


DeepSeekMTP.make_empty_intermediate_tensors = _mtp_make_empty_intermediate_tensors


# ---------------------------------------------------------------------------
# PP+MTP runtime fix: carry spec_token_ids through ModelRunnerOutput and
# skip EngineCore.post_step in PP batch_queue mode to avoid the shared
# draft_token_ids_cpu buffer race between batches.
# ---------------------------------------------------------------------------


def _patch_model_runner_output() -> None:
    """Add spec_token_ids field to ModelRunnerOutput."""
    from vllm.v1 import outputs as outputs_mod

    cls = outputs_mod.ModelRunnerOutput
    fields = getattr(cls, "__dataclass_fields__", {})
    if "spec_token_ids" in fields:
        return
    cls.spec_token_ids = None

    orig_init = cls.__init__
    if getattr(orig_init, "_vllm_ascend_pp_mtp_patched", False):
        return

    @wraps(orig_init)
    def _patched_init(self, *args, spec_token_ids=None, **kwargs):
        orig_init(self, *args, **kwargs)
        self.spec_token_ids = spec_token_ids

    _patched_init._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    cls.__init__ = _patched_init

    empty = outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
    if not hasattr(empty, "spec_token_ids"):
        empty.spec_token_ids = None


def _patch_engine_core() -> None:
    """Skip post_step in PP batch_queue + spec_decode mode to avoid
    the shared draft_token_ids_cpu buffer race across batches."""
    from vllm.v1.engine.core import EngineCore

    orig_post_step = EngineCore.post_step
    if getattr(orig_post_step, "_vllm_ascend_pp_mtp_patched", False):
        return

    @wraps(orig_post_step)
    def _patched_post_step(self, model_executed: bool) -> None:
        if (
            getattr(self, "batch_queue", None) is not None
            and not getattr(self, "async_scheduling", False)
            and getattr(self, "use_spec_decode", False)
            and model_executed
        ):
            return
        return orig_post_step(self, model_executed)

    _patched_post_step._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    EngineCore.post_step = _patched_post_step


def _patch_scheduler_update_from_output() -> None:
    """Update request.spec_token_ids from ModelRunnerOutput.spec_token_ids."""
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_update = Scheduler.update_from_output
    if getattr(orig_update, "_vllm_ascend_pp_mtp_patched", False):
        return

    from vllm.v1.outputs import ModelRunnerOutput

    @wraps(orig_update)
    def _patched_update_from_output(
        self,
        scheduler_output,
        model_runner_output: ModelRunnerOutput,
    ):
        result = orig_update(self, scheduler_output, model_runner_output)

        spec_token_ids = getattr(model_runner_output, "spec_token_ids", None)
        if spec_token_ids and scheduler_output.num_scheduled_tokens:
            sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                req_index = model_runner_output.req_id_to_index[req_id]
                new_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )
                if not new_token_ids:
                    if request.spec_token_ids:
                        request.spec_token_ids = []
                    continue
                next_spec_token_ids = spec_token_ids[req_index]
                request.spec_token_ids = next_spec_token_ids

        return result

    _patched_update_from_output._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = _patched_update_from_output


_patch_model_runner_output()

try:
    _patch_engine_core()
except Exception:
    pass

try:
    _patch_scheduler_update_from_output()
except Exception:
    pass
