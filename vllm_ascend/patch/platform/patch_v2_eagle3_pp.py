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
"""Allow Ascend MRV2 to experiment with EAGLE3 + pipeline parallelism.

Upstream vLLM currently blocks this combination because EAGLE3 auxiliary hidden
states need extra propagation across PP stages. vLLM Ascend provides a local
PP aux propagation patch, so filter only this specific unsupported marker and
leave all other upstream MRV2 feature gates intact.
"""

from vllm.config.vllm import VllmConfig


_ORIGINAL_GET_V2_UNSUPPORTED = VllmConfig._get_v2_model_runner_unsupported_features
_EAGLE3_PP_UNSUPPORTED = "EAGLE3 with pipeline parallelism"


def _patched_get_v2_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _ORIGINAL_GET_V2_UNSUPPORTED(self)
    speculative_config = self.speculative_config
    if (
        speculative_config is not None
        and speculative_config.method == "eagle3"
        and self.parallel_config.pipeline_parallel_size > 1
    ):
        unsupported = [
            feature
            for feature in unsupported
            if feature != _EAGLE3_PP_UNSUPPORTED
        ]
    return unsupported


VllmConfig._get_v2_model_runner_unsupported_features = (  # type: ignore[method-assign]
    _patched_get_v2_model_runner_unsupported_features
)
