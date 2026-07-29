# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pipeline parallel accuracy test for Model Runner V2.

Run:
pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_pipeline_parallel.py
"""

import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain pipeline parallelism in one sentence.",
    "The future of artificial intelligence is",
]
MAX_TOKENS = 16


def _generate(pipeline_parallel_size: int) -> list[tuple[list[int], str]]:
    with VllmRunner(
        MODEL,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="mp",
        max_model_len=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        async_scheduling=True,
    ) as runner:
        return runner.generate_greedy(PROMPTS, MAX_TOKENS)


@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "OMP_NUM_THREADS": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_dense_pp2_matches_pp1():
    """MRV2 PP2 must produce the same greedy output as MRV2 PP1."""
    pp1_outputs = _generate(pipeline_parallel_size=1)
    pp2_outputs = _generate(pipeline_parallel_size=2)

    check_outputs_equal(
        outputs_0_lst=pp2_outputs,
        outputs_1_lst=pp1_outputs,
        name_0="MRV2-PP2",
        name_1="MRV2-PP1",
    )
