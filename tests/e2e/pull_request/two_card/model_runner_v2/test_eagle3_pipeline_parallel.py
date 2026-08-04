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
"""Eagle3 speculative decoding with Model Runner V2 pipeline parallelism.

Run:
pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_eagle3_pipeline_parallel.py
"""

import os
from unittest.mock import patch

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MAIN_MODEL = "Qwen/Qwen3-8B"
EAGLE3_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"
NUM_SPECULATIVE_TOKENS = 3
BASELINE_ACCEPTANCE = [0.68, 0.40, 0.18]


def _acceptance_per_pos(metrics) -> list[float]:
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * NUM_SPECULATIVE_TOKENS
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    assert num_drafts > 0
    return [
        num_accepted_tokens / num_drafts
        for num_accepted_tokens in num_accepted_tokens_per_pos
    ]


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_eagle3_pp2_acceptance_full_decode_only():
    tokenizer = AutoTokenizer.from_pretrained(
        MAIN_MODEL,
        trust_remote_code=True,
    )
    prompts = [
        {"role": "user", "content": "Hello, my name is"},
        {"role": "user", "content": "The president of the United States is"},
        {"role": "user", "content": "The capital of France is"},
        {"role": "user", "content": "The future of AI is"},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )
    speculative_config = {
        "method": "eagle3",
        "model": EAGLE3_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[12],
    )

    with VllmRunner(
        MAIN_MODEL,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=2048,
        max_num_seqs=16,
        gpu_memory_utilization=0.7,
        disable_log_stats=False,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
    ) as runner:
        _ = runner.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    acceptance_per_pos = _acceptance_per_pos(metrics)
    assert all(acceptance > 0 for acceptance in acceptance_per_pos), (
        f"Eagle3 PP2 acceptance should be non-zero, got {acceptance_per_pos}"
    )
    assert all(
        abs(actual - expected) < 0.10
        for actual, expected in zip(acceptance_per_pos, BASELINE_ACCEPTANCE)
    ), f"acceptance_per_pos {acceptance_per_pos} does not match {BASELINE_ACCEPTANCE}"
