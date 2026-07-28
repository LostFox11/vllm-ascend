# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.v1.worker.gpu.attn_utils as gpu_attn_utils
import vllm.v1.worker.utils as worker_utils

from vllm_ascend.patch.worker.patch_qwen3_next_mtp import bind_kv_cache


def test_bind_kv_cache_is_patched_at_mrv2_call_site():
    assert worker_utils.bind_kv_cache is bind_kv_cache
    assert gpu_attn_utils.bind_kv_cache is bind_kv_cache
