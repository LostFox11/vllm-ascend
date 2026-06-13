import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])


class TestNPUModelRunnerAsyncSpecSkip(unittest.TestCase):

    def _build_runner(self, req_specs, discard_mask=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.is_kv_producer = False

        req_ids = [f"req{i}" for i in range(len(req_specs))]
        runner.input_batch = SimpleNamespace(
            num_reqs=len(req_specs),
            req_ids=req_ids,
        )
        if discard_mask is None:
            discard_mask = [False] * len(req_specs)
        runner.discard_request_mask = SimpleNamespace(
            np=np.array(discard_mask, dtype=bool),
        )
        runner.requests = {}
        for req_id, (max_tokens, output_len) in zip(req_ids, req_specs):
            runner.requests[req_id] = SimpleNamespace(
                sampling_params=SimpleNamespace(max_tokens=max_tokens),
                output_token_ids=[-1] * output_len,
            )
        return runner

    def test_skip_when_all_sampled_reqs_reached_max_tokens(self):
        runner = self._build_runner([(1, 1), (1, 1)])

        self.assertTrue(runner._all_sampled_reqs_reached_max_tokens())
        self.assertTrue(runner._async_spec_state_can_be_skipped())
        self.assertTrue(runner._pp_skip_sampled_token_broadcast())

    def test_skip_before_non_final_pp_placeholder_is_appended(self):
        runner = self._build_runner([(1, 0)])

        self.assertFalse(runner._all_sampled_reqs_reached_max_tokens())
        self.assertTrue(
            runner._all_sampled_reqs_reached_max_tokens(
                include_current_sample=True,
            )
        )
        self.assertTrue(
            runner._async_spec_state_can_be_skipped(
                include_current_sample=True,
            )
        )

    def test_do_not_skip_when_any_sampled_req_can_continue(self):
        runner = self._build_runner([(1, 1), (2, 1)])

        self.assertFalse(runner._all_sampled_reqs_reached_max_tokens())
        self.assertFalse(runner._async_spec_state_can_be_skipped())
        self.assertFalse(runner._pp_skip_sampled_token_broadcast())

    def test_kv_producer_does_not_skip_decode_handoff_state_at_max_tokens(self):
        runner = self._build_runner([(1, 1)])
        runner.is_kv_producer = True

        self.assertTrue(runner._all_sampled_reqs_reached_max_tokens())
        self.assertFalse(runner._async_spec_state_can_be_skipped())
        self.assertFalse(runner._pp_skip_sampled_token_broadcast())

    def test_discarded_prefill_chunks_do_not_force_draft(self):
        runner = self._build_runner(
            [(32, 0), (1, 1)],
            discard_mask=[True, False],
        )

        self.assertTrue(runner._all_sampled_reqs_reached_max_tokens())
        self.assertTrue(runner._async_spec_state_can_be_skipped())
        self.assertTrue(runner._pp_skip_sampled_token_broadcast())

    def test_wait_async_token_comm_sets_cpu_valid_count(self):
        runner = self._build_runner([(4, 1)])
        comm_state_cls = __import__(
            "vllm_ascend.worker.model_runner_v1",
            fromlist=["AsyncPPTokenCommState"],
        ).AsyncPPTokenCommState

        class DoneWork:
            def wait(self):
                return None

        valid_count = torch.tensor([2], dtype=torch.int32)
        comm_state = comm_state_cls(
            works=[DoneWork()],
            tensors=[valid_count],
            valid_sampled_token_count=valid_count,
            receive_side=True,
        )
        runner._async_pp_token_comm_states = [comm_state]

        runner._pp_wait_async_token_comm(comm_state, activate=True)

        self.assertEqual(runner._pp_valid_sampled_token_count.tolist(), [2])
        self.assertEqual(runner._async_pp_token_comm_states, [])

    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_last_pp_rank_prepare_does_not_clear_local_draft(self, mock_pp_group):
        runner = self._build_runner([(4, 1)])
        runner.use_async_scheduling = True
        runner._draft_token_ids = torch.tensor([[11, 12, 13]])
        runner._async_pp_token_comm_states = []

        mock_pp_group.return_value = SimpleNamespace(
            world_size=4,
            is_last_rank=True,
        )
        scheduler_output = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"req0": 1},
        )

        runner._pp_prepare_async_token_state_for_execute(scheduler_output)

        self.assertIsNotNone(runner._draft_token_ids)
        self.assertFalse(runner._pp_current_batch_requires_prev_token_state)

    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_pending_pp_state_can_be_used_after_independent_batch(self, mock_pp_group):
        runner = self._build_runner([(4, 1), (4, 1)])
        runner.use_async_scheduling = True
        runner.use_async_spec_decode = True
        runner.valid_sampled_token_count_gpu = None
        runner._draft_token_ids = None
        runner._pp_valid_sampled_token_count = None
        runner.input_batch.prev_sampled_token_ids = torch.tensor([[5]])
        runner.input_batch.prev_req_id_to_index = {"req0": 0}

        comm_state_cls = __import__(
            "vllm_ascend.worker.model_runner_v1",
            fromlist=["AsyncPPTokenCommState"],
        ).AsyncPPTokenCommState

        class DoneWork:
            def wait(self):
                return None

            def is_completed(self):
                return True

        prev_tokens = torch.tensor([[5]], dtype=torch.int32)
        draft_tokens = torch.tensor([[7, 8, 9]], dtype=torch.int32)
        valid_count = torch.tensor([2], dtype=torch.int32)
        comm_state = comm_state_cls(
            works=[DoneWork()],
            tensors=[prev_tokens, draft_tokens, valid_count],
            prev_sampled_token_ids=prev_tokens,
            draft_token_ids=draft_tokens,
            valid_sampled_token_count=valid_count,
            prev_req_id_to_index={"req0": 0},
            receive_side=True,
        )
        runner._async_pp_token_comm_states = [comm_state]

        mock_pp_group.return_value = SimpleNamespace(
            world_size=4,
            is_last_rank=False,
        )

        runner._pp_prepare_async_token_state_for_execute(
            SimpleNamespace(
                total_num_scheduled_tokens=1,
                num_scheduled_tokens={"req1": 1},
            )
        )

        self.assertEqual(runner._async_pp_token_comm_states, [comm_state])
        self.assertIsNone(runner.input_batch.prev_sampled_token_ids)

        runner._pp_prepare_async_token_state_for_execute(
            SimpleNamespace(
                total_num_scheduled_tokens=1,
                num_scheduled_tokens={"req0": 1},
            )
        )

        self.assertIs(runner.input_batch.prev_sampled_token_ids, prev_tokens)
        self.assertEqual(runner.input_batch.prev_req_id_to_index, {"req0": 0})
        self.assertIs(runner._draft_token_ids, draft_tokens)
        self.assertIs(runner.valid_sampled_token_count_gpu, valid_count)
        self.assertEqual(runner._pp_valid_sampled_token_count.tolist(), [2])
        self.assertEqual(runner._async_pp_token_comm_states, [])


if __name__ == "__main__":
    unittest.main()
