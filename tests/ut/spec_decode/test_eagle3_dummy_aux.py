# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Eagle3 auxiliary hidden states used by dummy profile runs."""

from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    _get_eagle3_expected_aux_count,
    _normalize_eagle3_dummy_aux_hidden_states,
)


def _make_draft_model(**inner_attributes):
    return SimpleNamespace(model=SimpleNamespace(**inner_attributes))


def test_get_expected_aux_count_supports_eagle3_model_variants():
    hidden_states = torch.empty(4, 8)

    qwen_draft = _make_draft_model(num_aux_layers=3)
    llama_draft = _make_draft_model(num_aux_hidden_states=4)
    fallback_draft = _make_draft_model(
        fc_input_size=24,
        config=SimpleNamespace(target_hidden_size=8),
    )

    assert _get_eagle3_expected_aux_count(qwen_draft, hidden_states) == 3
    assert _get_eagle3_expected_aux_count(llama_draft, hidden_states) == 4
    assert _get_eagle3_expected_aux_count(fallback_draft, hidden_states) == 3


def test_normalize_dummy_aux_states_pads_without_allocating_tensor():
    draft_model = _make_draft_model(num_aux_layers=3)
    last_hidden_states = torch.ones(4, 8)
    first_aux = torch.full((4, 8), 2.0)
    second_aux = torch.full((4, 8), 3.0)
    aux_hidden_states = [first_aux, second_aux]

    normalized = _normalize_eagle3_dummy_aux_hidden_states(
        draft_model,
        last_hidden_states,
        aux_hidden_states,
    )

    assert normalized is not None
    assert normalized[0] is first_aux
    assert normalized[1] is second_aux
    assert normalized[2] is first_aux
    assert len(aux_hidden_states) == 2
    assert aux_hidden_states[0] is first_aux
    assert aux_hidden_states[1] is second_aux


def test_normalize_dummy_aux_states_uses_last_hidden_state_when_empty():
    draft_model = _make_draft_model(num_aux_hidden_states=3)
    last_hidden_states = torch.ones(4, 8)

    normalized = _normalize_eagle3_dummy_aux_hidden_states(
        draft_model,
        last_hidden_states,
        None,
    )

    assert normalized is not None
    assert len(normalized) == 3
    assert all(hidden_state is last_hidden_states for hidden_state in normalized)


def test_normalize_dummy_aux_states_trims_excess_states():
    draft_model = _make_draft_model(num_aux_layers=2)
    last_hidden_states = torch.ones(4, 8)
    aux_hidden_states = [torch.full((4, 8), value) for value in range(3)]

    normalized = _normalize_eagle3_dummy_aux_hidden_states(
        draft_model,
        last_hidden_states,
        aux_hidden_states,
    )

    assert normalized is not None
    assert len(normalized) == 2
    assert normalized[0] is aux_hidden_states[0]
    assert normalized[1] is aux_hidden_states[1]
    assert len(aux_hidden_states) == 3


def test_normalize_dummy_aux_states_respects_disabled_aux_input():
    draft_model = _make_draft_model(
        use_aux_hidden_state=False,
        num_aux_layers=3,
    )

    normalized = _normalize_eagle3_dummy_aux_hidden_states(
        draft_model,
        torch.ones(4, 8),
        [torch.ones(4, 8)],
    )

    assert normalized is None
