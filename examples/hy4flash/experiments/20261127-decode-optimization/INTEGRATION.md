# Integration with the updated Hy4-flash-opti branch

## Target revision and conflict resolutions

Integrated target revision `e0bf79ae1` and its preceding changes:

- `1bec7f497`: the original precision-fix PR was already merged by the target.
- `cb8d877e8`: paged IndexPool and compact C8 Decode paths.
- `adb9c45a7`: pipeline-parallel eager Prefill and companion vLLM patch.
- `e0bf79ae1`: Attention common/eager/Decode modules and operator cleanup.

The integration retains the target's Attention mixins, paged IndexPool,
compact C8 sparse/window branches, pipeline-parallel stage ownership and
checkpoint filtering. It ports per-layer metadata, packed Engram projection
and gates, and C8/RoPE layout preparation onto that structure.

The non-compact Decode fallback uses the candidate's restored CANN MLA FIA,
including sink correction, host length providers and graph task updates.
Stable compaction reuses the target's shared KV helpers. Packed BF16 KV is
adapted to the native ABI; the target's compact C8 branches stay in place.
MoE keeps the measured BF16 sum followed by native RMSNorm.

No new environment variable, CLI argument, `additional-config` key or launch
script change is introduced relative to this target revision. The target's
existing paged C8/Indexer options are retained. Internal fusion selection
and guarded fallback behavior remain as requested.

## Hardware evidence boundary

**788.48 token/s** and **AIME2026 30/30 correct (100%)** were measured on the
container candidate imported as `267c9a373`, before this integration. The
report, raw results and source hashes remain unchanged as historical
evidence. The integration changes source layout and combines newer target
paths, so those numbers are not a rerun of the resulting merge revision.

Full-model AIME, graph capture/replay and same-load throughput on the combined
source remain hardware validation work before merge. Existing services were
not restarted or rebound during the integration.

## Integration checks

- HY4 component suite on the combined source: **295 passed**, with
  `TRITON_INTERPRET=1` and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. This includes
  pipeline Prefill tests, per-layer metadata, packed projection, fallback,
  compaction/host-length tests, and a mocked native CANN dispatch regression.
- Ruff check and Ruff formatting check: **passed**, 89 Python files checked
  for formatting in the model package and HY4 unit tests.
- Markdown lint and `git diff --check`: **passed**.
- `bash format.sh ci`: attempted with a 30-second timeout; the first
  pre-commit GitHub dependency fetch did not complete. Full repository
  formatting/secret-scan checks are not claimed as passed.
- The first broader test run exposed a pre-existing test mock using the old
  zero-argument metadata lookup. The fixture now accepts the layer name;
  production metadata selection remains per-layer.

These CPU and mocked checks do not establish native NPU graph replay or
full-model accuracy/throughput for the integrated revision. Integration
check logs are stored alongside the original [check logs](checks/).
