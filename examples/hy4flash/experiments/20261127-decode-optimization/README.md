# HY4 Decode optimization: performance and AIME2026

The container candidate completed **788.48 token/s** end-to-end output
throughput on the AISBench workload below and **30/30 correct (100%)** on
AIME2026. These are separate workloads on the same source candidate. There
is no paired baseline measurement establishing an optimization speedup.

## Source and deployment

- Server: `141.61.81.118`.
- Model container: `hy4-pr22839-363c1616b-1127` (`a926367c9014`).
- Evaluation container: `hy4-118-pd-dfly5-1121`.
- Tested source: `/opt/hy4/validation/decode-layout-gate-fusion-20261127/source`.
- Source branch: `peterDengcx/vllm-ascend:codex/hy4-fia-moe-precision-20261127`.
- P: physical devices 0-3, TP4 eager. D: devices 4-7, TP1/DP4/EP4.
- BF16 model, FP8 KV cache, Engram CPU offload, DFly5 and
  `hy4flash_dfly_dp_graph=true`, with target FULL_DECODE_ONLY graphs.
- Both services use max sequences 128 and max batched tokens 4096.
- Performance service context: 262144; AIME service context: 131072.

The source synchronization adds per-layer metadata lookup, packed Engram
Decode projection, four-way gate fusion, packed C8 layout/mask preparation,
and RoPE layout packing with existing fallbacks. CANN FIA owns NPU
QK/softmax/PV again, with sink correction and graph task updates retained.
C8 dequantization remains FP32 before BF16 storage. MoE preserves the BF16
expert-sum boundary before CANN RMSNorm. IndexPool precision fixes, DFly5
and Draft graphs remain enabled. The older FP32 attention module remains
as a reference; the NPU runtime no longer calls it.

The previous precision-fix report describes an earlier source and a partial
29-question result. It is retained as historical evidence and must not be
used as the final report for this candidate.

## AISBench performance

The original command ran from `/home/d00943518/aisbench_auto_tools_prefix`:

```bash
python3 aisbench_test.py --input_len 131072 --output_len 1024 \
  --data_num 128 --request_rate 0 --concurrency 32 --seed 1024 \
  --dataset_type prefix_cache --repeat_rate 0.90 --prefix_test
```

All 128 raw request records were checked: all succeeded, each input had
131093 tokens (131072 raw tokens plus 21 chat-template tokens), and each
output had 1024 tokens. The benchmark exited with code 0.

| Metric | Mean / total | P99 |
| --- | ---: | ---: |
| End-to-end output throughput | **788.4775 token/s** | N/A |
| Generated tokens | 131072 | N/A |
| Benchmark duration | 166.2343 s | N/A |
| Successful / failed requests | 128 / 0 | N/A |
| Maximum / average concurrency | 32 / 29.615 | N/A |
| TPOT | 35.6 ms/token | 48.4 ms/token |
| TTFT | 2068.0 ms | 6158.6 ms |
| End-to-end request latency | 38461.1 ms | 53390.9 ms |
| Stream inter-token latency | 112.5 ms | 161.2 ms |
| D preemptions | 0 | N/A |

The P-side actual prefix hit rate was **99.98398%**: 16777216 hits from
16779904 queried tokens. All 128 prompts had been prewarmed. The CLI
`repeat_rate=0.90` is the generation setting, not the observed cache hit
rate. DFly accepted 90088 of 206480 draft tokens (43.63%). Stream ITL and
TPOT differ because speculative decoding can emit multiple tokens together.

A separate 20.0392-second D-only interval produced 14933 tokens, or
**745.1887 token/s across four D devices**. Each rank had eight running
requests at both endpoints; P was idle and P/D prompt counters did not
change. No confirmed steady window was selected in the recorded analysis.
This interval is not a steady-state throughput claim, and does not replace
the completed 128-request end-to-end result. No profiling result is claimed
by this performance run.

## AIME2026 accuracy and latency

EvalScope 1.15.0 evaluated all 30 questions once, with concurrency 16,
`temperature=0.9`, `top_p=1`, `top_k=-1`, streaming and the official model
chat template. No additional output limit or generation seed was supplied.
The dataset revision was
`evalscope/aime26@12ae93b463353acd4d272dbbc7dbecbcfb902642`.
Integer answers were converted to strings for the local dataset adapter;
question text and numeric answers were unchanged.

| Metric | Result |
| --- | ---: |
| Correct / completed / requested | **30 / 30 / 30** |
| Accuracy | **100%** |
| Errored requests | 0 |
| Mean TPOT | 25.801 ms/token |
| Mean TTFT | 1.489025 s |
| Mean request latency | 173.641917 s |
| Mean output length | 7406.733333 tokens |
| Total output tokens | 222202 |
| EvalScope average per-request output rate | 42.66 token/s |

The per-request output rate is not aggregate service throughput. This
single sampled evaluation establishes the observed 30-question result;
it does not establish bitwise equivalence for every input.

## Evidence and verification

- [Final per-question scores](aime2026-final-results.json).
- [Original EvalScope report](raw/eval/results/20261127_195122/reports/hy4-flash/aime26.json).
- [Dataset provenance](raw/eval/dataset-provenance.json).
- [AISBench analysis](raw/server/night-1900-20261127/performance-summary.json).
- [AISBench latency statistics](raw/server/night-1900-20261127/outputs/20261127_194243/performances/vllm-api-stream-chat/gsm8k.csv).
- [AISBench throughput statistics](raw/server/night-1900-20261127/outputs/20261127_194243/performances/vllm-api-stream-chat/gsm8k.json).
- [Post-run source hashes and service bindings](raw/server/night-1900-20261127/final-source-and-bindings.json).

The 47 model-source hashes recorded after the night experiment were checked
against the container source before synchronization: no mismatch. Source
files are transferred from that candidate; platform build artifacts and
Python caches are excluded. The only production-source adjustment after
transfer is Ruff formatting of `decode_fusions.py`; its parsed AST matches
the tested container file. An existing IndexPool test fixture was updated
for per-layer metadata without changing production behavior.

Focused and related CPU regressions passed **88/88** in an isolated source
copy, including all 33 new focused cases. Ruff checks and formatting checks
passed for the changed Python files. `git diff --check` passed. Full
`bash format.sh ci` was attempted but could not complete because its first
pre-commit GitHub dependency fetch stalled; it was terminated, and is not
reported as passed. Check logs are retained in [checks/](checks/).

Earlier pre-restoration component checks do not
validate the restored CANN FIA path; the completed model-level performance
and AIME runs above are the relevant hardware evidence.

The container log filenames use its local clock. Preserve their original
timestamps and epoch records when correlating runs across environments.
