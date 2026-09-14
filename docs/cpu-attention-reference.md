# Offline CPU attention reference

This is a local diagnostic reference, not an official CUDA execution or a
full-model qualification. Python remains an offline validation dependency only.

`tools/reference/cpu_reference.py` reads official checkpoint attention tensors
without modifying them. It verifies M1 source and shard-header identities and
records hashes of the tensor bytes read. RoPE functions are extracted from the
verified official source. FP8 projection and sparse attention use explicit CPU
formula transcriptions; CPU reduction order is not the CUDA reduction order.

`trace_cpu_attention.py` takes frozen native layer 0 `attn_in` and computes ten
attention boundaries independently thereafter. It supports a position-zero
token-serial trace of 1–128 tokens. mHC, MoE, Engram and subsequent layers are
outside this diagnostic's scope.

Run from the repository root:

```sh
bash tools/benchmark/run_cpu_attention_trace.sh
```

The default input is `artifacts/logits-trace/native-20260914-145608-21879`
(60 tokens); another input directory can be supplied as the first argument.
The script uses `/Volumes/SDXC-512/deltafin/.venv/bin/python`, four CPU threads,
and an estimated few GiB of RAM. It does not use the GPU or load the full model.
It may take several minutes and is intended for user execution.

Outputs live in `artifacts/cpu-attention/run-<timestamp>-<pid>/`: identities,
`test.log`, `trace-exit-code.txt`, overall `exit-code.txt`, trace arrays and
manifest, `comparison.log`, and `report.json`. Exit zero means the report was
generated, not numerical agreement. Review identities and numerical differences
before drawing conclusions. The first real-checkpoint run was reviewed on
2026-09-14; results follow below.

On failure, rerun the command to create a fresh directory and preserve the
previous logs. Token-level resume is not implemented. If trace generation
completed, rerun only comparison:

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/compare_traces.py \
  --native <input-directory> --oracle <run-directory>/trace \
  --output <run-directory>/report.json
```

The comparator rejects incompatible token sequences, duplicate boundary names,
an empty boundary intersection, and mismatches between manifest and array
shape/storage. Reports retain per-array hashes, logical dtype differences,
bit mismatches, finite numerical differences, and nonfinite counts. Partial
boundary coverage remains explicit. A numerical tolerance never establishes
full-path qualification or replaces the local optimized-path bit-exact contract.

Short tests, without loading checkpoint tensors:

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python -m unittest discover -s tests/reference
```

Ten tests passed on 2026-09-14. They cover comparator contracts and small CPU
arithmetic cases; they do not validate the complete attention computation.

## Reviewed 60-token result (2026-09-14)

Run `run-20260914-160650-23333` completed with trace and overall exit code zero.
`artifacts/cpu-attention/reviewed-result.json` records review of script, M1,
official source, input/output manifest and compared array hashes, the completed
60-token log, and metrics. All checked hashes matched; all ten boundaries were
finite on both sides. Checkpoint tensor payloads were not rehashed during review;
their hashes recorded by the exporter remain provenance, not a new snapshot audit.

| Boundary | Bit mismatches | Max absolute difference |
| --- | ---: | ---: |
| attn_in (fixed input) | 0 | 0 |
| attn_qr | 3 | 0.0078125 |
| attn_qb | 44 | 0.0078125 |
| attn_q | 50 | 0.015625 |
| attn_kv_norm | 3 | 0.001953125 |
| attn_kv | 2 | 0.001953125 |
| attn_kv_quant | 0 | 0 |
| attn_o_raw | 201 | 0.001953125 |
| attn_o | 205 | 0.001953125 |
| attn_out | 16907 | 0.015625 |

The first recorded difference is Q after projection and normalization, before
RoPE. Unlike the earlier frozen-Q RoPE diagnostic, this experiment also includes
upstream projection and norm differences. Quantized SWA KV agrees exactly in
this sample. Output projection increases the mismatch count; that observation
alone does not establish an output-projection bug.

`diagnose_cpu_qr.py` ran a short CPU sensitivity check (under two seconds).
FP64 accumulation of the same FP8-quantized Q projection differed from FP32 in
five BF16 projection elements. Both `rsqrt` and `1/sqrt` normalization variants still
differed from native Q-normalized output in three elements. The report is
`artifacts/cpu-attention/diagnose-qr-20260914.json`. This rules out those CPU
variant changes as a fix for this sample, not a native implementation error or
CUDA rounding difference. Native pre-norm Q is not present in the existing trace;
capturing that boundary is the next diagnostic step. No numerical acceptance
threshold or full-path promotion contract was changed.

## Native pre-norm Q isolation

`dsv41-q-projection` now loads only layer 0 Q projection/norm and the frozen
60-token attention input. The GPU run took about 0.48 seconds, followed by a
CPU diagnostic under two seconds. Report:
`artifacts/cpu-attention/diagnose-native-qr-20260914.json`.

| Comparison | BF16 bit mismatches | Max absolute difference |
| --- | ---: | ---: |
| Recomputed native qr vs original native trace | 0 | 0 |
| Native qa vs CPU FP32 block accumulation | 3 | 1.9073486328125e-6 |
| Native qa vs CPU FP64 block accumulation | 2 | 1.4901161193847656e-8 |
| CPU norm on native qa vs native qr | 1 | 0.0078125 |

This separates projection differences from a normalization difference on identical
inputs. It does not yet isolate the normalization difference into variance,
reciprocal square root or final multiplication, nor demonstrate official CUDA
agreement. The original three differing qr elements must not be attributed only
to RoPE or only to projection. See [MLX acceptance approach](mlx-numerical-acceptance.md)
for how these observations feed a future cross-backend criterion.

Reproduce with fresh output paths (short diagnostic, no full backbone load):

```sh
cmake -S . -B build-mlx
cmake --build build-mlx --target dsv41-q-projection -j 4
build-mlx/dsv41-q-projection \
  /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  artifacts/checkpoint/summary.json \
  artifacts/logits-trace/native-20260914-145608-21879 \
  artifacts/cpu-attention/native-q-repeat
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/diagnose_cpu_qr.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --native artifacts/logits-trace/native-20260914-145608-21879 \
  --native-q artifacts/cpu-attention/native-q-repeat \
  --output artifacts/cpu-attention/diagnose-native-qr-repeat.json
```

## Q normalization root cause (60-token sample)

The extended native probe records FP32 variance, variance + epsilon, inverse
square root, normalized activation and weighted activation. Its final BF16 qr
still reproduces the original full-backbone trace exactly. GPU and CPU runs
took about 0.54 and 0.44 seconds respectively. Results and reviewed identities:
`artifacts/cpu-attention/diagnose-qnorm-stages-20260914.json` and
`artifacts/cpu-attention/qnorm-stages-reviewed-20260914.json`.

| Stage, same native Q input | Bit mismatches | Max absolute difference |
| --- | ---: | ---: |
| FP32 variance | 21 / 60 | 4.656612873077393e-10 |
| FP32 inverse square root | 7 / 60 | 1.9073486328125e-6 |
| FP32 normalized | 8667 / 76800 | 7.152557373046875e-7 |
| FP32 weighted | 8345 / 76800 | 9.5367431640625e-7 |
| BF16 result | 1 / 76800 | 0.0078125 |

Substituting the native adjusted variance into CPU `rsqrt` or `1/sqrt` gives
bit-identical inverse square roots. Using native inverse square roots also makes
CPU multiplication and the final BF16 output bit-identical; casting native
weighted FP32 output to BF16 reproduces qr exactly. The observed norm discrepancy
is therefore attributable to variance reduction, not inverse square root or
weight multiplication on identical inputs in this sample.

At token 34/channel 469, native weighted FP32 is `-1.02734375`, exactly the BF16
midpoint, whereas CPU is `-1.0273436307907104`, one FP32 ULP away. BF16 outputs are
`-1.03125` and `-1.0234375`. This illustrates why an absolute bound on intermediate
FP32 arithmetic does not translate directly to the same absolute bound after a
BF16 cast. It does not establish a global one-ULP bound for RMSNorm, resolve the
separate Q projection discrepancy, or measure routing/logits effects.

For a fresh stage trace produced by `dsv41-q-projection`, run:

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/diagnose_qnorm_stages.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --native-q <native-stage-directory> --output <fresh-report.json>
```

## Independent synthetic Q inputs (2026-09-14)

`export_qnorm_cases.py` generates 76 rows independently of the coding trace:
zero, constant, alternating signs, one nonzero coordinate, seeded normal
samples at scales 2^-60 through 2^30, and adjacent BF16 perturbations. These are
calibration/exploration inputs, not held-out acceptance data. The native probe's
`--norm-input` mode bypasses projection and uses the real layer 0 norm weight.
Synthetic rows use `sample_ids`, not invented token IDs.

All 97,280 BF16 output elements agreed with CPU. FP32 weighted output differed
in 13,122 elements, max_abs 9.5367431640625e-7. Native adjusted variance injected
into CPU inverse square root again yields exact results, as do subsequent
same-input multiplication and BF16 cast. Max symmetric relative error is
1.3810851821575396e-7 after epsilon addition and 3.198423134748927e-7 in weighted
output. Absolute variance/inverse errors alone are misleading across these scales.

An underflow corner is also visible: the sparse 2^-60 row has zero native variance
while CPU retains a subnormal. Raw variance relative error is therefore 1, even
though epsilon 1e-20 dominates and the final output agrees. This corner must not
be described as uniformly small relative error or solely reduction reordering.
No observed bound here is promoted into a universal RMSNorm tolerance.

GPU and CPU runs each took under one second. Fixture/input preservation,
script and output hashes were reviewed in
`artifacts/cpu-attention/qnorm-cases-reviewed-20260914.json`; the diagnostic with
relative errors is `artifacts/cpu-attention/diagnose-qnorm-cases-relative-20260914.json`.
Short reference tests: 10 passed. The known real-input midpoint mismatch remains
unresolved under the current acceptance contract. Routing/logits effects remain
unmeasured.

Reproduction (fresh directories/report required):

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_qnorm_cases.py --output <fixture>
build-mlx/dsv41-q-projection \
  /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  artifacts/checkpoint/summary.json <fixture> <native-output> --norm-input
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/diagnose_qnorm_stages.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --native-q <native-output> --output <report.json>
```
