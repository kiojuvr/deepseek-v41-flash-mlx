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
