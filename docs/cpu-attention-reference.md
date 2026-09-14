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
before drawing conclusions. This real-checkpoint run is still pending as of
2026-09-14.

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
