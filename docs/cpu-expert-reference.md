# CPU FP4 expert reference

`tools/reference/cpu_reference.py` now decodes routed I8 weights as packed E2M1
values (`0, .5, 1, 1.5, 2, 3, 4, 6` with sign bit) and applies E8M0 group-32
scales. The CPU linear path uses the same explicit ascending block accumulation
as the FP8 reference. This is an offline diagnostic implementation.

`tools/reference/cpu_moe_fixture.py` validates the implementation against the
existing real layer 0 expert 0 fixture. The fixture contains ten BF16 inputs and
the stored CPU block-reference outputs for `w1` (23,040 elements) and `w2`
(51,200 elements). The current result is in
`artifacts/linear/cpu-moe-fixture-20260914-v2.json`:

- w1: zero bit mismatches, zero absolute difference
- w2: zero bit mismatches, zero absolute difference
- SwiGLU clamp probe: finite output, clamp limits respected

This validates packed FP4 decode and individual expert linears. It does not
validate checkpoint routing, six-expert aggregation, shared expert composition,
or full-model logits. Runtime and production Python dependencies are unchanged;
the script is offline validation only.

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/cpu_moe_fixture.py \
  --output <fresh-report.json>
```
