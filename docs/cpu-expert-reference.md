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

## Full layer 0 CPU MoE comparison

`tools/reference/cpu_moe_compare.py` uses the real native `ffn_in`, frozen native
route IDs/weights, and independently decoded CPU FP4 experts. It computes each
routed expert only on first use, applies the official SwiGLU clamps and route
weights, adds the shared expert, and compares the BF16 `moe_out` with native.

Run the prepared script because expert slabs may take several minutes and multiple
GiB of RAM to load:

```sh
bash tools/benchmark/run_cpu_moe_compare.sh
```

Results and progress logs are saved under
`artifacts/cpu-attention/cpu-moe-<timestamp>-<pid>/`. Exit zero means the report
was written, not that outputs agree. A failed run leaves logs; retrying creates a
fresh directory. Mid-token resume is not implemented.

The comparison now also writes CPU `shared` and `routed` component arrays next
to the report (`report.shared.bf16.npy` and `report.routed.bf16.npy`). Run the
script again after a native component replay to obtain directly comparable
components; the previous CPU result predates these files.

## Reviewed six-expert comparison (2026-09-14)

The first real 60-token run loaded 152 unique routed experts. CPU output differs
from native in 270,799 of 307,200 BF16 elements; maximum absolute difference is
`0.03125`, mean `0.0016184431733563542`. Both outputs are finite. The reviewed
record is `artifacts/cpu-attention/cpu-moe-20260914-reviewed.json`.

This cannot be classified as gate score tolerance: routes were frozen from
native, and no route ID disagreement was introduced. Candidate causes are FP4
expert projection accumulation, SwiGLU cast placement, shared-expert arithmetic,
and MoE summation order. Component traces are required before any numerical
policy change.

## Corrected native comparison (2026-09-14)

After fixing route-weight placement to after `w2`, the CPU comparison was rerun
against the corrected native replay. The aggregate difference fell from 270,799
to 72,351 BF16 elements; max_abs fell from `0.03125` to `0.015625`, and mean
abs from `0.0016184431733563542` to `7.552796159870923e-05`. All values remain
finite. Shared expert remains close (215 mismatches, max_abs `0.0009765625`),
while routed sum has 101,146 mismatches, max_abs `0.0078125`, mean
`7.550499140052125e-05`. The reviewed result is
`artifacts/cpu-attention/cpu-moe-20260914-212331-reviewed.json`.

This validates the semantic correction and localizes the remaining difference to
routed FP4 arithmetic/reduction or expert-specific inputs. It is no longer a
route-weight placement error and cannot be addressed by gate tolerance. Expert
251's three selected contributions were bit-identical in the separate stage
trace; broader expert IDs still need sampling before attributing the residual to
a universal backend reduction effect. No MoE or full-model qualification is
claimed.

## Reviewed component comparison (2026-09-14)

The corrected 60-token CPU run was compared with the native component replay.
The shared expert is close (215 BF16 mismatches, max_abs `0.0009765625`, mean
`6.269321772833791e-08`). The routed six-expert sum accounts for almost all of
the aggregate error: 287,263 mismatches, max_abs `0.0234375`, mean
`0.001618312089703977`. The aggregate remains 270,799 mismatches with max_abs
`0.03125`. Both paths are finite and route IDs were frozen from native.

This localizes the unresolved issue to routed FP4 expert arithmetic or its
weighted accumulation. It is not evidence for relaxing gate conditions. The
review record is `artifacts/cpu-attention/cpu-moe-20260914-component-reviewed.json`.
The next diagnostic should compare one routed expert's w1/SwiGLU/w2 output and
the route-weighted contribution against a native per-expert trace before changing
the CPU formula or production MLX path.

After the route-weight correction, pass the corrected native replay trace as the
third argument:

```sh
bash tools/benchmark/run_cpu_moe_compare.sh \
  artifacts/logits-trace/native-20260914-145608-21879 \
  artifacts/cpu-attention/gate-native-weights-20260914 \
  artifacts/cpu-attention/moe-replay-20260914-192425-27753/trace
```

The first argument supplies `ffn_in`; the third supplies corrected `moe_out`.
Omitting it compares against the pre-correction full-backbone output and is not
valid for judging the fixed implementation.
