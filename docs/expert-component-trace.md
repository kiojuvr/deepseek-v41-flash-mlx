# Routed expert component trace

`dsv41-expert-trace` records the route-weighted contribution of one routed
layer-0 FP4 expert for every token in a frozen native trace. It uses native MLX
expert weights and the saved native route IDs/weights. The trace is intended to
pair with the independent CPU expert implementation and isolate w1/SwiGLU/w2
and route-weighted contribution differences.

The first token selects expert 251. Build and run a fresh trace directory:

```sh
cmake -S . -B build-mlx
cmake --build build-mlx --target dsv41-expert-trace -j 4
build-mlx/dsv41-expert-trace \
  /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  artifacts/checkpoint/summary.json \
  artifacts/logits-trace/native-20260914-145608-21879 \
  artifacts/cpu-attention/gate-native-weights-20260914 \
  <fresh-output-directory> 251
```

The expert slab load and GPU execution may take tens of seconds or longer. The
tool saves `encoder.layer0.expert_contribution.npy` and a manifest with token
IDs, expert ID and occurrence count. Checkpoint data remains read-only. An
existing output directory is rejected; rerun with a fresh path after failure.

This is a native component trace, not an official CUDA oracle or full-model
qualification. Route tolerance does not apply to expert contribution equality.

## Expert 251 result (2026-09-14)

The native trace for expert 251 was compared with the independent CPU FP4 path.
It was selected at tokens 0, 18 and 40. The route-weighted contribution differs
in 14,581 BF16 elements, maximum absolute difference `0.0087890625`, mean
`0.000051083494327`; both outputs are finite. The reviewed record is
`artifacts/cpu-attention/expert-251-cpu-compare-20260914-reviewed.json`.

This confirms that the routed-sum discrepancy is already present in one expert
contribution. It cannot be repaired by gate score tolerance or by changing only
the six-expert accumulation order. The remaining split is w1/w3 FP4 projection,
SwiGLU clamp/cast, w2 projection, and route-weight multiplication. Native
per-stage tracing is the next step; this result is not a CPU/CUDA match or
qualification.

## Root cause correction (2026-09-14)

The previous native `ExpertReference::forward` applied the route weight to the
intermediate SwiGLU activation before `w2`. The official model applies the
route weight after the expert's `w2` output. `ExpertReference::forward` now
delegates to the corrected post-`w2` component path. The earlier
270,799-element CPU MoE difference was therefore measured against an invalid
native path; existing full-backbone and MoE replay results were regenerated.

A fresh stage trace must be generated after this correction before any claim is
made about gate, up, activation, or contribution equality. The corrected
60-token MoE comparison currently reports 72,351 BF16 mismatches, concentrated
in the routed sum; see `docs/cpu-expert-reference.md`. No expert-stage or
full-model qualification is carried forward from the pre-correction trace.
