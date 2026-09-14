# Layer 0 gate routing diagnostic

`dsv41-gate-trace` records the official layer 0 gate calculation from frozen
native `encoder.layer0.ffn_in`. It saves raw scores, bias-corrected scores,
selected top-6 expert IDs and the sixth-minus-seventh corrected-score margin.
This is a local MLX diagnostic; it does not establish official CUDA agreement,
expert output equality, logits equality or full-model qualification.

The checked 60-token run is `artifacts/cpu-attention/gate-native-20260914/`.
The route trace contains 60 rows and 360 selected IDs. Margins range from
`3.147125244140625e-05` to `0.1359872817993164`; nine rows are below `1e-3`.
Thus several positions are close enough to require explicit score and ID
comparison when CPU and MLX implementations are compared. A small activation
difference must not be silently classified as harmless if it changes a selected
expert. No top-6 boundary tie occurred in this trace.

Reproduce with a fresh output directory (short GPU diagnostic):

```sh
build-mlx/dsv41-gate-trace \
  /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  artifacts/checkpoint/summary.json \
  artifacts/logits-trace/native-20260914-145608-21879 \
  <fresh-output-directory>
```

The next CPU MoE reference will consume the same frozen FFN input and compare
raw scores, corrected scores, selected IDs, route weights and expert outputs.
Tie policy remains explicit: exact top-6 boundaries are not an oracle pass; the
current local fallback chooses the lowest expert ID and records the event.
