
## CPU formula comparison (2026-09-14)

`tools/reference/compare_gate_cpu.py` recalculates layer 0 gate scores from the
frozen native `ffn_in` using the official score function (`sqrt(softplus(x))`),
then applies the checkpoint bias and stable expert-ID tie order. The reviewed
report is `artifacts/cpu-attention/gate-cpu-compare-20260914-v2.json`.

For all 60 tokens, the selected top-6 expert IDs agree with native. Raw score
maximum absolute difference is `1.6689300537109375e-6` (mean
`2.1458528465245763e-7`); bias-corrected maximum is `1.9073486328125e-6`
(mean `2.1536317262871307e-7`). There are no nonfinite values. The sixth minus
seventh margin ranges from `3.147125244140625e-5` to `0.135986328125`, with nine
rows below `1e-3`.

This supports agreement of the local MLX and CPU gate calculation for this frozen
input, including near-margin rows. It is not an official CUDA oracle, does not
compare route weights or expert outputs, and does not qualify the full model.
The result also does not justify a universal score tolerance: score differences
must still be checked against the margin and selected IDs.

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/compare_gate_cpu.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --native artifacts/logits-trace/native-20260914-145608-21879 \
  --gate artifacts/cpu-attention/gate-native-20260914 \
  --output <fresh-report.json>
```
