# Numerical investigation decision — 2026-09-14

## Decision

Keep the current native RMSNorm and projection arithmetic unchanged. Stop tuning
them to make this particular CPU comparison bit-identical. Continue the independent
CPU reference through layer 0 mHC and MoE, then downstream layers, so discrete
decisions and logits can be evaluated. This is a development sequencing decision,
not cross-backend numerical acceptance or M2 qualification.

Official tensor identity, model formulas, cast boundaries, KV encoding and state
semantics remain mandatory. Local reference → optimized path still requires bit
equality. No new numerical tolerance is adopted. Known differences remain visible
and must be revisited if independent inputs reveal semantic errors, unstable
growth, or different discrete decisions. CPU and fixed oMLX evidence are not an
official CUDA oracle.

## Evidence and limits

- On the original 60-token trace, native Q normalization has one BF16 midpoint
  discrepancy when CPU uses the same pre-norm Q. Substitution isolates the cause
  to variance reduction; same-input inverse square root, multiplication and cast
  agree. Projection has separate discrepancies.
- On 76 independent synthetic calibration inputs, final BF16 normalization agrees
  in all 97,280 elements. Intermediate FP32 differences and an underflow corner
  remain. These inputs do not prove a universal tolerance or cover other norm widths.
- In `artifacts/cpu-attention/q-effect-20260914.json`, rerunning the common CPU
  sparse-attention calculation reproduces the recorded CPU output exactly. The
  quantized native and CPU KV inputs are also bit-identical.

| Controlled comparison, 60 tokens | BF16 mismatches | Max absolute difference |
| --- | ---: | ---: |
| Native Q vs CPU Q through the same CPU sparse attention | 155 | 0.0009765625 |
| Native sparse attention vs CPU, with identical native Q and KV | 46 | 0.001953125 |
| Original native vs CPU sparse-attention outputs | 201 | 0.001953125 |

The controlled Q substitution includes all upstream Q differences: projection,
normalization and RoPE. It does not isolate the RMSNorm discrepancy alone. The
46-element residual is a same-input attention arithmetic difference whose internal
cause has not been localized. Counts from separate comparisons are not an additive
causal decomposition, even though the totals happen to sum in this sample.

This experiment ends at `attn_o_raw`, before inverse RoPE and output projection.
It measures neither expert routing nor logits; attention-vector argmax is not a
proxy for either. The previously recorded final Attention output difference remains
16,907 elements, max_abs 0.015625. Its full propagation has not been isolated.

## Next implementation boundary

Extend the CPU formula reference with layer 0 mHC post-attention, FFN preparation,
and MoE routing. Compare both frozen native inputs and continuously propagated CPU
inputs. Record scores, selected IDs, sixth/seventh margins, ties and route weights;
do not use near-equal tensors to excuse a discrete difference. Keep full teacher-
forced logits and long-context qualification pending until their paths exist and
are measured. Long validation runs remain user-executed scripts.

The Q-effect diagnostic is a short CPU-only check (about 0.53 seconds here), reads
only the sink checkpoint tensor and saved trace arrays, and saves input/source/
script hashes with its metrics. Reproduce with a fresh report path:

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/diagnose_attention_q_effect.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --native artifacts/logits-trace/native-20260914-145608-21879 \
  --cpu artifacts/cpu-attention/run-20260914-160650-23333/trace \
  --output <fresh-report.json>
```
