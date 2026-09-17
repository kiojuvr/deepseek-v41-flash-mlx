# M3 Ultra prefill execution-work gap audit

Date: 2026-09-17

This audit precedes any new local Metal kernel.  It compares execution work,
not peak kernel throughput.  The current path retains the released checkpoint,
official FP8/FP4 precision, causal order, and persistent-state semantics.  The
comparison sources are pinned oMLX `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
and DwarfStar/ds4 `8db1d1d155cb0400a86a86b9c62d0defb3a6148b`.

## Result in one sentence

The remaining near-10x wall gap is not one slow component: the current runtime
runs the whole 40-layer schedule **17 times** at 128 rows, creates **2,040**
routed QMM invocations instead of a wide-sweep grouped schedule, fragments the
attention core into **613,439 scalar QK plus 117,579 AV batch invocations**, and
crosses **340 blocking `mx::eval` plus 17 explicit synchronize calls** during
the measured prefill.  oMLX and ds4 admit approximately 2,048 rows, visit the
40 layers once, keep state decisions on device, and fuse each layer's QK,
softmax, and AV into one attention compute invocation.

This establishes the next unit of work as a 2,048-row transactional layer
sweep with fused attention and grouped routed MoE.  Further optimization of a
single existing scalar QK, AV, QMM, or SWA primitive is not justified.

## Reviewed current measurement

`artifacts/context-ladder/32k-run-20260917-123159-42514` is a clean run of
`7a7c66ad` (exit 0, empty tracked patch, matching identities).  It measured
2,063 prefill tokens in 106.233166 seconds, or 19.41955 token/s, and generated
the unchanged next token 339.  Model construction took 26.252531 seconds and
is outside prefill.  Forty resident expert banks were constructed once; the 17
request chunks constructed none.  Route and index diagnostic readbacks were
both zero.

Peak MLX allocation was 302,822,383,907 bytes, maximum RSS was
273,313,529,856 bytes, peak process footprint was 305,867,454,968 bytes, and
swap remained zero.  VM compression/decompression still increased by about
32.08/32.04 GiB.  This run is the execution-work baseline, not a 32K or
cross-runtime performance qualification.

The user-observed official-precision oMLX rate on the same M3 Ultra is about
193.6 token/s, making the directional throughput ratio 9.97x.  The oMLX run is
not yet a paired 2,063-token trace with the same prompt and cache condition, so
the audit does not assign that entire ratio to any one row below.

## Normalized 2,048-row execution work

`current` values are dynamic counters from the reviewed 2,063-token run unless
marked source-counted.  oMLX/ds4 values are code-path counts for a 2,048-row
first sweep and are not presented as captured Metal totals.

| Work unit | Current, 2,063 rows | oMLX, about 2,048 rows | ds4, 2,048 rows | Amplification established |
|---|---:|---:|---:|---:|
| Request/scheduler chunks | 17 | 1 | 1 wide sweep | 17x |
| Layer visits | 680 | 40 | 40 | 17x |
| Routed assignments | 495,120 | same semantic `6T` | same semantic `6T` | no arithmetic amplification implied |
| Routed QMM invocations | 2,040, three per layer visit | grouped primitives once per layer; exact Metal expansion needs capture | MXFP4 pre-M5 path: map + fused gate/up/SwiGLU + down + sum, 4 kernels/layer | Current projection calls are 25.5x ds4's two projection kernels/layer; 12.75x versus all four routed kernels |
| Routed QMM rows | 1,485,360 | same three logical projections, expert-major | same logical projections, non-empty expert tiles | row count is expected work; dispatch shape is the defect |
| Scalar QK invocations | 613,439 | 40 packed-attention compute invocations | 40 attention compute invocations | 15,336x invocation fragmentation |
| AV invocations | 117,579 equal-shape batches | fused into the same 40 attention invocations | fused into the same 40 attention invocations | 2,939x invocation fragmentation |
| Attention core dispatches | not yet a captured Metal total; at least the QK/AV counts above | one custom packed QK/softmax/AV dispatch per layer | layers 0--1: one raw dispatch each; layers 2--39: sort + indexed attention, total 78 | current primitive counters already exceed both by orders of magnitude |
| Fully token-serial attention calls | 12,378 (`6 × 2,063`) | 0 on the packed path | 0 on the batch path | eliminated by both comparison schedules |
| Chunk-attention host groups | 17,910 | no token/shape host grouping in the core | no token/shape host grouping in the core | 447.75 groups per layer versus one core dispatch |
| Index score rows | 13,837,056 | device query tiles, normally up to 512 query rows | 32-query-row score batches on index-source layers | all keep results on device; current still over-materializes consumers |
| Route/index result readbacks | 0 / 0 | 0 in resident normal path | 0 | gap already closed |
| Attention state concatenations counted | 102, 25,257,984 copied bytes | cache arrays updated in the lazy chunk graph | preallocated workspace and bulk publication | current count excludes additional reuse gather/temporary arrays |
| Warm expert layout conversions | 0; 40 one-time model banks | 0 | 0 | gap already closed |

The QK/AV ratios compare invocation granularity, not floating-point operation
counts.  A single comparison-runtime invocation covers all admitted query rows
and heads, so the ratio is exactly the scheduling problem this audit is meant
to expose.

## Host work and synchronization

The current canonical runner fixes `prefill_chunk_tokens=128`.  Each chunk
executes the encoder and decoder layer loops and then performs these blocking
boundaries with finite checks disabled:

| Boundary per 128-row chunk | Count |
|---|---:|
| Four producer compressors materialize latent/tail state | 4 |
| Two KV formats per producer evaluate an error flag and read it on host | 8 |
| Four producer cache publications materialize packed main/index state | 4 |
| Encoder and decoder final tensors | 2 |
| Canonical runner output and finite checks | 2 |
| Total `mx::eval` | 20 |
| Additional runner `mx::synchronize` | 1 |

Across 17 chunks this is exactly 340 `mx::eval` calls plus 17 explicit
`mx::synchronize` calls on the measured path.  Of the eval calls, 306 are in
the runtime/state path and 34 are benchmark validation.  Some calls may find
an already-materialized array, so the count is a host boundary/API count, not
an assertion that every call submits a non-empty command buffer.

Pinned oMLX has no blocking `mx.eval` inside the DeepSeek V4.1 layer forward.
Its scheduler evaluates the cache state once at the admitted chunk boundary;
the two Engram layer boundaries use `mx.async_eval`.  ds4 explicitly encodes a
layer/chunk command buffer and marks the graph invalid until the complete
sweep commits position/history.  Neither requires per-token or per-expert host
publication.

The dominant host loops are therefore current's 17 scheduler chunks, 680
layer visits, 12,378 token-serial attention calls, 17,910 shape groups, and
613,439 scalar QK block iterations.  oMLX has one outer chunk and a 40-layer
Python loop; ds4 has one C layer sweep of 40 visits.  ds4's 32-row index score
encoding loop occurs only on index-source layers and never reads the selected
rows back to the CPU.

## Per-layer GPU structure

Current, for each of 680 layer visits:

1. build mHC/attention projection lazily;
2. form token-dependent window/selected-row arrays on the host;
3. run scalar QK blocks, shape-bucket softmax, and AV batches;
4. materialize producer quantization/error/publication where applicable;
5. device route, stable assignment sort, three assignment-major gathered QMMs,
   activation, route reduction, and the shared branch;
6. return to a host-owned state/publication object; encoder/decoder and
   producer boundaries force evaluation.

oMLX, once for each of 40 layers in a 2,048-row graph:

1. chunk-wide mHC and Q/KV projections;
2. packed index score/top-k on device;
3. one `deepseek_v41_packed_attention` dispatch fusing QK, block softmax, and
   AV for all query rows/heads;
4. device route sort, expert-major block metadata, paired gate/up when
   available, grouped down, and `combine_sorted_experts` unpermute/reduction;
5. lazy cache state, evaluated by the scheduler at the chunk boundary.

ds4, once for each of 40 layers in a 2,048-row sweep:

1. preallocated batch mHC, attention projection, publication/index selection;
2. raw or indexed mixed attention over all rows; indexed layers issue one
   chronological sort and one fused QK/softmax/AV kernel;
3. one router batch and shared-expert batch;
4. one expert-major work map reused by fused gate/up/SwiGLU, down, and sum;
5. batch HC expand, then commit the frontier only after the sweep succeeds.

## Temporary materialization and gather/scatter

The current path allocates vectors of per-token windows, selected rows,
ordered rows, validity masks, projected rows, and publication objects.  The
reviewed telemetry counts 102 state concatenations but does not count every
MLX `take`, temporary `concatenate`, or allocator event inside the reuse path;
those values must not be invented.  The focused Metal capture below records
target resource-allocation, command-buffer, and encoder tables.  Those close
submission counts, but they do not by themselves expose individual kernel
dispatches.

oMLX uses one device route permutation and inverse permutation per layer, then
one custom combine/unpermute.  ds4 builds one token-to-expert map and compact
non-empty tile list per layer, reuses it for gate/up/down, then performs one
sum.  Both perform gather/scatter as bounded layer operations rather than as
token/64-key host groups.

## DwarfStar kernel portability decision

DwarfStar is MIT licensed at the pinned commit.  Its published V4.1 Q2/Q4
weights and their quantized arithmetic cannot be used by this runtime because
official checkpoint precision is a hard invariant.  The following split is
therefore binding:

| ds4 mechanism | Official-path fit | Decision |
|---|---|---|
| Token-centric `kernel_mul_mm_id_map_scatter_work_ne20_6` and compact non-empty work trailer | Route IDs/weights and six-route semantics match; independent of checkpoint quantization | **First-choice port** for work-list ownership and dispatch |
| Pre-M5 MXFP4 paired gate/up/SwiGLU, down, tail-cull, and half-LUT kernels | Same 32-value E2M1 + E8M0 format class and correct M3 path, but ds4 interleaves `block_mxfp4` weights/scales while the current atlas exposes separate packed/scales buffers | **Adapt, do not rewrite**: either bind separate scale storage in the MIT kernel or create a transactional model-lifetime interleaved view; qualify official bytes and arithmetic boundaries |
| Indexed mixed attention kernel | Correct wide execution shape and atomic publication model; ds4 storage/accumulation is tied to its graph and released quantized model path | Reuse scheduling and MIT implementation where layouts match; do not claim drop-in semantic compatibility |
| Q2_K, IQ2_XXS, or Q4_K expert kernels | Different checkpoint precision | Reject for production official path |
| Wide sweep workspace and invalid-until-complete frontier | Matches causal/persistent-state requirements | Port execution structure directly |

oMLX's Apache-2.0 `deepseek_v41_packed_attention` is the most direct
official-model attention candidate: it is selected specifically for BF16,
64 heads, head dimension 512, and dispatches once for the full query length.
For attention, it should be evaluated alongside the ds4 MIT kernel before a
new implementation is designed.  For routed experts, ds4's MXFP4 work-map and
pre-M5 kernels are the primary source because they already encode the M3
expert-major schedule requested here.

## Metal count closure

The dynamic counters above are sufficient to reject further local primitive
tuning, but they are not a fabricated total Metal dispatch count.  A
reproducible capture runner is provided:

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_prefill_gap_metal_capture.sh
```

It runs one official-checkpoint resident model, 2,063-token prefill and one
decode under the focused `Metal Application` and `GPU` instruments.  Allow
10--20 minutes, up to 340 GB Unified Memory, and about 289 GB of one-time
checkpoint reads.  The focused trace is expected to remain far below a full
system trace, but free space must still be monitored.  Trace overhead
invalidates wall time.  Logs are retained under
`artifacts/prefill-gap/metal-<timestamp>-<pid>/`; failure preserves partial
files, but partial traces are not counts.  There is no safe resume; rerun with
fresh model/request state.

After a successful run:

```sh
python3 tools/benchmark/summarize_prefill_metal_trace.py \
  artifacts/prefill-gap/metal-<timestamp>-<pid>/metal.trace
```

The summarizer exports command-buffer, encoder, application/GPU interval, and
resource-allocation tables.  `Metal Application` rows are target-scoped;
`GPU` intervals may include other processes and are descriptive only.  If
Instruments does not expose kernel dispatch events, the summary reports
`kernel_dispatch_rows: null`; encoder count is never mislabelled as kernel
dispatch count.  A captured current submission total can then be paired with
an equivalently configured oMLX trace.  Exact individual dispatch totals
require Shader Timeline or MLX command-encoder instrumentation.  ds4's
released Q4 run is useful for schedule counts but is not an official-precision
performance pair.

### Rejected full System Trace

The first capture at
`artifacts/prefill-gap/metal-20260917-125156-43131` completed the target
process successfully at revision `59448dd6`, retained the same workload
counters, and measured a trace-perturbed 121.323768-second prefill.  It then
spent more than 50 minutes finalizing a 12,740,704,656-byte attachment.
Although xctrace eventually returned zero, the package lacks template
metadata and `xctrace export --toc` fails with `Document Missing Template
Error`.  It is therefore **not a valid trace and contributes no Metal count**.
The focused runner above replaces it.

## Architecture decision

No new local kernel work begins until the focused capture is reviewed.  After review,
implementation order is:

1. replace the 128-row outer request schedule with a 2,048-row transactional
   layer sweep and preallocated state/workspace, eliminating the 17x layer and
   synchronization multiplier;
2. integrate the already-published full-chunk packed attention implementation
   rather than optimizing the 613,439 scalar QK calls individually;
3. port ds4's expert-major map and compatible MXFP4 kernels/adapters, preserving
   official bytes, shared/routed semantics, and device reduction;
4. remove production KV error-flag host reads by accumulating a device error
   status and checking it only at the atomic chunk commit;
5. repeat the existing route/index/publication/continuation/logits/generation
   qualification and same-fixture full-path measurement.

CED/deferred decoder prefill remains excluded from the 2K critical path.  It
is reconsidered only after these measured 2K work amplifications are removed.
