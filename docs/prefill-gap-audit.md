# M3 Ultra prefill execution-work gap audit

Date: 2026-09-17

This audit precedes any new local Metal kernel.  It compares execution work,
not peak kernel throughput.  The current path retains the released checkpoint,
official FP8/FP4 precision, causal order, and persistent-state semantics.  The
comparison sources are pinned oMLX `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
and DwarfStar/ds4 `8db1d1d155cb0400a86a86b9c62d0defb3a6148b`.

## Result in one sentence

The remaining near-10x wall gap is not one slow component: the current runtime
runs the whole 40-layer schedule **17 times** at 128 rows, explicitly lowers
wide dense projections into **594,144 one-row packed QMM invocations** plus
**181,544 one-row/batched-one-row FP GEMMs**, creates another **2,040** routed
gather-QMM invocations, fragments the attention core into **613,439 scalar QK
plus 117,579 AV batch invocations**, and crosses **340 blocking `mx::eval` plus
17 explicit synchronize calls** during the measured prefill.  This expands to
**269,115 command buffers, 237,345 compute encoders, and 12,358,908 Metal
compute dispatches**.  oMLX and ds4
admit approximately 2,048 rows, visit the 40 layers once, keep state decisions
on device, and fuse each layer's QK, softmax, and AV into one attention compute
invocation.

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
| Dense packed QMM invocations | **594,144**, all explicitly one-row | 288 wide logical QMM calls: seven/layer plus one on eight index-source layers; Metal expansion pending paired capture | wide layer tensors | **2,063x per corresponding projection**, 2,063.0x in aggregate |
| Routed QMM invocations | 2,040, three per layer visit | official MXFP4 path: paired gate/up + down once/layer, 80 projection dispatches | MXFP4 pre-M5 path: paired gate/up + down once/layer | **25.5x** versus 80 paired/down projection dispatches |
| Routed QMM rows | 1,485,360 | same three logical projections, expert-major | same logical projections, non-empty expert tiles | row count is expected work; dispatch shape is the defect |
| Token-serial FP GEMMs | **181,544**: router 82,520; grouped `wo_a` 82,520; index weight 16,504 | corresponding inputs remain wide | corresponding inputs remain wide | host-created one-row graph amplification |
| Explicit activation quantize/decode kernels | **18,912** | about 288 single-pass quantizers on the source-visible wide official path | fused into wide expert/projection schedules | approximately **65.7x** |
| Scalar QK invocations | 613,439 | 40 packed-attention compute invocations | 40 attention compute invocations | 15,336x invocation fragmentation |
| AV invocations | 117,579 equal-shape batches | fused into the same 40 attention invocations | fused into the same 40 attention invocations | 2,939x invocation fragmentation |
| Attention core dispatches | not yet a captured Metal total; at least the QK/AV counts above | one custom packed QK/softmax/AV dispatch per layer | layers 0--1: one raw dispatch each; layers 2--39: sort + indexed attention, total 78 | current primitive counters already exceed both by orders of magnitude |
| Fully token-serial attention calls | 12,378 (`6 × 2,063`) | 0 on the packed path | 0 on the batch path | eliminated by both comparison schedules |
| Chunk-attention host groups | 17,910 | no token/shape host grouping in the core | no token/shape host grouping in the core | 447.75 groups per layer versus one core dispatch |
| Index score rows | 13,837,056 | device query tiles, normally up to 512 query rows | 32-query-row score batches on index-source layers | all keep results on device; current still over-materializes consumers |
| Route/index result readbacks | 0 / 0 | 0 in resident normal path | 0 | gap already closed |
| Metal command buffers / compute encoders | **269,115 / 237,345** | paired capture pending | source schedule owns one wide graph | 395.8 / 349.0 per current layer visit |
| Metal compute dispatches | **12,358,908** | paired capture pending | source schedule is wide/fused | 5,990.7/token; 18,174.9 per current layer visit |
| Attention state concatenations counted | 102, 25,257,984 copied bytes | cache arrays updated in the lazy chunk graph | preallocated workspace and bulk publication | current count excludes additional reuse gather/temporary arrays |
| Warm expert layout conversions | 0; 40 one-time model banks | 0 | 0 | gap already closed |

The QK/AV ratios compare invocation granularity, not floating-point operation
counts.  A single comparison-runtime invocation covers all admitted query rows
and heads, so the ratio is exactly the scheduling problem this audit is meant
to expose.

### QMM/GEMM shape distribution

The packed-linear count is source-exact for 2,063 prefill rows.  Every call to
`PackedLinearReference::project_quantized()` loops over its first dimension and
submits a separate one-row `mx::quantized_matmul`; the existing
`project_batch_diagnostic()` already demonstrates that this is an execution
policy, not a checkpoint-layout constraint.

| Current one-row packed operation | Shape `(M,K,N)` | Invocations |
|---|---:|---:|
| Attention `wq_a` | `(1,5120,1280)` | 82,520 |
| Attention `wq_b` | `(1,1280,32768)` | 82,520 |
| Attention `wkv` | `(1,5120,512)` | 82,520 |
| Attention `wo_b` | `(1,8192,5120)` | 82,520 |
| Shared expert `w1` + `w3` | `(1,5120,2304)` | 165,040 |
| Shared expert `w2` | `(1,2304,5120)` | 82,520 |
| Eight index-query projections | `(1,1280,4096)` | 16,504 |
| **Total** | | **594,144** |

The additional routed MXFP4 gather-QMM distribution is 1,280 gate/up calls at
`(768,5120,2304)`, 640 down calls at `(768,2304,5120)`, 80 tail gate/up calls
at `(90,5120,2304)`, and 40 tail down calls at `(90,2304,5120)`.  The known FP
matmul distribution is 82,520 router calls at `(1,5120,384)`, 82,520 grouped
attention projections at batched `(8,1,4096)x(8,4096,1024)`, and 16,504 index
weight calls at `(1,5120,32)`.  These counts exclude chunk-wide mHC and index
score matmuls rather than guessing their Metal expansion.

At the pinned oMLX revision the same official expert bytes are losslessly
repacked as MXFP4.  One wide layer builds one expert block list, dispatches one
paired gate/up kernel, one paired SwiGLU/FP8 activation kernel, one down kernel,
and one combine/unpermute kernel.  Dense attention/shared projections receive
the admitted multi-row tensor directly.  Thus the decisive difference is not
QMM arithmetic volume: it is that the current correctness-oriented API fixes
the reduction at `M=1` 594,144 times.

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
layer visits, 594,144 one-row packed projections, 181,544 token-serial FP
GEMMs, 12,378 token-serial attention calls, 17,910 shape groups, and 613,439
scalar QK block iterations.  oMLX has one outer chunk and a 40-layer Python
loop; ds4 has one C layer sweep of 40 visits.  ds4's 32-row index score
encoding loop occurs only on index-source layers and never reads the selected
rows back to the CPU.

## Per-layer GPU structure

Normalized over the complete 2,063-row prefill, each current non-index-source
layer creates 14,441 one-row packed QMM calls (`7T`), 4,126 token-serial FP
GEMMs (`2T`), and 51 routed gather-QMM calls (three projections across 17
chunks).  Each of the eight index-source layers creates 16,504 packed QMMs
(`8T`), 6,189 token-serial FP GEMMs (`3T`), and the same 51 routed calls.  This
is before counting elementwise, sort, gather/scatter, attention, and state
kernels.  The corresponding oMLX logical projection schedule is seven wide
dense QMMs plus paired gate/up and one routed down per ordinary layer; an
index-source layer adds one wide dense QMM.  Its packed attention core is one
additional dispatch per layer.

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
target resource-allocation, command-buffer, and encoder tables.  An audit-only
process-local hook also counts the two `MTLComputeCommandEncoder` dispatch
selectors and their grid/threadgroup shapes.  The hook is DYLD-inserted only
into the captured target and is never linked into the production runtime.

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
decode with a process-local Metal selector hook.  Instruments is not launched.
Allow 5--10 minutes, up to 340 GB Unified Memory, and about 289 GB of one-time
checkpoint reads.  Selector-hook overhead invalidates wall time.  Logs are retained under
`artifacts/prefill-gap/metal-<timestamp>-<pid>/`; failure preserves partial
files, but partial runs are not counts.  There is no safe resume; rerun with
fresh model/request state.

The hook is enabled only across the three prefill phases and writes exact
prefill command-buffer, compute-encoder, compute-dispatch, and dispatch-shape
counts to `metal-dispatch-counts.json`.  It adds a mutex-protected shape
counter, so capture wall time remains invalid.
The same hook can be applied to an equivalently configured oMLX target.  ds4's
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

A second capture at `artifacts/prefill-gap/metal-20260917-194217-45001`
repeated the mistake of combining `Metal Application` with the system-wide
`GPU` instrument.  The target and selector hook completed, but the 12 GB trace
again remained in finalization until interrupted.  Its trace was deleted; its
result and 12,358,933 whole-process dispatch count are retained only as a
diagnostic and are not a prefill-scoped count.  The runner now uses only
`Metal Application`; the later replacement below removes Instruments
entirely.

A third capture at `artifacts/prefill-gap/metal-20260917-213606-46543` proved
that `Metal Application` alone is also unsuitable: it generated an 11 GB
package and remained in finalization after target exit.  The process-local
counter nevertheless completed and measured **12,358,907 prefill compute
dispatches** (10,468,136 `dispatchThreads`, 1,890,771
`dispatchThreadgroups`, 7,066 distinct shapes) for 2,063 tokens.  That is
5,990.7 dispatches/token and 18,174.9 dispatches per 128-token layer visit.
The invalid trace was deleted.  The runner now launches the target directly
with the hook and creates no Instruments package.

The no-Instruments confirmation
`artifacts/prefill-gap/metal-20260917-214856-46825` is clean at `2e363b7`
(both exits zero, empty tracked patch).  It reproduced 12,358,908 prefill
dispatches and additionally measured 269,115 command buffers and 237,345
compute encoders.  The one-dispatch difference from the preceding run is
normal setup/lazy-cache variation and does not alter the amplification result.
Its 107.777-second prefill is not a performance sample because the selector
hook serializes shape accounting.

The paired oMLX counter uses the same hook without Instruments:

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_omlx_prefill_dispatch_audit.sh
```

It pins clean oMLX `b390b31`, loads the same read-only official checkpoint,
uses the reviewed `DeepSeek-V4.1-Flash` settings (`deepseek_v41_engram_ssd_offload=true`,
MTP weights preserved), uses the same 2,063 token IDs in one model call, and
enables counting only after model load/cache creation.  Allow 5--10 minutes, up to 340 GB Unified
Memory, and about 289 GB checkpoint reads.  Logs are written to
`artifacts/prefill-gap/omlx-metal-<timestamp>-<pid>/`; partial output is not a
count and there is no resume.

The first oMLX attempt `omlx-metal-20260917-220157-47153` intentionally remains
as a failed diagnostic.  It used resident Engram rather than the reviewed oMLX
setting and aborted during model load with Metal OOM (408,665,719,208-byte peak
footprint, before prefill or counter output).  It contributes no comparison
count.

The second attempt `omlx-metal-20260917-220634-47256` loaded successfully with
Engram offload but mixed pinned oMLX (which requires MLX 0.32.2) with the older
app-bundle MLX 0.31.2.  That runtime rejects the model's zero-length packed
cache initialization and failed before evaluated prefill completion.  It also
contributes no count.  The runner now requires the existing clean
`/Users/kioju/.venvs/omlx-0.7.0.dev2` environment and verifies oMLX 0.7.0.dev2
plus MLX 0.32.2 before loading the model.

## Architecture decision

No new local kernel work begins until the paired oMLX count is reviewed.  After review,
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

Within step 1, the first no-new-kernel candidate is to replace the optimized
path's one-row `PackedLinearReference::project_quantized()` schedule with the
already-existing multi-row QMM path.  Reference retains the one-row reduction
order as the oracle; optimized qualification is decided at routes, state,
logits, and generation as required by the Phase 0 contract.

## Batched dense-QMM implementation checkpoint

`DSV41_RUNTIME_BATCHED_DENSE_QMM=1` now selects the existing multi-row MLX QMM
only from `PackedLinearReference::forward()` when `M > 1`.  It is opt-in and
adds no Metal kernel; unset/default execution continues to select the original
one-row oracle.  The short fixed linear fixture passed both selector modes.
The batched candidate matched the pinned oMLX QMM bit-for-bit in all three
cases.  Relative to the one-row oracle, expert `w1`/`w2` remained exact and
attention `wq_a` differed at one of 12,800 BF16 elements by one ULP
(`1.4901161e-8` absolute), the previously recorded diagnostic difference.

The next promotion gate is reproducible but intentionally not launched by the
agent because it takes several minutes and about 240 GB Unified Memory:

```sh
bash tools/benchmark/run_batched_dense_backbone_check.sh
```

It compares the token-serial one-row oracle with two 128-token, 40-layer
candidate chunks; route ties, state/publication/hash, logits argmax, and
invalid-token atomicity remain exact gates, while hidden/pre-mix/logits use the
existing relative-RMS bound.  Logs are retained under
`artifacts/prefill-gap/batched-dense-backbone-<timestamp>-<pid>/`; failure has
no resume and must be rerun from fresh state.  Only after that result is
reviewed should the 2,063-token wall observation run:

```sh
bash tools/benchmark/run_batched_dense_prefill_measurement.sh
```

That second run uses one resident model, up to 340 GB Unified Memory, about
289 GB one-time checkpoint reads, and writes the canonical context-ladder
result directory.  It is a performance observation, not a substitute for the
backbone semantic gate.

CED/deferred decoder prefill remains excluded from the 2K critical path.  It
is reconsidered only after these measured 2K work amplifications are removed.
