# DeepSeek V4.1 Flash prefill execution architecture comparison

Date: 2026-09-17

This note compares execution architecture, not quantization quality or kernel
throughput.  The three implementations do not use equivalent weights:

- **current**: this repository's released-checkpoint/official-precision path;
- **oMLX**: `v0.7.0.dev2`, commit
  [`b390b31e`](https://github.com/jundot/omlx/tree/b390b31e0c6831225fed0f24d278eb1db7fcb68b);
- **DwarfStar / ds4**: commit
  [`8db1d1d1`](https://github.com/antirez/ds4/tree/8db1d1d155cb0400a86a86b9c62d0defb3a6148b).

The conclusion is structural: before adding another local Metal kernel, the
current runtime should adopt the lifetime, batching, and synchronization shape
already demonstrated by DwarfStar.  In particular, full-resident weights must
be model-lifetime objects, routing and index selections must remain on device,
and mHC/attention/MoE must operate on the whole admitted chunk.  These changes
target all three measured dominant buckets and do not require adopting ds4's
quantization arithmetic.

## Measurement being explained

The canonical 2,063-token prefill in
[`artifacts/context-ladder/32k-run-20260916-233434-29168/result.json`](../artifacts/context-ladder/32k-run-20260916-233434-29168/result.json)
reports:

| Measurement | Seconds | Share of 225.927 s | Notes |
|---|---:|---:|---|
| Total prefill | 225.927 | 100.0% | 9.131 tok/s, 17 chunks |
| Layer wall | 225.605 | 99.9% | 680 layer calls |
| MoE path | 106.119 | 47.0% | Includes compact-bank creation/read |
| Attention path | 71.066 | 31.5% | Includes projection, indexer, per-token attention, publication |
| Post-MoE | 38.743 | 17.1% | Principally token-loop mHC post and concatenation |
| Unattributed layer wall | 9.678 | 4.3% of layer wall | Pre-attention/pre-MoE work and boundary overhead |
| Bank construction | 41.772 | 18.5% of total; 39.4% of MoE | 39.781 s reads, 680 constructions, 212,208 reads |

The component timers force GPU completion after attention, MoE, and post-MoE,
so they perturb the normal lazy schedule.  Bank construction is nested inside
the MoE bucket.  Consequently these rows are attribution, not additive
speedup predictions.

Other useful facts from the same run are 70,736 unique expert loads over 680
route unions (mean 104.0 experts per layer/chunk), only one exact union reuse,
1,485,360 gathered QMM rows, and 13,837,056 indexer rows.  Those facts explain
why request-local compact-bank reuse is ineffective even though adjacent
unions overlap by 83.1%.

## Structural comparison

| Concern | Current runtime | oMLX `v0.7.0.dev2` | DwarfStar V4.1 Metal |
|---|---|---|---|
| Prefill scheduling | Outer 128-token chunk, then layers 0..39. It is layer-major only inside each small chunk. | Default 2,048-token scheduler chunk; one chunk tensor runs through the ordinary 40-layer model forward. Thus chunk-major across long prompts, layer-major within a chunk. | Explicit layer-major sweep with graph-selected wide chunks. The V4.1 caps step through 2K/4K/8K, and long sweeps can defer decoder work. |
| Routed MoE assignment grouping | GPU top-6 is copied to CPU to form a chunk union. Compute then uses flattened `(token, slot)` gathered rows; it does not make expert-local work lists. | Flattens routes, GPU-sorts by expert, gathers token rows in sorted order, and builds native block metadata for grouped QMM. | Router output stays on device. A scatter kernel builds per-expert `(token, slot)` lists and a compact work list containing only non-empty expert tiles. |
| Expert batch shape | Three gathered QMMs over `[T*6, 1, K]`, `T <= 128`; each selected assignment is a row. | Variable `N_e` rows per expert after sorting, tiled by the selected native block variant. | Variable `N_e` per expert, dispatched as non-empty 32-row tiles; output is scattered back to fixed token/slot destinations. |
| Routed gate/up/down unit | Separate gathered gate, up, down operations over all assignments, with activation split into assignment chunks and a final route-reduction kernel. | Native grouped path can pair gate/up, apply SwiGLU/route weight, execute grouped down, then combine sorted routed outputs. | One routed-MoE batch encoding: expert-grouped gate/up with SwiGLU epilogue, grouped down, and device reduction. Quant-specific math differs, but the scheduling unit is reusable. |
| Expert weight runtime layout | At every layer/chunk, selected expert tensors are read and repacked into a new compact `[union, rows, packed-K]` bank plus scales. | Full-resident converted checkpoints use persistent stacked `SwitchLinear` tensors. Optional SSD offload uses fixed-capacity slot tensors, not a new shape/layout per request. | GGUF stores each layer's routed tensors in a conversion-time layout with stable offsets. Kernels bind the mapped tensors directly; no route-union repack exists in resident mode. |
| Weight/bank preparation lifetime | A compact bank lives for one layer call and is explicitly released immediately afterward. | Resident layout lives for the model lifetime. Offload plan/readers and expert slots also live for the model lifetime and retain hits across calls. | Model mmap, tensor descriptors, buffers, and resident weight views live for the engine lifetime. Streaming mappings/caches are also engine-owned. |
| Per-request bank/layout construction | Yes: 680 constructions for 17 chunks × 40 layers. | No in resident mode. Offload may replace slot contents on route misses, but its layout is allocated once. | No in full-resident mode. Streaming mode stages/maps admitted ranges but does not construct a new packed expert bank per request. |
| Shared and routed experts | Shared expert and routed bank are separate lazy branches, then added. They share the component evaluation but have no explicit joint schedule or fused combine. | Shared and routed branches are built in one lazy graph; a custom combine kernel restores token order, sums six routed rows, and adds the shared row. | Shared gate/up/SwiGLU/down and routed grouped MoE are encoded in the same layer command buffer, then device-added/reduced before the next mHC stage. |
| Inter-layer materialization/eval | Each block returns a materialized boundary in practice; profiling adds three hard synchronizations per layer. Bank release and CPU route/index consumers prevent a graph spanning layers. | The model builds a lazy graph across its layer loop and the scheduler evaluates cache states once per admitted chunk. Some bounded index-top-k and SSD-offload paths deliberately introduce internal evals. | Activations stay in preallocated GPU workspaces. Normal V4.1 prefill drains at a layer/chunk boundary; no token/expert boundary is required. Wide carry buffers bridge chunk partitions. |
| Metal command submission | MLX owns submission. Current host readbacks and `mx::eval` boundaries split the graph; submission structure is not a first-class runtime contract. | One engine stream and one outer `mx.eval` per scheduler chunk, with custom primitives embedded in that graph. | Explicit `begin_commands`/`end_commands`. Attention projection/core/output, mHC, shared/routed FFN, and expand are encoded into the layer/chunk command buffer. |
| CPU readback | Route IDs/boundaries/errors for every layer/chunk; index IDs, boundary scores/errors, and candidate masks for index-source layers; final logits. | Resident normal path keeps route sorting, index IDs, attention, and recurrent arrays on GPU. CPU readback appears in cache metadata and the optional expert-offload `ensure` path. | Route IDs, expert lists, index selections, and state remain device-resident. CPU sees progress/cancellation and final logits/snapshot copies, not per-layer routing/index results. |
| Attention/indexer batching | Q/K/V projections and score matrix are partly batched. Index score/mask/top-k are chunk-wide, but results are copied to CPU; sparse attention and output projection then loop token by token. Pure SWA also loops token by token. | Chunk-wide Q/KV projection, packed index score/top-k, and native packed sparse attention. Query rows are dispatched together; selected indices remain array values. | Batched projection and publication, packed index score/top-k, and batched raw/mixed/indexed attention. Final ring/publication state is copied in bulk. |
| Recurrent/compressed state publication | Copies the C++ state at entry, constructs one `SharedAttentionReference` per token, mutates a publication vector across reuse layers, and swaps state only after successful evaluation. This is atomic but host-object-heavy. | Request-local cache arrays hold window KV, compressed KV, index K, compressor tails, and Engram history. Lazy cache mutations are forced at the scheduler chunk boundary. | Preallocated device tensors receive bulk window/compressed/index publication. A partial layer sweep marks the graph invalid; position/history become valid only after the complete sweep. |
| CED/deferred decoder prefill | Absent: every chunk runs encoder layers 0..19 and decoder layers 20..39. | Absent at the pinned revision; [oMLX issue #3605](https://github.com/jundot/omlx/issues/3605) identifies it as missing. | Present. Large sweeps can run encoder-only and later rebuild only the decoder's exact dependency suffix; decoder layers also skip prefixes outside their required suffix. |
| Continued-prefill fast path | Reuses persistent state but always re-enters the same 128-token full-stack path. | Cached suffixes use the same chunked prefill machinery (normally 2,048 rows); no decoder deferral. | Resident and TP modes automatically batch continued prefills; very short appends retain a lower-latency short path, while long appends use the layer-major/deferred schedule. |
| Full-resident I/O | Routed experts still perform compact-bank reads and copies. Engram is mmap-backed. A model-lifetime full packed atlas is not the production path. | Main tensors are resident for the model lifetime; only optional expert/Engram offload performs demand I/O, with Engram prefetch support. | Main GGUF weights are directly mapped and resident on a large Mac, so routed-weight I/O disappears. Engram deliberately remains disk-backed and is prefetched/pipelined. In streaming mode, next-layer reads overlap current-layer compute. |
| Reference/optimized parity | Strong reference contract: exact hidden/pre-mix/logits/state/publication/hash/route checks over two 128-token chunks and invalid-token atomicity. Optimized promotion is currently bitwise-gated. | Synthetic official-reference fixtures plus focused reference-vs-native attention, top-k, routing, mHC, quantization, cache, and offload tests. Many numerical checks use tolerances; it is not a blanket full-path bitwise contract. | A/B flags compare scalar and batched schedules. Tests `memcmp` complete logits and every saved state span, then restore snapshots and continue decode. Documentation separately acknowledges that scalar/batched/TP and Q4 probability scores are not universally bit-identical. |

### Code anchors

Current runtime:

- chunk cap and per-layer bank release:
  [`src/model/text_encoder.cpp:29`](../src/model/text_encoder.cpp#L29),
  [`src/model/text_decoder.cpp:25`](../src/model/text_decoder.cpp#L25);
- token-loop post-MoE:
  [`src/model/block.cpp:52`](../src/model/block.cpp#L52),
  [`src/model/compressed_block.cpp:54`](../src/model/compressed_block.cpp#L54),
  [`src/model/reused_block.cpp:54`](../src/model/reused_block.cpp#L54);
- CPU route readback and one-row gate reductions:
  [`src/moe/reference.cpp:113`](../src/moe/reference.cpp#L113),
  [`src/moe/reference.cpp:164`](../src/moe/reference.cpp#L164);
- route-union bank construction and lifetime:
  [`src/moe/reference.cpp:265`](../src/moe/reference.cpp#L265),
  [`src/moe/expert_bank.cpp:139`](../src/moe/expert_bank.cpp#L139);
- token/slot gathered expert execution:
  [`src/moe/expert_bank.cpp:209`](../src/moe/expert_bank.cpp#L209);
- batched index score followed by CPU publication:
  [`src/attention/index_query.cpp:143`](../src/attention/index_query.cpp#L143);
- token-serial SWA/sparse attention:
  [`src/attention/swa_layer.cpp:39`](../src/attention/swa_layer.cpp#L39),
  [`src/attention/compressed_layer.cpp:149`](../src/attention/compressed_layer.cpp#L149).

oMLX:

- [chunk scheduler and one cache-state eval per chunk](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/scheduler.py#L3520-L3935);
- [layer loop and request-local state](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v41/language.py#L833-L930);
- [sorted/expert-grouped MoE and shared combine](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v41/language.py#L353-L653);
- [chunk-wide attention/indexer](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v41/language.py#L134-L351);
- [model-lifetime offload slots](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v41/moe_offload.py#L185-L284).

DwarfStar:

- [layer-major V4.1 sweep and command boundaries](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4.c#L41635-L41961);
- [device router and batched shared/routed MoE](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4.c#L41189-L41264);
- [per-expert assignment maps and non-empty tile work lists](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/metal/moe.metal#L7790-L7950);
- [deferred-decoder dispatch](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4.c#L75380-L75455);
- [full-state/logit/continued-decode parity tests](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/tests/test_deepseek41_graph.c#L499-L587) and
  [deferred-decoder parity/recovery](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/tests/test_deepseek41_graph.c#L1061-L1145).

## Missing structural optimizations and affected wall time

The estimates below identify the measured bucket that can move. They are not
summable and are not claims that the entire bucket can be removed.

| Priority | Missing architecture | Directly affected measured wall | Evidence and likely effect |
|---:|---|---|---|
| 1 | **Model-lifetime full-resident expert layout** | Bank construction **41.8 s**, inside MoE **106.1 s** | This is the cleanest structural removal. The current union changes almost every call, so caching by exact union cannot work. oMLX keeps stacked expert tensors; ds4 binds stable GGUF offsets. On the 512 GiB target, eliminate request/layer bank construction rather than optimizing its reads. The 41.8 s is the measured upper bound attributable to construction, not a guaranteed net saving. |
| 2 | **Chunk-wide mHC pre/post and state expansion** | Post-MoE **38.7 s**, plus part of the unattributed **9.7 s** | Current code performs `hc_post_reference` in two token loops per layer and concatenates rows. oMLX expresses mHC over the chunk tensor; ds4 has batch HC mix/expand stages. This is a schedule/layout problem before it is a new arithmetic kernel problem. |
| 3 | **Device-resident expert-major MoE scheduling** | The remaining portion of MoE after bank time: nominally **64.3 s** | Current compute is assignment-major `[6T]` gathered QMM and requires CPU IDs to build a bank. oMLX sorts routes; ds4 builds expert-local lists and only non-empty tiles. Preserve official FP8/FP4 projection semantics, but adopt expert-major work ownership and fixed output slots. |
| 4 | **End-to-end chunk-wide attention** | Attention **71.1 s** | Current index score/top-k is already partially batched, but CPU publication breaks the pipeline and the actual sparse attention/output is token-serial. Both comparison runtimes keep selected rows on device and dispatch query rows together. A dedicated SWA kernel should be considered only after this complete dataflow is specified and the non-bitwise gate is defined. |
| 5 | **Device-only route/index publication with diagnostic side channels** | MoE **106.1 s**, attention **71.1 s**, and synchronization overhead | Route IDs and index IDs are observables for parity, but they need not be normal-path CPU control inputs. Keep device tensors authoritative; copy compact tie/error/hash diagnostics only at qualification points. This also makes a single layer command graph possible. |
| 6 | **Wider admitted chunks and reusable GPU workspaces** | All component buckets; especially repeated launch/materialization overhead | Current 2,063-token run executes 17 × 40 layer calls. oMLX normally uses 2,048 rows; ds4 uses up to 2K/4K/8K by regime. A wider chunk alone is unsafe while banks scale with the route union, so it follows model-lifetime weights and bounded workspaces. |
| 7 | **One explicit layer/chunk submission and atomic commit** | The unattributed **9.7 s** and overlap lost inside all buckets | Preserve current transactional semantics, but represent pending state in preallocated device buffers and publish the frontier only after the layer sweep completes. ds4 demonstrates that cancellation can invalidate a partial sweep and rebuild exactly without per-token host publication objects. |
| 8 | **Deferred decoder / exact dependency suffix** | **0 s at the current 2,063-token frontier**; potentially large at 16K/32K+ | This does not explain the present 225.9 s result. It becomes important only for long/wide sweeps. ds4 skips redundant decoder prefixes and verifies exact state/logits afterward; oMLX does not yet have it. It should be copied as an execution algorithm after the short-path architecture is fixed. |
| 9 | **Continued-prefill dispatch distinct from decode and cold prefill** | No saving in this initial-prefill sample; future append wall | Current state is reusable but the execution shape remains 128-token full-stack. ds4 selects a short append path or wide layer-major path automatically. Preserve exact recurrent state while choosing the schedule by suffix length. |
| 10 | **I/O overlap for nonresident data** | Bank/Engram I/O; up to the exposed part of **41.8 s** in streaming configurations | For the requested full-resident M3 Ultra path, main-weight I/O should be absent. For smaller machines, ds4 overlaps next-layer mapping/read with current-layer compute and overlaps Engram reads. This is a fallback architecture, not a reason to retain request-local banks on the 512 GiB machine. |

## Implications for the next implementation loop

### Optimized-path qualification boundary

The token-serial path remains the bit-exact diagnostic oracle.  Production
promotion instead treats the released checkpoint, official FP8/FP4 precision,
causal ordering, and persistent-state semantics as hard invariants.  Route and
index decisions (including boundary ties), atomic publication, continuation
state, logits, and generated tokens are the optimized-path qualification
surface.  Intermediate tensor bit identity and the reference floating-point
reduction order are recorded when useful, but neither alone rejects a
candidate whose qualified observables satisfy their fixed gates.  A partial
layer/chunk sweep is never publishable.

### Near-term implementation order

| Phase | Structural change | Measured bucket addressed | Current repository action |
|---:|---|---:|---|
| 1 | Model-owned, transactionally published 40-layer resident expert atlas | 41.772 s bank construction | Move construction from first request to `TextBackboneReference` initialization; share immutable banks across every request/chunk. |
| 2 | Chunk-wide mHC pre/post and state expansion | 38.743 s post-MoE, plus part of 9.678 s overhead | Apply the existing batched mHC algebra directly to `[T,4,5120]`; retain token-serial methods as oracle. |
| 3 | Device route to expert-major work lists and grouped gate/up/down/reduce | Remaining nominal 64.3 s MoE | Replace assignment-major `[6T]` scheduling after Phases 1/2 qualification. |
| 4 | Chunk-wide attention/index/publication with atomic frontier commit | 71.066 s attention | Remove intermediate host control and token attention loops; do not start an SWA-only kernel first. |
| 5 | CED/deferred decoder and bounded replay | No 2K saving | Begin only after the 2K structural gates above. |

The prior full-resident experiment proved 40-bank reuse but built those banks
during the first prefill chunk and experienced substantial VM compression.
Phase 1 therefore separates model-initialization cost from warm request wall
and keeps memory pressure as an explicit promotion gate; it does not reinterpret
the earlier run as production qualification.

Implementation status (2026-09-17): Phase 1 ownership and Phase 2 chunk-wide
mHC are connected.  Bounded Phase 2 checks passed bitwise for layer 0 at
2x128 tokens, layers 0--2 including compressed publication, and layers 0--3
including the first reuse consumer.  Phase 3 has started: the resident path
omits route diagnostic readback, stable-sorts all `6T` assignments by expert on
device, runs gate/up/down in that order, and maps the canonical expert-ID
reduction order through the inverse permutation.  A 128-token/768-assignment
layer-0 fixture matched the assignment-major and token-serial accumulated and
BF16 routed outputs bitwise.  This is not yet DwarfStar-style non-empty tile
dispatch or full-path Phase 3 qualification.

The fresh full-backbone parity and resident full-path measurements are kept as
user-run validations because each takes several minutes:

```sh
bash tools/benchmark/run_layer_major_backbone_check.sh
bash tools/benchmark/run_resident_atlas_prefill_measurement.sh
```

The first uses a token-serial oracle and checks two 128-token chunks across all
40 layers (about 578 GB logical reads, 240 GB memory budget, no resume).  The
second runs one model with a transactionally constructed atlas, 2,063-token
prefill and one decode (about 289 GB one-time reads, 340 GB memory budget, no
resume).  Both retain failed run directories under `artifacts/`; neither is
called passed until its fresh result has been reviewed.

The next loop should be architecture-first and preserve the project's stated
correctness priority:

1. Define the non-bitwise qualification gate before changing reduction order:
   route IDs/ties, index IDs/ties/candidates, publication bytes, persistent
   state, logits/top-token behavior, and official-precision quality must be
   reported separately from reference bit-exactness.
2. Replace per-layer/per-chunk bank construction with a model-lifetime,
   full-resident, conversion-time-described expert layout. This can be probed
   with existing MLX primitives; it does not require a new attention kernel.
3. Make route/index arrays device-authoritative and allocate reusable
   request workspaces for route lists, selected indices, candidate masks,
   window KV, compressed KV, and pending publications.
4. Batch mHC pre/post and publication over the chunk. This directly attacks
   the independently measured 38.7-second bucket.
5. Re-measure the same component telemetry. Only then choose whether the next
   native kernel belongs to expert grouping, packed sparse attention, or SWA.
6. Add wider/continued prefill and, for 16K+, the ds4 decoder-suffix/CED
   algorithm. Validate complete state/logits and following decode, not an
   isolated primitive.

This ordering changes the role of telemetry: it verifies how much of the
known architecture gap has been closed instead of rediscovering the same gap
one kernel at a time.

## External performance context (not a speed claim for this runtime)

The magnitude of the architecture gap is independently visible, but the
numbers are not apples-to-apples:

- oMLX reports 452.19 tok/s for a 32K prefill on an M3 Ultra 512 GiB with its
  oQ4e checkpoint, model loading excluded
  ([release notes](https://github.com/jundot/omlx/releases/tag/v0.7.0.dev2)).
- DwarfStar records a V4.1 Q4 full-resident M3 Ultra median of 341.75 tok/s for
  initial 4K, then 641.49/715.96 tok/s for 8K/16K continued additions. Engram
  remains disk-only and loading is excluded
  ([release QA record](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/QA_BEFORE_RELEASES.md#17-deepseek-v41-flash)).
- DwarfStar's model guide explicitly states that large prefills use wide
  layer batches, streaming overlaps next-layer reads, and resident inference
  batches continued prefills
  ([model guide](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/docs/MODELS.md#deepseek-v41-flash)).

Those results use different quantization and kernels and therefore cannot
predict official-checkpoint throughput. They do establish that the scheduling,
state-lifetime, and submission architecture is practical on the exact target
hardware/model family.
