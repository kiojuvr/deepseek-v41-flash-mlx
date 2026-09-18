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
| Routed MoE assignment grouping | The compact reference path copies GPU top-6 to CPU to form a chunk union. The resident optimized path instead keeps route IDs/weights on device and stable-sorts all `(token, slot)` assignments by expert. It does not yet form a compact non-empty tile list. | Flattens routes, GPU-sorts by expert, gathers token rows in sorted order, and builds native block metadata for grouped QMM. | Router output stays on device. A scatter kernel builds per-expert `(token, slot)` lists and a compact work list containing only non-empty expert tiles. |
| Expert batch shape | Three gathered QMMs over `[T*6, 1, K]`, `T <= 128`; each selected assignment is a row. | Variable `N_e` rows per expert after sorting, tiled by the selected native block variant. | Variable `N_e` per expert, dispatched as non-empty 32-row tiles; output is scattered back to fixed token/slot destinations. |
| Routed gate/up/down unit | Separate gathered gate, up, down operations over all assignments, with activation split into assignment chunks and a final route-reduction kernel. | Native grouped path can pair gate/up, apply SwiGLU/route weight, execute grouped down, then combine sorted routed outputs. | One routed-MoE batch encoding: expert-grouped gate/up with SwiGLU epilogue, grouped down, and device reduction. Quant-specific math differs, but the scheduling unit is reusable. |
| Expert weight runtime layout | The reference/low-memory path still creates a compact `[union, rows, packed-K]` bank. The optimized 512 GiB path builds the official packed tensors for all 40 layers once without modifying the checkpoint and shares the immutable atlas. | Full-resident converted checkpoints use persistent stacked `SwitchLinear` tensors. Optional SSD offload uses fixed-capacity slot tensors, not a new shape/layout per request. | GGUF stores each layer's routed tensors in a conversion-time layout with stable offsets. Kernels bind the mapped tensors directly; no route-union repack exists in resident mode. |
| Weight/bank preparation lifetime | Compact banks retain layer-call lifetime. The optimized atlas is built transactionally during model initialization, published only after all 40 banks succeed, and then lives for the model lifetime. | Resident layout lives for the model lifetime. Offload plan/readers and expert slots also live for the model lifetime and retain hits across calls. | Model mmap, tensor descriptors, buffers, and resident weight views live for the engine lifetime. Streaming mappings/caches are also engine-owned. |
| Per-request bank/layout construction | Reference compact mode: yes. Optimized resident mode: no; the reviewed 17-chunk run performed 40 model-construction banks and zero request-path constructions. | No in resident mode. Offload may replace slot contents on route misses, but its layout is allocated once. | No in full-resident mode. Streaming mode stages/maps admitted ranges but does not construct a new packed expert bank per request. |
| Shared and routed experts | Shared expert and routed bank are separate lazy branches, then added. They share the component evaluation but have no explicit joint schedule or fused combine. | Shared and routed branches are built in one lazy graph; a custom combine kernel restores token order, sums six routed rows, and adds the shared row. | Shared gate/up/SwiGLU/down and routed grouped MoE are encoded in the same layer command buffer, then device-added/reduced before the next mHC stage. |
| Inter-layer materialization/eval | Each block returns a materialized boundary in practice; profiling adds three hard synchronizations per layer. Bank release and CPU route/index consumers prevent a graph spanning layers. | The model builds a lazy graph across its layer loop and the scheduler evaluates cache states once per admitted chunk. Some bounded index-top-k and SSD-offload paths deliberately introduce internal evals. | Activations stay in preallocated GPU workspaces. Normal V4.1 prefill drains at a layer/chunk boundary; no token/expert boundary is required. Wide carry buffers bridge chunk partitions. |
| Metal command submission | MLX owns submission. Current host readbacks and `mx::eval` boundaries split the graph; submission structure is not a first-class runtime contract. | One engine stream and one outer `mx.eval` per scheduler chunk, with custom primitives embedded in that graph. | Explicit `begin_commands`/`end_commands`. Attention projection/core/output, mHC, shared/routed FFN, and expand are encoded into the layer/chunk command buffer. |
| CPU readback | The resident path has zero route or index-result readbacks when their diagnostics are disabled. Top-k rows and candidate masks remain device-authoritative through publication and attention gather; final logits still reach the host and the token-serial attention loop remains unresolved. | Resident normal path keeps route sorting, index IDs, attention, and recurrent arrays on GPU. CPU readback appears in cache metadata and the optional expert-offload `ensure` path. | Route IDs, expert lists, index selections, and state remain device-resident. CPU sees progress/cancellation and final logits/snapshot copies, not per-layer routing/index results. |
| Attention/indexer batching | Q/K/V projections and score matrix are partly batched. Index score/mask/top-k are chunk-wide, but results are copied to CPU; sparse attention and output projection then loop token by token. Pure SWA also loops token by token. | Chunk-wide Q/KV projection, packed index score/top-k, and native packed sparse attention. Query rows are dispatched together; selected indices remain array values. | Batched projection and publication, packed index score/top-k, and batched raw/mixed/indexed attention. Final ring/publication state is copied in bulk. |
| Recurrent/compressed state publication | Copies the C++ state at entry, constructs one `SharedAttentionReference` per token, mutates a publication vector across reuse layers, and swaps state only after successful evaluation. This is atomic but host-object-heavy. | Request-local cache arrays hold window KV, compressed KV, index K, compressor tails, and Engram history. Lazy cache mutations are forced at the scheduler chunk boundary. | Preallocated device tensors receive bulk window/compressed/index publication. A partial layer sweep marks the graph invalid; position/history become valid only after the complete sweep. |
| CED/deferred decoder prefill | Absent: every chunk runs encoder layers 0..19 and decoder layers 20..39. | Absent at the pinned revision; [oMLX issue #3605](https://github.com/jundot/omlx/issues/3605) identifies it as missing. | Present. Large sweeps can run encoder-only and later rebuild only the decoder's exact dependency suffix; decoder layers also skip prefixes outside their required suffix. |
| Continued-prefill fast path | Reuses persistent state but always re-enters the same 128-token full-stack path. | Cached suffixes use the same chunked prefill machinery (normally 2,048 rows); no decoder deferral. | Resident and TP modes automatically batch continued prefills; very short appends retain a lower-latency short path, while long appends use the layer-major/deferred schedule. |
| Full-resident I/O | The optimized path reads and directly packs all routed experts once during model construction; warm prefill has no routed checkpoint reads or bank construction. Engram remains mmap-backed. The reviewed run stayed swap-free but caused about 32.6/32.5 GiB of VM compression/decompression, so memory pressure remains explicit. | Main tensors are resident for the model lifetime; only optional expert/Engram offload performs demand I/O, with Engram prefetch support. | Main GGUF weights are directly mapped and resident on a large Mac, so routed-weight I/O disappears. Engram deliberately remains disk-backed and is prefetched/pipelined. In streaming mode, next-layer reads overlap current-layer compute. |
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
| 1 | Model-owned, transactionally published 40-layer resident expert atlas | 41.772 s bank construction | **Connected and full-path observed:** 40 model-lifetime banks; zero warm-chunk construction/read. A literal second fresh request on the same model remains a narrow qualification item. |
| 2 | Chunk-wide mHC pre/post and state expansion | 38.743 s post-MoE, plus part of 9.678 s overhead | **Connected and full-backbone qualified:** two 128-token chunks match the token-serial oracle through logits/state/publication. |
| 3 | Device route to expert-major work lists and grouped gate/up/down/reduce | Resident profile: 53.882 s MoE | **Schedule promoted, grouped tile candidate rejected:** device routes and stable expert-major assignment order are connected. Direct oMLX grouped MXFP4 dispatch was slower for the official 768-assignment shape, so the current gather-QMM schedule remains pending a better full-path candidate. |
| 4 | Chunk-wide attention/index/publication with atomic frontier commit | Shape-bucket profile: 46.708 s attention (from 69.652 s) | **In progress:** device top-k/candidate arrays remain authoritative through publication and attention gather; equal-shape AV work is batched and host vectors are diagnostic-only. Scalar QK, residual token-serial attention, and chunk-atomic frontier commit remain. |
| 5 | CED/deferred decoder and bounded replay | No 2K saving | Begin only after the 2K structural gates above. |

The prior full-resident experiment proved 40-bank reuse but built those banks
during the first prefill chunk and experienced substantial VM compression.
Phase 1 therefore separates model-initialization cost from warm request wall
and keeps memory pressure as an explicit promotion gate; it does not reinterpret
the earlier run as production qualification.

Reviewed results (2026-09-17):

- `expert-bank/layer-major-backbone-20260917-003137-31497` passed both
  128-token chunks across all 40 layers with bit-exact hidden, pre-mix, logits,
  state, publication, hash, and route-tie observations, plus invalid-token
  atomicity.  This closes the Phase 2 full-backbone correctness gate; its
  two-model/80-bank wall is not performance evidence.
- `context-ladder/32k-run-20260917-003631-31787` constructed exactly 40 banks
  during model initialization (26.154 s model construction; 16.225 s bank
  total), then performed zero bank construction/load in every one of 17
  request chunks.  Prefill was 127.717 s / 16.153 tok/s versus the reviewed
  compact baseline's 225.927 s / 9.131 tok/s: 43.47% less wall time and 1.769x
  throughput in these single observations.  It used 680 device route batches,
  680 expert-major batches, 495,120 assignments, and zero route diagnostic
  readbacks; the generated token remained 339.
- The resident run was below its 340 GB budget and had no swap.  Peak MLX was
  302,822,327,564 bytes and process peak footprint was 305,189,124,288 bytes,
  but VM counters show about 32.6 GiB compressed and 32.5 GiB decompressed.
  The atlas is therefore a viable target-machine path, not evidence of ample
  memory headroom.  A second fresh request on the same model is still needed
  to close that exact Phase 1 qualification clause.

Phase 3's first production schedule omits route diagnostic readback,
stable-sorts all `6T` assignments by expert on device, runs gate/up/down in
that order, and maps the canonical expert-ID reduction order through the
inverse permutation.  A 128-token/768-assignment layer-0 fixture matched the
assignment-major and token-serial accumulated and BF16 routed outputs bitwise.
The reviewed resident component profile
`context-ladder/32k-run-20260917-005021-32331` was produced from clean commit
`7567138` and passed its identity and exit checks.  Its 2,063-token prefill was
126.641 s / 16.290 tok/s.  Additive 40-layer GPU-completion wall was 126.130 s:
Attention 69.652 s (55.22%), MoE 53.882 s (42.72%), and Post-MoE 2.390 s
(1.90%).  It again recorded 40 initialization banks, zero warm constructions,
680 device routes, zero route readbacks, and 680 expert-major batches.  The
component synchronizations perturb normal lazy execution, so the 126.641 s is
an attribution observation rather than a new performance acceptance result.

Two bounded grouped-dispatch candidates were then tested against the same
official 128-token/768-assignment shape.  A new dynamic Metal tile kernel was
both numerically wrong and slower and was removed.  The directly adapted oMLX
grouped MXFP4 primitive had small non-bitwise differences but was also slower:
11.15 ms single / 10.98 ms paired gate-up versus 9.57 / 9.55 ms for the current
expert-major gather-QMM schedule.  Since the full-path profile now makes
Attention the largest bucket, neither local MoE candidate is promoted.

Phase 4 now carries top-k relative row IDs and candidate masks as MLX arrays
from index selection through `SharedAttentionReference`, republish, candidate
consumer, and main-KV gather.  `DSV41_RUNTIME_INDEX_DIAGNOSTICS=1` preserves
the host-visible oracle/tie path; the resident production runner sets it to
zero and treats any index-result readback as an error.  This removes a host
boundary but does not yet remove the per-token attention invocation or provide
an atomic full-chunk frontier commit.

The clean full-path observation
`context-ladder/32k-run-20260917-084044-36764` reviewed commit `9bfeb71`
with exit 0, an empty tracked patch, and all recorded identities matching.
It retained token 339, 40 model-initialization banks, zero warm bank
construction, zero route readbacks, and **zero index-result readbacks**.
Prefill was 129.090 s / 15.981 tok/s, 1.93% slower than the preceding
126.641-second observation; this single-run difference is not credited as a
speedup. Peak MLX was 302,822,401,315 bytes, process peak footprint was
305,053,726,576 bytes, swap remained zero, and VM compression/decompression
was about 30.80/30.66 GiB. The result qualifies the removed host boundary,
not Phase 4 performance.

Code-level comparison with DwarfStar's current `ds41_attention_batch` confirms
that its prefill path publishes/selects in batches, executes raw/mixed/indexed
attention with batch kernels, and commits the final ring only after the batch.
Its focused batch fixture requires exact state/index publication and bounded
attention output error (`relative RMS < 0.002`) rather than scalar/batch bit
identity. The next candidate follows that split: reuse-layer rows share one
padded MLX graph, while the token-serial path remains the default oracle.

The next long validation is the 40-layer semantic promotion gate. It has a
reproducible runner because it takes several minutes:

```sh
bash tools/benchmark/run_chunk_attention_backbone_check.sh
```

It compares a token-serial and chunk-attention model over two 128-token chunks
and all 40 layers. Hidden/pre-mix/logits must remain finite and below the
pre-fixed relative-RMS 0.002 bound, every logits argmax must match, and route
ties plus all persistent state/publication/hash bytes must be exact. It may
read about 578 GB logically while constructing 80 comparison banks, uses a
240 GB Unified Memory budget, retains failed logs under `artifacts/attention/`,
and has no resume. It is not passed until reviewed.

The first run of that gate,
`attention/chunk-backbone-20260917-085608-37522`, was reviewed and **rejected**.
It was a clean `62517e2` run (empty tracked patch and matching identities),
but the first 128-token chunk already produced hidden relative RMS
`0.00554266`, max absolute error `2048`, mean absolute error `1.02469`, and
2,595,308 BF16 mismatches.  That exceeds the pre-fixed `0.002` bound, so the
run stopped before state/route qualification.  It used no swap and cannot be
used for a performance conclusion.  The padded rank-3 MLX matmul candidate
therefore remains disabled and is not a production path.  A follow-up attempt
to apply MLX `vmap` to the scalar operator failed in the short fixture because
the operator graph is not vectorizable in MLX 0.32.2; the gate is not relaxed.

The follow-up DwarfStar-style Metal candidate assigned one threadgroup to one
token and eight heads. Its clean run
`attention/chunk-backbone-20260917-092514-39493` was also **rejected**: chunk 0
hidden relative RMS was `0.0055142` (max `2048`, mean `0.845199`, 2,587,396
BF16 mismatches), above the unchanged `0.002` gate. The run used no swap and
stopped before discrete/state qualification, so its 103.71-second harness wall
is not a performance result. The rejected kernel and generated build plumbing
were removed.

MLX v0.32.2 source inspection localized the numerical split: scalar QK at
`M=64, K=512, N<=512` selects Steel split-K, whereas rank-3 batch QK selects a
regular GEMM. AV blocks select regular GEMM in both cases. The replacement
hybrid therefore keeps QK and AV token-wise for now, but batches the surrounding
softmax graph by equal execution shape. Tokens are bucketed by raw offset and
selected-row count instead of padding every token to the chunk maximum. The
official layer-2-to-3 fixture is bit-exact for both positions 0--127 and
128--255, including output and state. Telemetry explicitly counts the retained
scalar QK/AV calls; this candidate still requires the unchanged 40-layer gate.

That gate is now closed by clean run
`attention/chunk-backbone-20260917-113353-41239` at commit `731c6aa` with an
empty tracked patch and matching identities. Both 128-token chunks matched
hidden, pre-mix, and logits bitwise; route ties and all state/publication/hash
checks were exact, including invalid-token atomicity. Maximum RSS was
160,473,219,072 bytes, peak footprint 165,164,675,696 bytes, and swap remained
zero. The 227.03-second two-model harness wall is not a production performance
result. After that gate, AV was batched only within an equal-shape bucket. The
same official 0--127 and 128--255 fixtures remained bit-exact for output and
state, while scalar Steel split-K QK was left unchanged. The next measurement
uses one resident model with component boundaries and records shape buckets,
retained scalar QK calls, and AV batches. This AV extension remains a candidate:
if its component profile warrants retention, it must repeat the 40-layer gate.

The reviewed resident component run
`context-ladder/32k-run-20260917-120246-41847` was clean at `cbca6c1`, exited
zero, had an empty tracked patch and matching identities, and used no swap.
Prefill fell from 126.641 s to 107.299 s (15.27%), while Attention fell from
69.652 s to 46.708 s (32.94%) under the same component-synchronization
semantics. It retained 613,439 scalar QK calls but recorded zero scalar AV calls
and 117,579 AV batches. Model construction remained 40 banks, warm construction
and route/index readbacks remained zero, and decode still produced token 339.
The AV extension is therefore retained as a performance candidate and must now
repeat the unchanged 40-layer correctness gate.

That repeated gate is closed by
`attention/chunk-backbone-20260917-121232-42128`: clean `130fcf6`, exit zero,
empty tracked patch, and matching identities. Both 128-token chunks matched
hidden, pre-mix, and logits bitwise, while route ties, all state/publication/hash
checks, and invalid-token atomicity were exact. Maximum RSS was
160,476,315,648 bytes, peak footprint was 165,187,973,744 bytes, and neither
swap nor compression increased. Its 194.38-second two-model harness wall is not
a performance result. Chunk attention is now explicitly connected to the
resident optimized lazy-schedule runner; the global default remains unchanged
until that full-path observation is reviewed.

The next loop should be architecture-first and preserve the project's stated
correctness priority:

1. Qualify device index/candidate publication in the full resident path and
   confirm zero normal-path index readback without changing logits/token 339.
2. Replace the token-serial attention body and state publication loop with a
   chunk graph and atomic frontier commit.  DwarfStar's full-sweep publication
   remains the ownership reference; do not begin with the SWA-only kernel.
3. Retain DwarfStar's
   [`kernel_mul_mm_id_map_scatter_work`](https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/metal/moe.metal#L7790-L7950)
   as the scheduling reference; oMLX's
   [`switch_layers.py`](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/patches/deepseek_v4/switch_layers.py)
   and grouped
   [`deepseek_moe.metal`](https://github.com/jundot/omlx/blob/b390b31e0c6831225fed0f24d278eb1db7fcb68b/omlx/custom_kernels/glm_moe_dsa/csrc/deepseek_moe.metal)
   show the directly compatible sorted-row/block-metadata alternative for the
   official packed expert tensor shapes.
4. Keep shared expert, routed gate/up/down, route weighting, and canonical
   reduction in one device graph.  Preserve the existing diagnostic oracle,
   but do not add host work-list construction.
5. Add wider/continued prefill and, for 16K+, the ds4 decoder-suffix/CED
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

## 2026-09-17 gap-audit checkpoint

The production-shape lazy run
`context-ladder/32k-run-20260917-123159-42514` was reviewed clean at
`7a7c66a`: 2,063-token prefill was 106.233166 seconds / 19.41955 token/s,
next token remained 339, warm bank construction and route/index readback were
zero, and swap remained zero.  This is a full-path observation, not a paired
oMLX qualification.

The next action is now governed by the
[execution-work gap audit](prefill-gap-audit.md).  Its dynamic counters and
pinned-source comparison identify simultaneous multipliers: 17 request chunks
/ 680 layer visits; 594,144 one-row dense packed QMMs plus 181,544
token-serial FP GEMMs; 2,040 routed gather-QMMs; 613,439 scalar QK plus 117,579
AV batch invocations; and 340 blocking eval calls plus 17 explicit
synchronizations in the canonical measured prefill.  The initial 12 GB Metal
System Trace did not finalize into an exportable document and is explicitly
rejected in the audit.  Both `Metal Application + GPU` and `Metal Application`
alone showed the same failure mode after target exit.  The retained
process-local counter confirmation measured 269,115 command buffers, 237,345
compute encoders, and 12,358,908 prefill compute dispatches, or 5,990.7
dispatches/token and 18,174.9 per 128-token layer visit.  The replacement
runner therefore launches no Instruments trace and gates command-buffer,
encoder, and dispatch selector counters to prefill.  A paired pinned-oMLX
runner applied the same counter to one 2,063-row official-checkpoint sweep and
measured only 1,595 command buffers, 918 compute encoders, and 7,463 dispatches.
The current/oMLX ratios are therefore 168.7x, 258.5x, and 1,656.0x
respectively, directly explaining the observed 9.75--9.89x wall gap.

The first implementation after audit closure introduces a transactional
request-wide layer-sweep API.  It preserves 128-row primitive contracts while
reversing `17 chunks × 40 layers` into `40 layers × 17 internal microtiles`,
retains each layer's expert layout across the request, and publishes copied
persistent state only after the full sweep succeeds.

That gate passed cleanly in
`prefill-gap/layer-sweep-backbone-20260917-222206-47915`: hidden, pre-mix, and
logits were bit-exact; route ties, state/publication/hash, logits argmax, and
invalid-request atomicity were exact.  The gate constructed 40 banks and used
no swap.  The optimized generation API now selects bounded 4,096-token sweeps
only with `DSV41_RUNTIME_LAYER_SWEEP=1`; its default and its one-token decode
path remain the reference schedule.  The next full-path observation is
`bash tools/benchmark/run_layer_sweep_prefill_measurement.sh`, with resident
atlas and qualified chunk attention enabled and batched dense QMM explicitly
disabled so the schedule inversion is measured in isolation.

That full-path observation completed cleanly in
`context-ladder/32k-run-20260917-223115-48574`.  Prefill fell from 106.233166
to 65.403335 seconds (38.43%) and throughput rose from 19.41955 to 31.54273
tok/s.  Token 339, 40 model-lifetime banks, zero warm construction, zero
route/index readbacks, and no swap were preserved.  The remaining gap to the
paired oMLX hook run is about 6.0x.  Invocation telemetry is not yet reduced:
680 device MoE batches, 2,040 expert QMMs, 613,439 scalar QKs, 117,579 AV
batches, and 12,378 token-serial attention calls remain.  Therefore the next
decision uses the same-sweep component profile rather than immediately adding
a local kernel.

That synchronized profile (`32k-run-20260917-224750-48769`) attributes 41.180
of 60.379 component seconds (68.2%) to attention, versus 17.838 seconds (29.5%)
to MoE and 1.130 seconds (1.9%) to post-MoE. This closes the component gate
required before a kernel change. The first one-dispatch attention candidate
failed the fixed full-backbone gate because its SIMD reduction diverged from
the scalar Steel split-K oracle (`pre-mix RMS 0.00896325`) and was removed.
The replacement preserves MLX 0.32.2's exact per-token BM32/BN32/BK16,
WM2/WN2 split-K reduction for complete 64-key blocks while batching the
independent token matrices into two dispatches per block. Short tails remain
on native scalar Steel because the separately compiled tail reduction was not
float32-exact. Existing softmax and qualified batched AV remain
unchanged. Synthetic boundary cases and official layer-3 128-token fixtures
are bit-exact. Clean full-backbone run
`attention/batched-splitk-qk-backbone-20260918-122034-57470` then passed both
128-token chunks with hidden, pre-mix, and logits bit-exact; route ties,
persistent state, publication, hashes, continuation, and invalid-request
atomicity were exact. It replaced 21,937 full-width token loops with batched
calls while retaining 8,583 scalar tail calls. The default remains off until
the isolated full-path wall result is reviewed.

The reviewed isolated 2,063-token result
`context-ladder/32k-run-20260918-122624-57828` is 62.619395 seconds / 32.9451
token/s, versus the paired scalar-QK layer-sweep baseline's 65.403335 seconds /
31.5427 token/s: 2.783939 seconds or 4.26% less wall. Scalar QK calls fell from
613,439 to 24,711, with 100,428 batched full-width calls. Since both native
scalar and batched paths use split plus ordered accumulation, this implies
approximately 1,226,878 versus 250,278 QK Metal dispatches, a 4.90x dispatch
reduction. The much smaller wall reduction proves QK was only one part of the
attention amplification; 17,910 shape-group calls and 117,579 AV batches
remain. Against pinned oMLX's 10.901336 seconds, the remaining full-path ratio
is 5.74x.

The follow-up synchronized component run
`context-ladder/32k-run-20260918-124010-58187` completed at clean commit
`601629e`. Its synchronization-perturbed prefill was 59.934584 seconds / 34.4209
token/s. Attention fell from 41.179713 to 37.928151 seconds (3.251562 seconds,
7.90%), while MoE was 18.251121 seconds and post-MoE 1.100472 seconds.
Attention still owns 65.9% of the 57.544021-second summed layer wall. The run
retained 17,910 shape groups, 100,428 batched full QK calls plus 24,711 scalar
tails, and 117,579 AV batches. Separate QK batching has therefore reached its
architectural limit; the next boundary is the attention operation.

## Fused prefill-attention topology decision

The pinned code-level comparison used DwarfStar/ds4 commit
`8db1d1d155cb0400a86a86b9c62d0defb3a6148b` and oMLX commit
`b390b31e0c6831225fed0f24d278eb1db7fcb68b`.

DwarfStar's `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` sorts a
chunk's selected rows on device and then dispatches
`kernel_dsv4_indexed_mixed_attention_heads16_dual` on a
`[token, ceil(head/16)]` threadgroup grid. Each threadgroup stages a KV row
once for 16 heads and performs QK, validity handling, online max/sum, value
accumulation, and sink normalization without materializing an attention
matrix. The dispatch topology is applicable, but the reviewed kernel casts
Q/K/V to FP16 and is not directly admissible under the official BF16/FP8
precision invariant.

oMLX's `deepseek_v41_packed_attention` is compatible. It uses one 256-thread
threadgroup per token, processes all 64 width-512 heads, tiles 64 KV rows at a
time, reads packed rows directly, keeps the online denominator in FP32, and
rounds probabilities to BF16 only at the PV boundary. Its pooled row is 256
packed value bytes plus 32 E4M3 scale bytes, exactly the runtime's
`GlobalKVState` layout. No QK matrix, probability matrix, or decoded pooled
cache is published.

The adapted candidate is behind `DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=1`. It
retains the current local window's official quantization-round-trip BF16
representation, consumes persistent pooled value/scale buffers in place, and
pads the per-token device-selected rows to the chunk maximum in one
`[tokens,1..512]` work list.
QK, causal/global masking, online softmax, BF16-rounded PV, AV, and sink
normalization execute in one Metal kernel. Scalar and decomposed chunk paths
remain qualification oracles. The adapted source retains upstream
Apache-2.0 attribution and license.

For a 2,063-token layer sweep, the schedule changes from 17,910 host shape
groups, approximately 250,278 QK dispatches, 117,579 AV batches, and 12,378
producer token-serial calls to at most 38 compressed layers x 17 chunks = 646
fused attention dispatches. That is a 569x reduction against QK+AV dispatches
alone and removes decoded pooled-row gather/materialization. It remains 17x
more attention dispatches than oMLX's one 2,063-token call per compressed
layer; widening the surrounding 128-row projection/state contract is a later
scheduling step.

The short Metal fixture passed: local-only output was bit-exact; a packed
pooled-row case measured relative RMS 0.000210066 and maximum absolute error
0.00390625, below the fixed 0.002 semantic gate. This is component evidence,
not full-backbone qualification. Run the prepared 40-layer gate first:

```sh
bash tools/benchmark/run_packed_attention_backbone_check.sh
```

Only after review of that result should
`tools/benchmark/run_packed_attention_prefill_measurement.sh` be run.
