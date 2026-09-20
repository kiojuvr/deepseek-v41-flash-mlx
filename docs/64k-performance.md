# 64K production performance milestone — 2026-09-21

Current review checkpoint: the bounded decode-stack experiment below improved
advancing decode mean from 3.37326 to 3.24626 seconds (-3.765%, seven of seven
tokens faster), with exact bounded state/logit parity and zero swap deltas.
This is useful evidence for the scheduling change, but falls far short of the
practical 64K milestone. At the user's request, commit this work for review
before proceeding; hold the proposed 64K measurement and all promotion.

Reviewed artifact: `artifacts/context-ladder/decode-stack-20260921-060113-18060/`.
Its `bounded/review.md` records raw-log and identity checks. Full-path wall was
68.19991 / 66.89917 seconds (-1.907%); prefill also varied despite being outside
the candidate scope, so one fixed-order pair does not establish a stable gain.
Peak footprint was 310.844 / 310.849 GB. The candidate deferred 280 layer
evaluations into 14 stack evaluations. Comparison JSON uses a floor-index p95
estimator; canonical result p95 was 3.47131 / 3.34653 seconds. Reviewers should
not mix these definitions or infer a robust tail estimate from seven tokens.

Committing changes HEAD and the tracked-patch identity used by the runner.
The pre-commit artifact remains historical evidence; its old resume command
will intentionally fail identity verification after this commit. Do not rewrite
the saved identity to bypass that check. A later measurement needs a new root
or an explicitly reviewed content-equivalence mechanism.

This priority supersedes historical ladder-next-step recommendations. Suspend
128K / 200K / 256K cumulative qualification until practical 64K performance is
reviewed as achieved. No numerical usability threshold has yet been agreed;
correctness PASS and isolated kernel wins do not establish usability.
Rejecting a candidate means selecting the next dominant **64K** structural cost.

## Frozen baseline

Historical baseline (read-only):
`artifacts/context-ladder/cumulative-long-20260920-201224-10109/64k-cumulative/attempt-001/`.
Revision `37bb734`; its `tracked.patch`, `identity.txt`, config, environment and
command files, rather than today's dirty worktree, define the measured build.
Result SHA-256: `3b1ec2efa58da0af7fd25515a021b029e97a85c9f68955a3d373b58d0c52e8a7`.
Tracked patch SHA-256: `ac33a822e96bb1235a9089470d33cbba7f044a99e4e0ff44dcd304115e240d62`.
Use new baseline/candidate pairs for causal comparison; never silently replace
this historical artifact with a new measurement.

The baseline is the resident 40-bank official precision path, device route/index
selection, packed layer sweep, fixed-tile/ragged attention, pending CED enabled,
16 GiB long-context idle cache limit, no component profiling. Grouped expert
pipeline and decode stack graph remain OFF. Checkpoints stay read-only.

32K comparison: same root, `32k-cumulative/attempt-002`. Each turn appends
8,128 fixture tokens and generates 64 greedy tokens. Subsequent appends also
commit the preceding turn's final token. First-token latency is from existing
prefill output, so advancing decode must also be reported separately.

| Metric | 32K / 4 turns | 64K / 8 turns | Scaling |
|---|---:|---:|---:|
| Session wall (s) | 1488.788 | 3003.068 | 2.017x |
| Append wall (s) | 635.847 | 1289.451 | 2.028x |
| Decode wall (s, includes first tokens) | 852.882 | 1713.574 | 2.009x |
| Decode mean (s) | 3.33157 | 3.34682 | 1.0046x |
| Decode p95 (s) | 3.49914 | 3.50673 | 1.0022x |
| Indexer rows | 3,489,529,856 | 13,958,381,568 | 4.0001x |
| Fixed-tile calls | 9,728 | 19,456 | 2x |
| Batched split-K QK / AV calls each | 97,280 | 194,560 | 2x |
| Device route batches | 20,320 | 40,640 | 2x |
| Recorded cache copy bytes | 600,041,472 | 1,200,875,520 | 2.0013x |
| Peak MLX bytes | 305,920,050,683 | 305,951,311,355 | ~1x |
| Compression / decompression pages delta | 2,334,736 / 2,310,189 | 2,319,110 / 2,291,378 | ~1x |
| Pending CED chunks / swap delta / host readbacks | 0 / 0 / 0 | 0 / 0 / 0 | unchanged |

The indexer does superlinear work; it must remain on the optimization queue.
It does not yet dominate observed wall: 64K turn-1 to turn-8 append rises 4.71%,
decode mean 0.82%. No pending CED transaction is opened by these 8K appends;
decoder suffix replay is not the source of this workload's cost.

Missing measurements are explicit: historical cumulative runs do not contain
encoder/decoder/component wall splits, actual Metal dispatch counts/durations,
allocator timings, or per-phase compression deltas. Logical counters are not
Metal dispatch counts. The short 2K + seven advancing-token synchronized profile
attributes 90.56% to the MoE path, 7.01% to attention, 2.44% to post-MoE; this is
evidence for prioritizing decode/MoE, **not a measured 64K component breakdown**.
Do not infer that 90.56% is solely expert arithmetic: the bucket evaluates lazy
dependencies including mHC/norm preceding MoE.

## Next structural candidate

The grouped-expert pipeline five-pair table shows four decode mean regressions
and one tiny win. It provides no promotion evidence. Its pre-existing opt-in
implementation is preserved, default OFF; this work does not rerun it or return
to 128K. Full raw-log certification of that experiment is separate.

Pinned architecture references already available locally:
oMLX `b390b31e`, `language.py` layer loop 895–930, builds across layers with
optional asynchronous Engram boundaries; DwarfStar `8db1d1d1` has resident decode
graphs and explicit command-buffer submission. See
[the architecture comparison](execution-architecture-comparison.md) for source
anchors and precision differences. Neither comparison is an equal-weight speed
claim. This candidate transfers scheduling; it imports no arithmetic kernel.

Current packed sweep synchronously evaluates each of 40 layer outputs and two
Engram outputs even for one-token decode. `DSV41_RUNTIME_DECODE_STACK_GRAPH=1`
defers those explicit boundaries and evaluates at the encoder and decoder exits.
It retains all projection/reduction/route arithmetic and private transactional
state copies. Each stack completes before publishing its local state; the
backbone commits the caller's state/revision only after both complete. Wider
prefill and nonresident paths retain layer materialization.

This is a testable submission/temporary-lifetime hypothesis, not a demonstrated
explanation of three-second TPT. It does not promise a single Metal command
buffer: MLX and lower-level state operations may still synchronize. Telemetry
counts explicit sweep evaluations and deferred evaluations separately. Lazy
graph growth may increase live memory, which is a rejection criterion.

## Reproducible gates

Start with:

```sh
bash tools/benchmark/run_decode_stack_graph.sh bounded
```

Approximately 10–20 minutes, one model at a time, up to 340 GB Unified Memory.
It first compares four state-advancing tokens after a 129-token prefix, including
hidden/pre-mix/logits, persistent state, device publication, hash continuation,
revision and invalid-request atomicity. It then runs fresh baseline/candidate
2,063-token prefill + eight generated-token full-path measurements. A bounded
pass cannot promote the candidate or establish 64K performance.

After reviewing that result, use the printed root:

```sh
DSV41_DECODE_GRAPH_DIR=artifacts/context-ladder/decode-stack-<timestamp>-<pid> \
bash tools/benchmark/run_decode_stack_graph.sh 64k
```

This runs the frozen eight-turn workload in two sequential fresh processes,
approximately 100–140 minutes; no 128K stage exists. Logs live under
`<root>/parity`, `<root>/bounded/{baseline,candidate}`, and
`<root>/64k/{baseline,candidate}`. Resume uses the same command/root, checks
source/binary/input identities, skips completed stages, and archives failed
measurement stages before retrying with fresh model state. Inspect raw results,
resource/system logs and compression/decompression deltas before accepting any
comparison. The comparison checks generation, positions, candidate invocation,
40 banks, topology/readbacks, footprint and process/system swap. It reports
full-path wall and advancing mean/p95; it never auto-promotes.

Promotion still requires the existing full-path correctness/memory contracts,
late-context stability and repeated performance evidence beyond run-order noise.
If rejected, next investigate shared versus routed MoE service time and actual
GPU/host submission gaps, then transfer resident scratch/command ownership from
the reference architectures. Indexer quadratic work remains a separate measured
scaling concern. No rejected candidate reopens the qualification ladder.
