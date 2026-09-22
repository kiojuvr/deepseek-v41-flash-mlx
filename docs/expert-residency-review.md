# Routed expert residency review — 2026-09-22

## Decision

Select **explicit, bounded MLX residency budget owned by the runtime lifecycle**
as the next structural candidate. Do not optimize GatherQMM arithmetic, add broad
instrumentation, or launch another full-model attribution run. This is a
code-supported candidate, not a demonstrated speedup or production promotion.
The 64K milestone and frozen baseline remain unchanged; the longer qualification
ladder remains paused.

## Evidence and measurement caveats

Reviewed latest result/resource/system logs:
`artifacts/context-ladder/decode-component-profile-20260922-025339-39036/`.
Preflight passed (486 GiB free+inactive, no other process above 32 GiB RSS).
The runner exited 0; seven advancing tokens exercised 280 layer calls.
Layer wall was 26.0153 s, routed service 21.0819 s, immediate routed repeat
0.242902 s, shared expert 1.95435 s, and attention 1.56676 s.
The reported routed/repeat ratio is 86.79x. This strongly prioritizes a
cold/residency mechanism over intrinsic M=1 arithmetic throughput.

Two important limits of that comparison are visible directly in the code:

1. `src/moe/reference.cpp::forward_batch_components` records `MoeRouted`
   **after** the warm repeat. Thus 21.0819 s includes the 0.242902 s repeat;
   the first service plus surrounding overhead is approximately 20.8390 s,
   not a separately measured 21.0819 s cold call.
2. The first service forces five stage evaluations; `diagnostic=true` skips
   those boundaries for the repeat. This is not an identical-schedule cold/warm
   A/B. The tiny no-op sync time does not by itself bound the effect of changing
   graph evaluation boundaries. The repeat rebuilds the operations using the
   same bank/routes/input, rather than simply evaluating the original output.

The nominal gate/up timer evaluates **only `up`**. The independent `gate` QMM
is not a dependency of `up`; its evaluation is required by the next
`down_input` boundary. Consequently:

- `routed_gateup_seconds` (6.49331 s) does not guarantee both QMMs completed;
- `routed_mid_seconds` (7.18385 s) includes the previously unevaluated gate QMM,
  not just SwiGLU/FP8 work;
- `routed_down_seconds` (6.92715 s) includes down QMM and weighting.

This explains why the ostensibly weight-free middle bucket is expensive
without requiring a slow SwiGLU kernel. Likewise the preparation boundary
only evaluates sorted RHS IDs, not every take or the input FP8 conversion.
Do not use the current stage labels as exact operator attribution.

The latest system-before/after counters show 2,150,874 decompressed pages
(35.24 GB / 32.82 GiB at 16 KiB/page) and 2,153,009 compressed pages, with
zero system swapin/swapout deltas. These counters cover **the entire process
interval**, not decode alone, and are system-wide. The previously reported
~34 GB decode decompression versus ~30 GB expert reads is supporting evidence
only; the latest endpoint logs cannot independently establish that phase-local
claim or identify the decompressed pages as expert weights. Process peak
footprint is 311.895 GB. Zero swap does not rule out compression.

## Ownership, mapping, materialization and storage comparison

### Current runtime

- `src/moe/expert_bank.cpp::load_bank` allocates final weight/scale storage
  directly through `mx::allocator::malloc`, then `TensorFile::read` fills it.
  `src/model/weights.cpp:55` implements this with read-only `pread`; the bank
  is **not** a live checkpoint mmap view.
- The final shapes are `[384,N,K/8]` uint32 weights and `[384,N,K/32]` uint8
  scales. The constructor transfers those buffers into arrays and calls
  `mx::eval` once. No lazy `stack` or concatenation of expert weights is
  retained in this path.
- `ResidentExpertAtlas` constructs 40 immutable banks once and returns them by
  const reference. Decode passes the persistent member arrays to gather-QMM;
  it does not reload, repack or reconstruct the bank. Ownership is genuinely
  model-lifetime, but ownership alone is not physical residency.
- MLX 0.32.2's `MetalAllocator::malloc` obtains shared Metal buffers and calls
  `residency_sets_.insert(buf)`. **Insert is not sufficient**: the residency
  manager's `capacity_` defaults to zero and leaves over-budget buffers tracked
  but outside any residency set.
- No `set_wired_limit` call exists in this runtime. Existing cache limits
  control unused allocator buffers, not the standing residency budget.

### MLX source anchors and identity

Local clean tag `v0.32.2`, revision
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`, inspected at
`/tmp/mlx-v0.32.2` (also available at `/tmp/mlx-0.32.2`):

- `mlx/memory.h:66-78`: wired limit defaults to 0;
- `mlx/backend/metal/resident.h:16,109`: explicit zero default;
- `mlx/backend/metal/resident.cpp:136-154`: insert tracks but does not admit
  allocations beyond capacity;
- `resident.cpp:175-219`: resize admits existing allocations when budget grows;
- `resident.cpp:50-66,221-242`: standing residency request and queue attachment;
- `mlx/backend/metal/allocator.cpp:100-104,149-163`: budget setter and allocation;
- `allocator.cpp:254-261`: requests above recommended working-set size fail.

`build-mlx/CMakeCache.txt` identifies MLX 0.32.2 from
`/Volumes/SDXC-512/glm53-flash-mlx/.venv/lib/python3.13/site-packages/mlx/`;
its installed public header also documents the zero default. The executable
links `@rpath/libmlx.dylib`. These establish source/API behavior, not a byte-for-
byte provenance proof of that dylib; retain binary identity in any future gate.
Do not equate Metal residency requests with unconditional physical page locking
under every OS/resource condition.

### DwarfStar, pinned `8db1d1d155cb0400a86a86b9c62d0defb3a6148b`

Inspected `git show <revision>:ds4_metal.m`, not the newer local working tree.

- Stable GGUF model mapping is exposed through no-copy shared Metal views
  (`newBufferWithBytesNoCopy`, around line 2285), rather than route-time copies.
- `ds4_gpu_model_residency_request_views` (2124-2170) registers every model
  view with a model residency set, commits, requests residency, and attaches
  the set to the command queue before inference.
- `ds4_gpu_model_residency_clear` (2095-2112) removes queue membership, ends
  residency and removes allocations during teardown.
- Explicit streaming, TP skip and `DS4_METAL_NO_RESIDENCY` bypass this policy.
- Its own comment distinguishes residency/budgeting and VM validation from
  necessarily faulting the entire file into RAM. Arithmetic/weight formats
  are not assumed equivalent to this runtime.

### oMLX, pinned `b390b31e0c6831225fed0f24d278eb1db7fcb68b`

Inspected `/Users/kioju/omlx-0.7.0.dev2`; relevant files are unchanged locally.

- `omlx/patches/deepseek_v41/convert.py:103-142`: official source conversion
  stacks experts by projection once, releases source readers/mappings before
  yielding each group, and does not stack weights during decode.
- `loading.py:207-263`: persistent projection modules take weight/scale arrays;
  `model.load_weights` and `mx.eval(values)` materialize each loaded group.
- `omlx/process_memory_enforcer.py:220-315`: explicit wired-budget application
  exists, but is **conditional**. When `iogpu.wired_limit_mb` is unset/zero,
  this helper skips `mx.set_wired_limit`; otherwise it requests a capped limit
  and reports failures. Do not claim every oMLX execution wires all weights.
- The current host's sysctl is zero at review time. That is not evidence of
  the value or allocator policy during historical oMLX runs; other generation
  contexts may also set a budget. DwarfStar provides the unconditional resident-
  mode structural comparison; the oMLX speed advantage is not yet attributed
  exclusively to this setting.

## First bounded result: 320 GiB rejected on memory gate

Reviewed root:
`artifacts/context-ladder/expert-residency-20260922-091600-41752/`.
Source/binary identities are present, preflight reported 480 GiB free+inactive
and no competing large process, parity passed the full hidden/pre-mix/logits/
state/publication/hash/revision/atomicity gate, both full paths exited 0, generated
identical token IDs, and process swap was zero. Peak footprint was 310.855 GB
baseline and 310.567 GB candidate, both below the fixed 340 GB bound.

The performance effect is material and directly supports the residency
mechanism:

| Metric | baseline | 320 GiB candidate | Change |
|---|---:|---:|---:|
| Advancing decode mean | 3.35027 s | 0.144694 s | -95.68%, 23.15x faster |
| Advancing decode range | 3.091–3.445 s | 0.1405–0.1548 s | all seven faster |
| Prefill | 43.9549 s | 43.5051 s | -1.02% |
| Prefill + decode latency | 67.7042 s | 44.7575 s | -33.89% |
| Process wall | 103.51 s | 84.62 s | -18.25% |
| Retired instructions | 926.14B | 638.74B | -31.03% |
| Peak MLX allocation | 304.386 GB | 304.216 GB | -0.06% |

System decompressions fell from 2,058,725 to 1,625,073 pages per process
interval (-21.1%, about 33.73 to 26.63 GB at 16 KiB/page). This remains a
system-wide whole-process counter, not expert-only attribution. The result
nevertheless falsifies intrinsic M=1 GatherQMM throughput as the primary
production decode cost much more strongly than the synchronized profile alone.

**The 320 GiB candidate is rejected by the unchanged memory contract.** Its
system `Swapouts` rose from 52,552 to 52,580 (+28 pages, 448 KiB), whereas the
baseline delta was zero. `Swapins` stayed flat and `/usr/bin/time` reported zero
process swaps, but neither fact permits relaxing the zero system-swap gate.
The comparator correctly stopped without writing a passing comparison. Do not
run the paired stage with this budget or reinterpret the run as promotion.

The next bounded candidate was initially 288 GiB (309,237,645,312 bytes),
above the measured 283.32 GiB peak MLX allocation. Its fresh parity run at
`artifacts/context-ladder/expert-residency-20260922-094635-42663/` passed exact
hidden/pre-mix/logits/state/publication/hash/revision/atomicity, reported zero
process swaps and 308.035 GB peak footprint, but system `Swapouts` increased
from 52,580 to 52,588 (+8 pages, 128 KiB). The runner correctly stopped before
full-path measurement. This run is also rejected; do not retry it to select a
convenient zero-delta sample.

The 272 GiB (292,057,776,128 byte) attempt at
`artifacts/context-ladder/expert-residency-20260922-095037-43289/` is also
rejected. Exact hidden/pre-mix/logits/state/publication/hash/revision/atomicity
passed, process swap was zero, and peak footprint was 308.036 GB, but parity
system `Swapouts` increased from 52,588 to 52,600 (+12 pages, 192 KiB). The
runner stopped before full-path measurement. Do not retry or continue paired.

This ends anonymous-atlas wired-budget tuning. The three reviewed attempts show
that a broad residency request can remove the dominant decode cost, but 320,
288 and 272 GiB all fail the unchanged zero-system-swap contract. Further
budget decrement would only pin a prefix of layer banks and does not address
the backing mismatch. The next structural candidate is a transactionally built,
read-only **file-backed packed atlas** exposed to MLX/Metal without copying into
anonymous allocator buffers. DwarfStar's pinned model views use this ownership
shape; current MLX public APIs support raw-pointer no-copy buffers. The official
checkpoint remains read-only and canonical. Any derived backing requires exact
byte/provenance manifests, atomic preparation, immutable runtime mapping and a
separate storage/resource review before implementation or measurement.

A checkpoint-free API probe now confirms MLX 0.32.2 can wrap a page-aligned,
`PROT_READ | MAP_PRIVATE` file mapping with `allocator::make_buffer` without
copying and execute a GPU reduction through an `mx::array`; release precedes
`munmap`/`close`. This closes API feasibility only. The proposed full format is
240 page-aligned immutable payloads (weight/scale for w1/w2/w3 across 40 layers),
about 288.78 GB logical plus alignment and manifest. Preparation must write a
new temporary root, bind every payload to checkpoint revision/tensor identities
and digest, fsync data/manifest, then atomically rename. Runtime must reject
missing/mutable/mismatched files, map read-only, and never regenerate inside a
latency-sensitive request. The format and preparation path are now implemented opt-in. Runtime activation
uses `DSV41_RUNTIME_EXPERT_BACKING_DIR`; absent means the existing anonymous
atlas. The completed root and manifest must be non-symlink, read-only objects;
each payload is checked against its recorded device/inode/size/mtime/ctime
before a read-only private mapping is wrapped by MLX. Runtime reports
`expert_backing_file_backed=true`. Arithmetic, shapes and gather-QMM calls are
unchanged.

Preparation is intentionally separate from model startup:

```sh
bash tools/benchmark/run_prepare_expert_backing.sh
```

Default destination is `artifacts/expert-backing/official-packed-v1`. Expect
about 288.78 GB plus alignment/manifest, 30–120 minutes depending on both
volumes, and at least 320 GiB free destination space. The script builds the C++
preparer, records source/binary/checkpoint-proof identity and runs one process;
the official checkpoint is opened read-only. Each of 240 payloads is SHA-256'd
while copied, fsynced, chmod 0444 and recorded with all 384 source tensor names,
shards, offsets and sizes. Progress is atomically updated after each payload.
The completed manifest/root become 0444/0555 and publish by atomic directory
rename only after fsync. Failure retains `<destination>.building`; resume with:

```sh
DSV41_EXPERT_BACKING_DIR=artifacts/expert-backing/official-packed-v1 \
bash tools/benchmark/run_prepare_expert_backing.sh
```

Resume refuses a changed preparer/binary/checkpoint-proof identity and skips
only payloads whose recorded stat identity still matches.

Preparation completed at
`artifacts/expert-backing/prepare-20260922-102432-44680/` in 1,702.77 s
(28.38 minutes), exit 0, process swaps 0, peak footprint 147.9 MB. All 240
payloads were published; the completed root contains 92,160 unique source
tensors, 288,777,830,400 logical/mapped bytes with no alignment overhead, and
checkpoint revision `dba1be0a40aa45a94ad051997016db3960a90277`.

Review found and fixed one format bug before runtime use: payload stat identity
was captured before temp-to-final rename, whose ctime nanoseconds changed.
Runtime correctly would have rejected it. The generator now captures post-rename
stat. A bounded repair tool accepted the existing root only after every
payload's device/inode/size/mtime and read-only mode matched, changed only the
recorded ctime, and atomically resealed the manifest/root. Repair evidence is
`artifacts/expert-backing/repair-20260922-post-rename-ctime/` (exit 0, 0.84 s,
0 swaps). A complete post-repair offline review passed all 240 file identities,
64-hex payload digests, source uniqueness/count, modes and total bytes.

The first runtime gate is **rejected and disabled as unsafe**. Do not resume
`artifacts/context-ladder/file-backed-residency-20260922-164206-46772` and do
not run its paired mode. A paste containing whitespace-only commands caused two
benign zsh `command not found` messages and selected a new default root, but
that was not the memory failure: backing/defaults were otherwise correct.

The parity process did prove exact file-backed arithmetic through 129-token
prefill plus four advancing tokens: hidden/pre-mix/logits/state/route ties/hash/
revision/invalid-request atomicity all passed. It then ran for 1,973.88 s,
reached 307.951 GB peak footprint and zero process-reported swaps, but system
Swapins rose 11,831 -> 18,778 (+6,947 pages) and Swapouts 52,600 -> 52,680
(+80 pages). The user observed the OS near forced restart. Baseline/candidate
full paths never started.

The proximate error was treating 232.6 GiB of post-preparation speculative file
cache as safely reclaimable alongside a ~308 GB Metal process. That preflight
change is reverted: speculative pages are excluded and the historical runner
now refuses execution without an explicitly forensic unsafe override. This is
not merely a failed performance candidate; it is a validation-harness safety
failure. No further full-model run is authorized until the host is clean and a
replacement uses completely separate processes/artifacts for expected and
candidate outputs, avoids retaining cross-model state, has a conservative
free+inactive gate, and defines file-cache conditioning without `purge` or
privileged operations. The replacement exactness harness is now implemented but deliberately split
across user-reviewed invocations. It never holds baseline and candidate models
in one process. Each process emits SHA-256 records for every hidden/pre-mix/
logits boundary and all persistent arrays covered by the prior in-process
`state_same` gates, plus publication indices/candidates, route ties, revision,
hash continuation and invalid-request atomicity. A third process performs only
JSON comparison.

The first separated baseline at
`artifacts/context-ladder/file-backed-exact-20260922-175104-1825/` emitted its
complete digest in 74.30 s and stayed below 307.932 GB, but is rejected:
system Swapins increased by 4 pages, Swapouts by 20, and the compressor reached
~30.2 GiB. Candidate did not run. A full anonymous resident atlas was still the
wrong exactness reference because checkpoint reads grew file cache beside the
269 GiB copied atlas.

The harness now uses the request-local compact expert-bank path for baseline
(`resident_atlas=0`, `compact_bank=1`) and reserves the full file-backed atlas
for candidate. This preserves checkpoint-derived arithmetic and the same digest
surface while bounding baseline expert memory. Because the failed run left
0.31 MB active swap, reboot is required before another attempt. After reboot,
start a **new root** with:

```sh
bash tools/benchmark/run_file_backed_exactness_safe.sh baseline
```

The compact baseline completed safely at
`artifacts/context-ladder/file-backed-exact-20260922-175746-1209/`: 34.01 s
wall, 107.961 GB peak footprint, zero process/system swap, zero compressor
pages before/after, five digest steps, three route ties and valid invalid-request
atomicity. Telemetry confirms resident atlas OFF, compact bank ON, backing OFF,
wired budget zero. This authorizes only the separate candidate exactness process.

Preflight requires at least 400 GiB
**free+inactive** (speculative excluded), zero active swap usage and no process
above 32 GiB. The runner records before/after VM counters, requires zero
process/system swap delta, then stops. Do not run candidate until baseline raw
logs are reviewed. Candidate uses the same root in a separate invocation and
may take 15–40 minutes on cold backing; comparison is a third offline
invocation. Failure never resumes model state. This is exactness only—no
performance, promotion, paired or 64K claim.

Reviewed candidate root:
`artifacts/context-ladder/file-backed-exact-20260922-180600-2153/`.
The compact baseline completed in 19.43 s at 107.962 GB peak with zero process/
system swap and no compression. The separately invoked file-backed+272 GiB
candidate completed in 1,667.91 s (cold backing read), reported only 19.082 GB
peak process footprint, zero process/system swap, and exact invalid-request
atomicity. System compression increased by 848,135 pages (~13.9 GB), but no
swap occurred and active swap remained zero.

The compact baseline and resident candidate differ at step-0 hidden because the
compact request-local MoE schedule has a different evaluation/reduction shape;
it is not a bitwise oracle for the resident expert-major schedule. This does not
implicate backing bytes. The candidate digest was therefore compared offline to
the independently produced anonymous **resident full-atlas** digest from the
rejected-memory run
`file-backed-exact-20260922-175104-1825/baseline/digest.json`. All five steps,
hidden/pre-mix/logits, complete recorded persistent state/publication, hash
continuation, invalid atomicity and route ties match exactly. Evidence:
`file-backed-exact-20260922-180600-2153/resident-oracle-comparison.json`, exit 0.
Thus file-backed storage exactness passes; the earlier anonymous oracle's swap
failure remains a memory rejection and is not reclassified.

No performance promotion follows. Cold 129+4 exactness wall principally reads
~289 GB and is not production decode evidence.

Candidate-only full path
`artifacts/context-ladder/file-backed-candidate-20260922-185613-3093/` confirms
the mechanism but rejects the 272 GiB policy: advancing decode mean was
0.14808 s/token (versus the historical clean anonymous ~3.35 s/token), prefill
54.836 s, and file-backed process footprint 21.688 GB. However system Swapouts
rose by 16 pages (256 KiB), leaving 0.25 MB active swap, so the unchanged memory
gate fails. Model construction was also unacceptable at 638.913 s, including
627.903 s in 40 bank mappings. The cause is per-buffer residency insertion:
the wired budget was active while all 240 no-copy views were created.

The next candidate changes scheduling, not arithmetic: create the complete atlas
with wired capacity zero, activate one residency resize after atlas mapping but
before encoder/decoder dense allocations, and reduce the budget to 256 GiB.
This mirrors DwarfStar's batch view registration and should remove repeated VM
validation while leaving later dense/state/scratch allocations outside the
finite budget as needed. Exact backing bytes remain covered by the reviewed
digest evidence. Active swap from the rejected run requires reboot before any
new measurement. A new-root candidate-only runner remains the next gate; frozen
baseline comparison is descriptive and cannot promote without repeated causal
evidence.

## Selected candidate and safety contract

Use the existing C++ MLX `set_wired_limit` API, initially opt-in, before model
allocation. Keep all arithmetic, packed layout, routing and state semantics
unchanged. No model conversion, checkpoint writes, Python runtime dependency,
blanket `mlock`, or automatic privileged sysctl changes are needed.

Implemented opt-in on 2026-09-22. `RuntimeResidencyLease` is the first member
and last destructor of `TextBackboneReference`. A nonzero
`DSV41_RUNTIME_WIRED_LIMIT_BYTES` acquires the process-global MLX budget before
the atlas or other model arrays are allocated; the default remains zero.
The preserved rejected experiment rejects malformed values, budgets below 272 GiB, device/OS-reserve
violations, late acquisition after >1 GiB active MLX memory, and concurrent
backbone ownership involving a wired model. It synchronizes and restores the
prior budget during teardown. Logs call this a policy explicitly, not proof of
physical pages. The production bridge inherits the same model-lifecycle owner.

Implementation must:

- Own the process-global budget at the runtime/device lifecycle, not at each
  bank or request; coordinate multiple runtime owners and in-flight work.
- Bound the requested budget by the supported device cap and an explicit OS
  reserve; account for dense weights, state and transient allocations, not
  just atlas bytes. MLX's allocation admission is not expert-prioritized.
- Set the budget before atlas construction so unrelated cache allocations
  cannot consume a late resize's unordered admission first.
- Report requested/applied policy and reject unsupported/insufficient candidate
  configurations rather than silently claiming residency. Do not change the
  existing allocation/memory ceiling or idle-cache policy implicitly.
- Synchronize at lifecycle teardown and restore the prior budget only when
  the owning runtime is finished; never unset residency mid-inference.

No new full-model run was launched for this review. Checkpoint-free
`dsv41-residency-test` passes parsing, cap/reserve rejection, exclusive owner,
GPU execution, constructor-unwind-style teardown and prior-budget restoration
with the rejected anonymous-atlas candidate. This is API/lifecycle evidence only.

`tools/benchmark/run_expert_residency_candidate.sh` and its comparator are
preserved solely to reproduce the reviewed 320/288/272 GiB experiments. Do not
run bounded again and do not run its paired mode. None auto-promotes or
qualifies 64K. A new runner will be prepared only after the file-backed atlas
format, provenance, preparation resource bound and no-copy lifetime have been
implemented and reviewed. The first target remains the cold service gap, not a
promised 86x end-to-end speedup.
