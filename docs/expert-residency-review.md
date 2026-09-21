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

## Selected candidate and safety contract

Use the existing C++ MLX `set_wired_limit` API, initially opt-in, before model
allocation. Keep all arithmetic, packed layout, routing and state semantics
unchanged. No model conversion, checkpoint writes, Python runtime dependency,
blanket `mlock`, or automatic privileged sysctl changes are needed.

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

No new full-model run was launched for this review. Future bounded correctness
and unprofiled baseline/candidate full-path performance gates must be prepared
as user-run scripts before execution. A successful API call or an isolated
probe is not promotion. Verify memory/compression behavior and unchanged
outputs as well as advancing decode and complete wall time. The first target
is the cold service gap, not a promised 86x end-to-end speedup.
