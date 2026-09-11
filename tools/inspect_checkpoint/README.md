# Checkpoint atlas

C++20 utility for this project's official DeepSeek-V4.1 checkpoint. It reads safetensors headers without loading weight tensors, validates the index and quantization relationships, and optionally hashes all payloads against a pinned Hugging Face manifest. No inference, conversion, MLX, Rust, or Python runtime is required.

## Build and inspect

On macOS with Xcode command-line tools and CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j 4
build/dsv41-inspect-checkpoint \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --output artifacts/checkpoint \
  --manifest artifacts/checkpoint/upstream-manifest.json \
  --verify-payload
```

Omit `--verify-payload` for a header inspection. Small ancillary files are still hashed. Full verification reads approximately 510 GB sequentially with an 8 MiB buffer and `F_NOCACHE`; it does not materialize the model or use the GPU. The checkpoint is read-only; the output directory must be outside it. Do not run against an actively changing download.

The pinned manifest is the `sha`, `siblings`, and `safetensors` fields from the official Hugging Face model API, plus its source URL. LFS files use whole-file SHA-256, regular Git files use Git blob SHA-1 (including the `blob SIZE\0` prefix). Computed SHA-256 is also recorded for every verified file. Download-cache metadata is recorded as provenance, but the upstream content digest decides validity. A tensor row's `digest` always covers its whole shard, not the individual tensor.

Exit codes: `0` = complete at the requested verification level; `2` = partial or invalid checkpoint; `1` = invalid CLI, manifest/config/index parsing failure, or operational failure. These are checkpoint accounting/integrity results, not model numerical correctness. Use a fresh output directory for a separate run: early fatal errors can leave prior output files in an existing directory.

## Outputs

- `atlas.jsonl`: one row per successfully classified tensor; storage/logical dtype and shape, offsets, scale relationship, owner, residency, and verification status. Approximately 92 MB for the release; reproducible and excluded from Git.
- `summary.json`: totals by owner, storage/logical dtype, component, layer, shard, and residency; all structural/semantic issues; header digests and byte counts. Unclassified tensors make the result invalid and appear in `issues`, not in classified totals.
- `verification.json`: per-file expected and computed hashes, snapshot identity, verification status and local download metadata.

`runtime_bytes` remains null because the atlas cannot measure MLX/Metal allocations, replay state, or graph scratch. `resident_payload_bytes` is the exact stored-weight contribution; the [M1 report](../../docs/checkpoint-atlas.md) defines separate runtime admission budgets. `alias_group` is the canonical tensor name: this snapshot contains no duplicate MTP embedding/head tensors to deduplicate. Runtime reuse of the backbone embedding/head does not allocate them again.

This is a model-specific inspector. It recognizes the released BF16/F32/F8_E4M3/F8_E8M0/I8 vocabulary. I8 routed-expert matrices are decoded logically as FP4 E2M1 using the official converter's low-nibble then high-nibble order, with E8M0 scales per 32 logical channels. Engram uses per-row groups of 32; dense FP8 uses 32×32 blocks. Unknown layouts are rejected. The tool validates supplied index/header coverage; it is not an independent generator of all expected model tensors or a proof that their values implement the model correctly.

The underlying container contract is documented by [safetensors](https://github.com/safetensors/safetensors#format). JSON parsing rejects duplicate keys and excessive nesting; descriptors reject unsupported dtypes, integer overflow, invalid shape/offsets, gaps/overlaps, trailing payload and unsafe source paths. Output is deterministic except elapsed timing, and the full verifier checks file stability during hashing and rechecks the inspected shard headers.

## Tests and hardware probe

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
build/dsv41-memory-probe
```

Tests use Python's standard library solely to generate small fixtures and invoke the C++ binary. They cover accounting, corruption, duplicate JSON keys, overflow, wrong offsets/dtypes/scales, missing and unexpected shards, unsafe paths and symlinks. The memory probe queries Foundation/Metal and physical-memory limits without allocating model buffers; sandboxed environments may need permission to expose the Metal device.
