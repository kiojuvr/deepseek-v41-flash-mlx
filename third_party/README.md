# Third-party sources

`nlohmann/json.hpp` is the unmodified single header from nlohmann/json **v3.12.0**:

- Source: https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp
- SHA-256: `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`
- License: [MIT](nlohmann/LICENSE.MIT), copyright Niels Lohmann; the header includes further upstream notices.
- License SHA-256: `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603`

Fetched directly from upstream on 2026-09-11. No GLM runtime source was copied.
Apple CommonCrypto, Foundation, and Metal are linked from the installed SDK.
MLX and deepseek-recipe are not build dependencies of the M1 atlas.

The fused DeepSeek V4.1 prefill-attention topology in
`metal/attention/packed_chunk_attention.metal` is adapted from oMLX commit
`b390b31e0c6831225fed0f24d278eb1db7fcb68b`, file
`omlx/custom_kernels/glm_moe_dsa/csrc/kernels/steel_deepseek_v41_packed_attention.h`.
It retains the upstream Copyright © 2026 OpenAI notice and is distributed
under the [Apache License 2.0](omlx/LICENSE.Apache-2.0). The adaptation changes
the local-KV input to the runtime's BF16 round-trip state, splits packed pooled
values/scales into their native persistent buffers, and uses MLX JIT arguments.
