#pragma once

#include <cstddef>
#include <stdexcept>

namespace dsv41 {

inline constexpr std::size_t kDecoderFirstLayer = 20;
inline constexpr std::size_t kDecoderLastLayer = 39;
inline constexpr std::size_t kDecoderRawHistory = 127;

// Exact dependency geometry used by DwarfStar's reviewed V4.1 deferred
// decoder: the final layer needs one row and every preceding decoder layer
// adds the 127 raw rows consumed by its successor.
inline std::size_t deferred_decoder_suffix_rows(std::size_t layer) {
  if (layer < kDecoderFirstLayer || layer > kDecoderLastLayer) {
    throw std::runtime_error("deferred decoder layer must be in 20..39");
  }
  return 1 + (kDecoderLastLayer - layer) * kDecoderRawHistory;
}

struct DeferredDecoderSpan {
  std::size_t first = 0;
  std::size_t rows = 0;
  std::size_t warm_first = 0;
  std::size_t warm_rows = 0;
};

inline DeferredDecoderSpan deferred_decoder_span(std::size_t total_rows, std::size_t layer) {
  const auto rows = deferred_decoder_suffix_rows(layer);
  if (total_rows < rows + kDecoderRawHistory) {
    throw std::runtime_error("deferred decoder sweep is too short for exact raw-history warmup");
  }
  const auto first = total_rows - rows;
  return {first, rows, first - kDecoderRawHistory, kDecoderRawHistory};
}

// Match the reviewed outer DwarfStar policy: deferral must remove a real
// decoder sweep and leave enough work for an exact completing sweep.
inline bool should_defer_decoder(std::size_t sweep_rows, std::size_t remaining_after) {
  return sweep_rows >= 16384 && remaining_after >= 8192;
}

}  // namespace dsv41
