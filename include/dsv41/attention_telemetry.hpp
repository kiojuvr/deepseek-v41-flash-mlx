#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
namespace dsv41 {
struct AttentionTelemetry { std::size_t concat_calls=0, concat_input_bytes=0, concat_output_bytes=0, cumulative_bytes_copied=0; std::size_t logical_tokens=0, attention_rows=0, indexer_rows=0, index_host_readbacks=0, token_serial_attention_calls=0, chunk_attention_calls=0, packed_chunk_attention_calls=0, wide_attention_calls=0, fixed_tile_attention_calls=0, chunk_batched_splitk_qk_calls=0, chunk_scalar_qk_calls=0, chunk_scalar_av_calls=0, chunk_av_batches=0; };
inline AttentionTelemetry& attention_telemetry(){ static AttentionTelemetry v; return v; }
inline std::mutex& attention_telemetry_mutex(){ static std::mutex m; return m; }
inline void reset_attention_telemetry(){ std::lock_guard l(attention_telemetry_mutex()); attention_telemetry()={}; }
inline AttentionTelemetry read_attention_telemetry(){ std::lock_guard l(attention_telemetry_mutex()); return attention_telemetry(); }
}
