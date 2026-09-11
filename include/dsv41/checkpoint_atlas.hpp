#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

namespace dsv41 {
struct AtlasOptions {
    std::filesystem::path checkpoint;
    std::filesystem::path output;
    std::filesystem::path manifest;
    bool verify_payload = false;
};
// 0: complete and valid at requested verification level; 2: partial/invalid.
// Throws on CLI, input manifest, or output I/O failures.
int inspect_checkpoint(const AtlasOptions& options);
// Shared strict readers used by the native catalog; no tensor payload is loaded.
nlohmann::json read_safetensors_header(const std::filesystem::path& path);
nlohmann::json read_json_file(const std::filesystem::path& path);
std::string sha256_text(const std::string& text);
}
