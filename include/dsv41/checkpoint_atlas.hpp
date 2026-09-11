#pragma once
#include <filesystem>

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
}
