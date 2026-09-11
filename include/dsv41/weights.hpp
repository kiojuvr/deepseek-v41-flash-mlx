#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace dsv41 {
class ShardFile;
class MappedTensor {
public:
    MappedTensor() = default;
    std::span<const std::byte> bytes(std::uint64_t offset, std::size_t length) const;
    void check_unchanged() const;
    std::uint64_t size() const;
private:
    struct Region;
    std::shared_ptr<Region> region_;
    friend class TensorFile;
};
class TensorFile {
public:
    std::string name, dtype, shard;
    std::vector<std::uint64_t> shape;
    std::uint64_t file_offset = 0, size_bytes = 0;
    void read(std::uint64_t offset, std::span<std::byte> destination) const;
    MappedTensor map() const;
private:
    std::shared_ptr<ShardFile> file_;
    friend class WeightCatalog;
};
class WeightCatalog {
public:
    // M1 proof is historical: inspect current headers/sizes against it. This does
    // not replace full payload attestation at production startup.
    WeightCatalog(const std::filesystem::path& checkpoint, const std::filesystem::path& m1_summary);
    ~WeightCatalog();
    WeightCatalog(WeightCatalog&&) noexcept;
    WeightCatalog& operator=(WeightCatalog&&) noexcept;
    TensorFile tensor(const std::string& name);
    std::string revision() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
