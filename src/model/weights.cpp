#include "dsv41/weights.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dsv41 {
namespace {
void check(bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); }
void bounds(std::uint64_t offset, std::uint64_t count, std::uint64_t size) {
    check(offset <= size && count <= size-offset, "tensor read out of bounds");
}
}
class ShardFile {
public:
    int fd = -1;
    struct stat identity{};
    nlohmann::json header;
    explicit ShardFile(const std::filesystem::path& path) {
        fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW);
        check(fd >= 0, "cannot open shard: " + path.string());
        if (fstat(fd,&identity) != 0 || !S_ISREG(identity.st_mode)) { close(fd); fd = -1; throw std::runtime_error("invalid shard file"); }
    }
    ~ShardFile() { if (fd >= 0) close(fd); }
    void unchanged() const {
        struct stat now{};
        check(fstat(fd,&now) == 0 && now.st_size == identity.st_size &&
              now.st_mtimespec.tv_sec == identity.st_mtimespec.tv_sec &&
              now.st_mtimespec.tv_nsec == identity.st_mtimespec.tv_nsec &&
              now.st_ctimespec.tv_sec == identity.st_ctimespec.tv_sec &&
              now.st_ctimespec.tv_nsec == identity.st_ctimespec.tv_nsec,
              "checkpoint shard changed while in use");
    }
};
struct MappedTensor::Region {
    std::shared_ptr<ShardFile> file;
    void* base = MAP_FAILED;
    std::size_t mapped_bytes = 0, displacement = 0;
    std::uint64_t logical_bytes = 0;
    ~Region() { if (base != MAP_FAILED) munmap(base,mapped_bytes); }
};
std::span<const std::byte> MappedTensor::bytes(std::uint64_t offset, std::size_t length) const {
    check(bool(region_), "empty mapping");
    bounds(offset,length,region_->logical_bytes);
    if (!length) return {};
    return {static_cast<const std::byte*>(region_->base)+region_->displacement+offset,length};
}
void MappedTensor::check_unchanged() const { check(bool(region_),"empty mapping"); region_->file->unchanged(); }
std::uint64_t MappedTensor::size() const { return region_ ? region_->logical_bytes : 0; }
void TensorFile::read(std::uint64_t offset, std::span<std::byte> destination) const {
    check(bool(file_),"unbound tensor"); bounds(offset,destination.size(),size_bytes); file_->unchanged();
    std::size_t done = 0;
    while (done < destination.size()) {
        const auto n = pread(file_->fd,destination.data()+done,destination.size()-done,
                             static_cast<off_t>(file_offset+offset+done));
        if (n < 0 && errno == EINTR) continue;
        check(n > 0,"short/failed positional read"); done += static_cast<std::size_t>(n);
    }
    file_->unchanged();
}
MappedTensor TensorFile::map() const {
    check(bool(file_),"unbound tensor"); file_->unchanged();
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    check(page > 0 && page <= 65536,"invalid page size");
    auto r = std::make_shared<MappedTensor::Region>(); r->file = file_; r->logical_bytes = size_bytes;
    const auto start = file_offset - file_offset%page;
    r->displacement = file_offset-start;
    check(size_bytes <= std::numeric_limits<std::size_t>::max()-r->displacement,"mapping length overflow");
    r->mapped_bytes = static_cast<std::size_t>(size_bytes)+r->displacement;
    if (size_bytes) {
        r->base = mmap(nullptr,r->mapped_bytes,PROT_READ,MAP_PRIVATE,file_->fd,static_cast<off_t>(start));
        check(r->base != MAP_FAILED,"mmap failed");
    }
    file_->unchanged();
    MappedTensor result; result.region_ = std::move(r); return result;
}
struct WeightCatalog::Impl {
    std::filesystem::path root;
    nlohmann::json index, proof;
    std::map<std::string,std::shared_ptr<ShardFile>> shards;
};
WeightCatalog::WeightCatalog(const std::filesystem::path& root, const std::filesystem::path& summary)
    : impl_(std::make_unique<Impl>()) {
    impl_->root = std::filesystem::canonical(root);
    impl_->index = read_json_file(impl_->root/"model.safetensors.index.json").at("weight_map");
    impl_->proof = read_json_file(summary);
    check(impl_->proof.at("status") == "verified","catalog requires completed M1 attestation");
}
WeightCatalog::~WeightCatalog() = default;
WeightCatalog::WeightCatalog(WeightCatalog&&) noexcept = default;
WeightCatalog& WeightCatalog::operator=(WeightCatalog&&) noexcept = default;
std::string WeightCatalog::revision() const { return impl_->proof.at("revision"); }
TensorFile WeightCatalog::tensor(const std::string& name) {
    const std::string shard = impl_->index.at(name);
    check(std::filesystem::path(shard).filename() == shard && shard != "." && shard != "..", "unsafe shard name");
    auto it = impl_->shards.find(shard);
    if (it == impl_->shards.end()) {
        const auto path = impl_->root/shard;
        auto file = std::make_shared<ShardFile>(path);
        file->header = read_safetensors_header(path);
        const auto& expected = impl_->proof.at("shards").at(shard);
        check(file->header.at("header_sha256") == expected.at("header_sha256") &&
              file->header.at("file_bytes") == expected.at("file_bytes"),"shard differs from M1 header/size");
        struct stat current{};
        check(lstat(path.c_str(),&current) == 0 && current.st_ino == file->identity.st_ino &&
              current.st_dev == file->identity.st_dev,"shard replaced during open");
        file->unchanged(); it = impl_->shards.emplace(shard,std::move(file)).first;
    }
    const auto& h = it->second->header;
    const auto& t = h.at("tensors").at(name);
    TensorFile result;
    result.file_ = it->second; result.name = name; result.shard = shard; result.dtype = t.at("dtype");
    result.shape = t.at("shape").get<std::vector<std::uint64_t>>();
    result.file_offset = 8+h.at("header_bytes").get<std::uint64_t>()+t.at("data_offsets")[0].get<std::uint64_t>();
    result.size_bytes = t.at("data_offsets")[1].get<std::uint64_t>()-t.at("data_offsets")[0].get<std::uint64_t>();
    return result;
}
}
