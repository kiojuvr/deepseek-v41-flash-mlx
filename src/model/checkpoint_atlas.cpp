#include "dsv41/checkpoint_atlas.hpp"
#include <nlohmann/json.hpp>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace dsv41 {
namespace {
using J = nlohmann::json;
using U = std::uint64_t;
namespace fs = std::filesystem;
constexpr U max_header = 100000000;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
U add(U a, U b) {
    require(b <= std::numeric_limits<U>::max() - a, "integer addition overflow");
    return a + b;
}
U mul(U a, U b) {
    require(a == 0 || b <= std::numeric_limits<U>::max() / a, "integer multiplication overflow");
    return a * b;
}
U uint(const J& j) {
    require(j.is_number_unsigned(), "expected unsigned JSON integer");
    return j.get<U>();
}
U numel(const J& shape) {
    require(shape.is_array() && shape.size() <= 32, "invalid shape rank");
    U n = 1;
    for (const auto& d : shape) n = mul(n, uint(d));
    return n;
}
std::string read_text(const fs::path& path) {
    require(fs::file_size(path) <= max_header, "JSON/text file exceeds size limit: " + path.string());
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "cannot read " + path.string());
    std::ostringstream out; out << f.rdbuf();
    require(!f.bad(), "read failed: " + path.string());
    return out.str();
}
J parse(const std::string& text) {
    std::vector<std::set<std::string>> keys;
    return J::parse(text, [&](int depth, J::parse_event_t ev, J& value) {
        require(depth <= 128, "JSON nesting exceeds limit");
        if (ev == J::parse_event_t::object_start) keys.emplace_back();
        if (ev == J::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "duplicate JSON key");
        if (ev == J::parse_event_t::object_end) keys.pop_back();
        return true;
    });
}
std::string hex(const unsigned char* bytes, size_t n) {
    std::ostringstream out;
    for (size_t i = 0; i < n; ++i)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    return out.str();
}
std::string sha256(const std::string& s) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(s.data(), static_cast<CC_LONG>(s.size()), digest);
    return hex(digest, sizeof(digest));
}
// Sequential bounded reads. F_NOCACHE avoids retaining a second model in the OS cache.
J hash_file(const fs::path& path, bool git_blob) {
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    require(fd >= 0, "cannot open regular file for hashing: " + path.string());
    struct Guard { int fd; ~Guard() { close(fd); } } guard{fd};
    struct stat before{}, after{}, current{};
    require(fstat(fd, &before) == 0 && S_ISREG(before.st_mode), "not a regular file");
    require(fcntl(fd, F_NOCACHE, 1) == 0, "cannot enable F_NOCACHE");
    CC_SHA256_CTX sha; CC_SHA256_Init(&sha);
    CC_SHA1_CTX blob; CC_SHA1_Init(&blob);
    if (git_blob) {
        std::string prefix = "blob " + std::to_string(before.st_size);
        prefix.push_back('\0');
        CC_SHA1_Update(&blob, prefix.data(), static_cast<CC_LONG>(prefix.size()));
    }
    std::vector<char> buffer(8 * 1024 * 1024);
    U bytes = 0;
    while (true) {
        const auto n = read(fd, buffer.data(), buffer.size());
        if (n < 0 && errno == EINTR) continue;
        require(n >= 0, "payload read failed");
        if (n == 0) break;
        CC_SHA256_Update(&sha, buffer.data(), static_cast<CC_LONG>(n));
        if (git_blob) CC_SHA1_Update(&blob, buffer.data(), static_cast<CC_LONG>(n));
        bytes = add(bytes, static_cast<U>(n));
    }
    require(fstat(fd, &after) == 0 && lstat(path.c_str(), &current) == 0, "file vanished during hash");
    require(bytes == static_cast<U>(before.st_size) && before.st_size == after.st_size &&
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec &&
            before.st_ino == current.st_ino && before.st_dev == current.st_dev,
            "file changed during hash");
    unsigned char s[CC_SHA256_DIGEST_LENGTH], b[CC_SHA1_DIGEST_LENGTH];
    CC_SHA256_Final(s, &sha); CC_SHA1_Final(b, &blob);
    return {{"sha256", hex(s, sizeof(s))}, {"git_blob_sha1", git_blob ? J(hex(b, sizeof(b))) : J(nullptr)},
            {"bytes", bytes}};
}
bool safe_relative(const std::string& name) {
    const fs::path p(name);
    if (p.empty() || p.is_absolute() || name.find('\\') != std::string::npos) return false;
    for (const auto& part : p) if (part == ".." || part == ".") return false;
    return true;
}
fs::path source_path(const fs::path& root, const std::string& name) {
    require(safe_relative(name), "unsafe source path: " + name);
    fs::path p = root;
    for (const auto& part : fs::path(name)) {
        p /= part;
        require(!fs::is_symlink(p), "source symlink rejected: " + name);
    }
    return p;
}
U dtype_bytes(const std::string& d) {
    if (d == "BF16") return 2;
    if (d == "F32") return 4;
    if (d == "I8" || d == "F8_E4M3" || d == "F8_E8M0") return 1;
    throw std::runtime_error("unsupported checkpoint storage dtype: " + d);
}
J header(const fs::path& path) {
    const U size = fs::file_size(path);
    require(size >= 8, "truncated safetensors length");
    std::ifstream f(path, std::ios::binary);
    std::array<unsigned char, 8> b{};
    f.read(reinterpret_cast<char*>(b.data()), 8);
    require(bool(f), "cannot read safetensors length");
    U n = 0;
    for (int i = 0; i < 8; ++i) n |= U(b[i]) << (8 * i);
    require(n > 0 && n <= max_header && n <= size - 8, "invalid/truncated safetensors header length");
    std::string s(n, '\0'); f.read(s.data(), static_cast<std::streamsize>(n));
    require(bool(f) && s.front() == '{', "invalid safetensors JSON header");
    J h = parse(s);
    require(h.is_object(), "header is not an object");
    if (h.contains("__metadata__")) {
        require(h["__metadata__"].is_object(), "invalid safetensors metadata");
        for (const auto& v : h["__metadata__"]) require(v.is_string(), "metadata value is not a string");
        h.erase("__metadata__");
    }
    std::vector<std::pair<U, U>> ranges;
    for (auto& [name, t] : h.items()) {
        require(t.is_object() && t.size() == 3, "invalid tensor descriptor: " + name);
        const auto& o = t.at("data_offsets");
        require(o.is_array() && o.size() == 2, "invalid offsets");
        const U begin = uint(o[0]), end = uint(o[1]);
        require(begin <= end && end <= size - 8 - n, "tensor offset out of bounds");
        require(end - begin == mul(numel(t.at("shape")), dtype_bytes(t.at("dtype").get<std::string>())),
                "shape/dtype/payload size mismatch: " + name);
        ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    U cursor = 0;
    for (auto [a, b_] : ranges) { require(a == cursor, "payload gap or overlap"); cursor = b_; }
    require(cursor == size - 8 - n, "unindexed trailing payload");
    return {{"tensors", h}, {"header_bytes", n}, {"header_sha256", sha256(s)},
            {"file_bytes", size}, {"payload_bytes", cursor}};
}

// Explicit vocabulary of the official checkpoint. Unknown names never become resident by default.
J classify(const std::string& name, const J& t, const J& cfg) {
    const auto& c = cfg.at("text_config");
    std::string owner, component, role, tail = name;
    int layer = -1, expert = -1;
    std::smatch m;
    static const std::regex layer_re(R"(^(layers|mtp)\.(\d+)\.(.+)$)");
    static const std::regex expert_re(R"(^ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$)");
    const bool scale = name.ends_with(".scale");
    if (std::regex_match(name, m, layer_re)) {
        layer = std::stoi(m[2]); tail = m[3];
        const bool mtp = m[1] == "mtp";
        require(layer < c.at(mtp ? "num_nextn_predict_layers" : "num_hidden_layers").get<int>(), "layer out of range");
        owner = mtp ? "dspark" : layer < c.at("num_hidden_layers").get<int>() / 2 ? "encoder" : "decoder";
        if (tail.starts_with("engram.")) {
            require(!mtp, "Engram in MTP");
            bool found = false;
            for (const auto& i : c.at("engram_layer_ids")) if (i == layer) found = true;
            require(found, "Engram at unconfigured layer");
            if (tail == "engram.embed.weight" || tail == "engram.embed.scale") {
                owner = "engram_backing"; component = "lookup_table";
            } else if (tail == "engram.k_weight" || tail == "engram.q_weight" ||
                       tail == "engram.wkv.weight" || tail == "engram.wkv.scale") {
                owner = "engram_resident"; component = "projection";
            }
        } else if (std::regex_match(tail, m, expert_re)) {
            expert = std::stoi(m[1]);
            require(expert < c.at(mtp ? "dspark_n_routed_experts" : "n_routed_experts").get<int>(), "expert out of range");
            component = "moe_routed";
            auto shape = t.at("shape");
            U out = uint(c.at("moe_intermediate_size")), in = uint(c.at("hidden_size"));
            if (m[2] == "2") std::swap(out, in);
            require(shape == J::array({out, scale ? in / 32 : in / 2}), "expert shape mismatch");
            require(t.at("dtype") == (scale ? "F8_E8M0" : "I8"), "expert dtype mismatch");
        } else {
            static const std::regex attn(R"(^attn\.(attn_sink|(q_norm|kv_norm)\.weight|(wq_a|wq_b|wkv|wo_a|wo_b)\.(weight|scale)|compressor\.(norm|wgate|wkv)\.weight|indexer\.(k_norm|weights_proj|wk)\.weight|indexer\.wq_b\.(weight|scale))$)");
            static const std::regex shared(R"(^ffn\.shared_experts\.w[123]\.(weight|scale)$)");
            static const std::regex gate(R"(^ffn\.gate\.(bias|bias_vl|weight)$)");
            static const std::regex hc(R"(^hc_(attn|ffn)_(base|fn|scale)$)");
            static const std::regex mtp_extra(R"(^(main_norm|norm)\.weight|main_proj\.(weight|scale)|markov_head\.(embed|head)\.weight|confidence_head\.proj\.weight$)");
            if (std::regex_match(tail, attn)) component = tail.starts_with("attn.indexer.") ? "indexer" : "attention";
            else if (std::regex_match(tail, shared)) component = "moe_shared";
            else if (std::regex_match(tail, gate)) component = "router";
            else if (std::regex_match(tail, hc)) component = "mhc";
            else if (tail == "attn_norm.weight" || tail == "ffn_norm.weight") component = "norm";
            else if (mtp && std::regex_match(tail, mtp_extra)) component = "draft_head";
        }
    } else {
        static const std::regex vision(R"(^vision\.(blocks\.\d+\.(attn\.(wo|wqkv)\.(weight|bias)|mlp\.w[12]\.weight|norm[12]\.weight)|norm\.weight|patch_embed\.proj\.(weight|bias))$)");
        static const std::regex align(R"(^aligner\.w[12]\.(weight|bias)$)");
        if (std::regex_match(name, vision) || std::regex_match(name, align) ||
            name == "image_start" || name == "image_end" || name == "image_newline") {
            owner = "vision"; component = name.starts_with("vision.") ? "encoder" : "projector";
        } else if (name == "embed.weight") { owner = "embedding"; component = "embedding"; }
        else if (name == "head.weight" || name == "norm.weight") { owner = "output"; component = "output"; }
    }
    require(!owner.empty() && !component.empty(), "unknown semantic owner: " + name);
    role = scale ? "quantization_scale" : name.ends_with("bias") || name.ends_with("bias_vl") ? "bias" : "parameter";
    J logical_shape = t.at("shape");
    std::string logical_dtype = t.at("dtype");
    if (logical_dtype == "I8") {
        require(component == "moe_routed" && !scale, "I8 outside known FP4 expert");
        logical_shape[1] = mul(uint(logical_shape[1]), 2);
        logical_dtype = "FP4_E2M1";
    }
    const U bytes = uint(t.at("data_offsets")[1]) - uint(t.at("data_offsets")[0]);
    return {{"name", name}, {"storage_dtype", t.at("dtype")}, {"storage_shape", t.at("shape")},
            {"storage_bytes", bytes}, {"storage_numel", numel(t.at("shape"))},
            {"logical_dtype", logical_dtype}, {"logical_shape", logical_shape},
            {"logical_numel", numel(logical_shape)}, {"semantic_owner", owner},
            {"component", component}, {"layer_id", layer < 0 ? J(nullptr) : J(layer)},
            {"expert_id", expert < 0 ? J(nullptr) : J(expert)}, {"role", role},
            {"residency", owner == "engram_backing" ? "ssd" : "unified_memory"},
            {"alias_group", name}, {"runtime_bytes", nullptr},
            {"resident_payload_bytes", owner == "engram_backing" ? 0 : bytes},
            {"scale_tensor", nullptr}, {"scale_format", nullptr}, {"group_size", nullptr}};
}
void add_totals(J& totals, const std::string& key, const J& row) {
    if (!totals.contains(key)) totals[key] = {{"tensors", U(0)}, {"storage_bytes", U(0)},
        {"storage_numel", U(0)}, {"logical_numel", U(0)}, {"logical_parameters", U(0)}};
    auto& v = totals[key];
    v["tensors"] = add(v["tensors"].get<U>(), 1);
    for (const auto* f : {"storage_bytes", "storage_numel", "logical_numel"})
        v[f] = add(v[f].get<U>(), row.at(f).get<U>());
    if (row.at("role") != "quantization_scale")
        v["logical_parameters"] = add(v["logical_parameters"].get<U>(), row.at("logical_numel").get<U>());
}
void write_json(const fs::path& path, const J& value) {
    std::ofstream f(path); require(bool(f), "cannot write " + path.string());
    f << value.dump(2) << '\n'; f.close(); require(bool(f), "output write failed");
}
}

nlohmann::json read_safetensors_header(const fs::path& path) { return header(path); }
nlohmann::json read_json_file(const fs::path& path) { return parse(read_text(path)); }
std::string sha256_text(const std::string& text) { return sha256(text); }

int inspect_checkpoint(const AtlasOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    const auto root = fs::canonical(options.checkpoint);
    const auto output = fs::weakly_canonical(options.output);
    const auto relative = output.lexically_relative(root);
    require(!relative.empty() && *relative.begin() == "..", "output must be outside checkpoint directory");
    const auto manifest_text = read_text(options.manifest);
    const J manifest = parse(manifest_text);
    const std::string revision = manifest.at("sha");
    require(std::regex_match(revision, std::regex("[a-f0-9]{40}")), "invalid manifest revision");
    std::map<std::string, J> sources;
    for (const auto& s : manifest.at("siblings")) {
        const std::string name = s.at("rfilename");
        require(safe_relative(name) && sources.emplace(name, s).second, "unsafe or duplicate manifest path");
    }
    const auto index_path = source_path(root, "model.safetensors.index.json");
    const auto index_text = read_text(index_path);
    const auto config_text = read_text(source_path(root, "config.json"));
    const J index = parse(index_text);
    const J cfg = parse(config_text);
    require(cfg.at("model_type") == "deepseek_v41", "wrong model_type");
    require(cfg.at("text_config").at("num_hidden_layers").get<int>() % 2 == 0, "CED requires even layer count");
    const auto& weight_map = index.at("weight_map");
    require(weight_map.is_object() && !weight_map.empty(), "invalid weight_map");
    std::set<std::string> expected_shards;
    for (const auto& value : weight_map) {
        const std::string name = value;
        require(fs::path(name).filename() == name && name.ends_with(".safetensors"), "invalid shard path");
        require(sources.contains(name), "shard absent from upstream manifest");
        expected_shards.insert(name);
    }
    J issues = J::array(), files = J::array(), shard_info = J::object();
    std::map<std::string, J> tensors, rows;
    U file_bytes = 0, payload_bytes = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        const std::string name = entry.path().filename();
        if (name.ends_with(".safetensors") && !expected_shards.contains(name))
            issues.push_back("unexpected shard: " + name);
    }
    for (const auto& [name, source] : sources)
        if (name.ends_with(".safetensors") && !expected_shards.contains(name))
            issues.push_back("upstream shard absent from index: " + name);
    for (const auto& name : expected_shards) {
        try {
            const auto path = source_path(root, name);
            J h = header(path);
            require(uint(h["file_bytes"]) == uint(sources.at(name).at("size")), "upstream size mismatch");
            // Validate the whole shard before publishing any tensor rows.
            for (const auto& [tensor_name, t] : h["tensors"].items()) {
                require(weight_map.contains(tensor_name) && weight_map[tensor_name] == name, "index/header mismatch: " + tensor_name);
                require(!tensors.contains(tensor_name), "duplicate tensor across shards");
            }
            for (auto [tensor_name, t] : h["tensors"].items()) {
                t["shard"] = name; t["payload_file_offset"] = add(8, uint(h["header_bytes"]));
                tensors.emplace(tensor_name, std::move(t));
            }
            h.erase("tensors"); shard_info[name] = h;
            file_bytes = add(file_bytes, uint(h["file_bytes"]));
            payload_bytes = add(payload_bytes, uint(h["payload_bytes"]));
        } catch (const std::exception& e) { issues.push_back(name + ": " + e.what()); }
    }
    for (const auto& [name, shard] : weight_map.items())
        if (!tensors.contains(name)) issues.push_back("missing tensor: " + name);
    if (payload_bytes != uint(index.at("metadata").at("total_size"))) issues.push_back("index total_size mismatch");

    for (const auto& [name, t] : tensors) {
        try {
            J row = classify(name, t, cfg);
            const bool scale = row["role"] == "quantization_scale";
            if (scale) {
                const std::string weight = name.substr(0, name.size() - 5) + "weight";
                require(tensors.contains(weight), "orphan scale");
                row["scale_for"] = weight;
                require(t.at("dtype") == "F8_E8M0", "unexpected scale dtype");
            } else if (t.at("dtype") == "I8" || t.at("dtype") == "F8_E4M3") {
                require(name.ends_with(".weight"), "quantized tensor without weight suffix");
                const std::string scale_name = name.substr(0, name.size() - 6) + "scale";
                require(tensors.contains(scale_name), "missing quantization scale");
                const auto& s = tensors.at(scale_name);
                require(s.at("dtype") == "F8_E8M0", "invalid quantization scale encoding");
                const auto shape = row.at("logical_shape");
                require(shape.size() == 2, "expected quantized matrix");
                const U r = uint(shape[0]), k = uint(shape[1]);
                J group;
                if (t.at("dtype") == "I8" || row.at("semantic_owner") == "engram_backing") group = J::array({1ULL,32ULL});
                else group = J::array({32ULL,32ULL});
                const U gr = uint(group[0]), gk = uint(group[1]);
                require(r % gr == 0 && k % gk == 0 && s.at("shape") == J::array({r/gr,k/gk}), "scale group/shape mismatch");
                row["scale_tensor"] = scale_name; row["scale_format"] = "E8M0"; row["group_size"] = group;
            }
            if (row.at("semantic_owner") == "engram_backing") {
                const auto& c = cfg.at("text_config");
                size_t position = 0;
                while (c.at("engram_layer_ids").at(position) != row.at("layer_id")) ++position;
                const U count = uint(c.at("engram_num_embeddings").at(position));
                const U dim = uint(c.at("engram_head_dim"));
                require(t.at("shape") == J::array({count,scale ? dim/32 : dim}), "Engram layout mismatch");
            }
            row["shard"] = t.at("shard"); row["data_offsets"] = t.at("data_offsets");
            row["file_offsets"] = J::array({add(uint(t.at("payload_file_offset")),uint(t.at("data_offsets")[0])),
                add(uint(t.at("payload_file_offset")),uint(t.at("data_offsets")[1]))});
            row["revision"] = revision;
            row["digest_scope"] = "whole_shard";
            row["expected_shard_sha256"] = sources.at(t.at("shard").get<std::string>()).at("lfs").at("sha256");
            rows.emplace(name,std::move(row));
        } catch (const std::exception& e) { issues.push_back(name + ": " + e.what()); }
    }
    // Per-file verification includes config, tokenizer, oracle, and index, not just weights.
    std::map<std::string, std::string> verified;
    std::set<std::string> failed_files;
    size_t counter = 0;
    for (const auto& [name, source] : sources) {
        J record = {{"path",name},{"revision",revision},{"status","unverified"}};
        try {
            const auto path = source_path(root,name);
            require(fs::is_regular_file(path), "missing file");
            require(fs::file_size(path) == uint(source.at("size")), "upstream file size mismatch");
            const bool large = name.ends_with(".safetensors");
            if (options.verify_payload || !large) {
                if (large) std::cerr << "hash " << ++counter << '/' << expected_shards.size() << ' ' << name << '\n';
                const bool lfs = source.contains("lfs");
                J digest = hash_file(path,!lfs);
                const std::string expected = lfs ? source.at("lfs").at("sha256") : source.at("blobId");
                const std::string actual = digest.at(lfs ? "sha256" : "git_blob_sha1");
                require(actual == expected, "upstream digest mismatch");
                if (name == "config.json") require(digest.at("sha256") == sha256(config_text), "config changed since parsing");
                if (name == "model.safetensors.index.json") require(digest.at("sha256") == sha256(index_text), "index changed since parsing");
                if (large && shard_info.contains(name))
                    require(header(path).at("header_sha256") == shard_info.at(name).at("header_sha256"), "header changed since inspection");
                record.update(digest); record["expected_digest"] = expected;
                record["verification"] = lfs ? "lfs_sha256" : "git_blob_sha1";
                record["status"] = "verified";
                verified[name] = digest.at("sha256");
            } else record["status"] = "size_checked";
            const fs::path meta = root / ".cache/huggingface/download" / (name+".metadata");
            if (fs::is_regular_file(meta)) {
                std::istringstream lines(read_text(meta)); std::string rev, etag;
                std::getline(lines,rev); std::getline(lines,etag);
                record["download_metadata_revision"] = rev;
                record["download_metadata_etag"] = etag;
                // Downloads can span revisions; the pinned upstream content digest decides validity.
            }
        } catch (const std::exception& e) {
            failed_files.insert(name); record["status"] = "failed"; record["error"] = e.what();
            issues.push_back(name+": "+e.what());
        }
        files.push_back(record);
    }
    J totals = J::object(), owners = J::object(), dtypes = J::object(), logical = J::object();
    J layers = J::object(), shards = J::object(), components = J::object(), residency = J::object();
    fs::create_directories(output);
    std::ofstream atlas(output / "atlas.jsonl"); require(bool(atlas), "cannot create atlas");
    for (auto& [name, row] : rows) {
        const auto shard = row.at("shard").get<std::string>();
        row["completeness"] = failed_files.contains(shard) ? "failed" : verified.contains(shard) ? "verified" : "header-valid";
        row["digest"] = verified.contains(shard) ? J(verified.at(shard)) : J(nullptr);
        add_totals(totals,"all",row); add_totals(owners,row.at("semantic_owner"),row);
        add_totals(dtypes,row.at("storage_dtype"),row); add_totals(logical,row.at("logical_dtype"),row);
        add_totals(shards,shard,row); add_totals(components,row.at("component"),row);
        add_totals(residency,row.at("residency"),row);
        const std::string layer_key = row.at("semantic_owner").get<std::string>() + "/" + row.at("layer_id").dump();
        add_totals(layers,layer_key,row);
        atlas << row.dump() << '\n';
    }
    atlas.close(); require(bool(atlas), "atlas output write failed");
    const bool valid = issues.empty();
    J summary = {{"schema_version",1},{"revision",revision},{"manifest_sha256",sha256(manifest_text)},
        {"status",valid ? options.verify_payload ? "verified" : "header-valid" : "partial_or_invalid"},
        {"expected_tensors",weight_map.size()},{"observed_tensors",tensors.size()},
        {"classified_tensors",rows.size()},{"expected_shards",expected_shards.size()},
        {"valid_headers",shard_info.size()},{"file_bytes",file_bytes},{"payload_bytes",payload_bytes},
        {"by_owner",owners},{"by_storage_dtype",dtypes},{"by_logical_dtype",logical},
        {"by_layer",layers},{"by_shard",shards},{"by_component",components},{"by_residency",residency},
        {"totals",totals},{"shards",shard_info},{"issues",issues},
        {"runtime_mapping_status","payload_accounted_runtime_peak_unqualified"},
        {"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
    write_json(output/"summary.json",summary);
    write_json(output/"verification.json",{{"revision",revision},{"manifest_sha256",sha256(manifest_text)},
        {"status",summary["status"]},{"files",files}});
    std::cout << summary["status"] << ": " << rows.size() << " tensors, " << payload_bytes << " payload bytes, "
              << issues.size() << " issues\n";
    return valid ? 0 : 2;
}
}
