#include "dsv41/linear.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

namespace mx = mlx::core;
int main(int argc, char** argv) { try {
    const bool norm_only = argc == 6 && std::string(argv[5]) == "--norm-input";
    if (argc != 5 && !norm_only) throw std::runtime_error("usage: dsv41-q-projection checkpoint summary input-directory output-directory [--norm-input]");
    const auto output = std::filesystem::weakly_canonical(argv[4]);
    for (const auto* input : {argv[1], argv[3]}) {
        auto relative = output.lexically_relative(std::filesystem::canonical(input));
        if (relative.empty() || *relative.begin() != "..") throw std::runtime_error("output must be outside inputs");
    }
    if (std::filesystem::exists(output)) throw std::runtime_error("output must be fresh");
    auto manifest = dsv41::read_json_file(std::filesystem::path(argv[3]) / "manifest.json");
    const auto ids = norm_only ? "sample_ids" : "token_ids";
    const auto count = manifest.at(ids).size();
    if (count < 1 || count > 128) throw std::runtime_error("requires 1..128 tokens");
    mx::set_default_device(mx::Device::gpu);
    auto input = mx::load((std::filesystem::path(argv[3]) / (norm_only ? "encoder.layer0.attn_qa.npy" : "encoder.layer0.attn_in.npy")).string());
    const int width = norm_only ? 1280 : 5120;
    if (input.dtype() != mx::bfloat16 || input.shape() != mx::Shape{int(count), width})
        throw std::runtime_error("invalid input shape/dtype");
    dsv41::WeightCatalog catalog(argv[1], argv[2]);
    std::unique_ptr<dsv41::PackedLinearReference> projection;
    if (!norm_only) projection = std::make_unique<dsv41::PackedLinearReference>(catalog, "layers.0.attn.wq_a");
    auto tensor = catalog.tensor("layers.0.attn.q_norm.weight");
    if (tensor.dtype != "BF16" || tensor.shape != std::vector<std::uint64_t>{1280})
        throw std::runtime_error("invalid norm weight");
    std::vector<std::uint16_t> data(1280);
    tensor.read(0, {reinterpret_cast<std::byte*>(data.data()), data.size()*2});
    auto weight = mx::view(mx::array(data.begin(), {1280}, mx::uint16), mx::bfloat16);
    std::vector<mx::array> qa, qr;
    std::map<std::string, std::vector<mx::array>> stages;
    for (int t = 0; t < int(count); ++t) {
        auto row = mx::slice(input, {t, 0}, {t+1, width});
        qa.push_back(norm_only ? row : projection->forward(row));
        qr.push_back(dsv41::rms_norm_reference(qa.back(), weight, 1e-20f));
        mx::eval(qa.back(), qr.back());
        auto h = mx::astype(qa.back(), mx::float32);
        auto variance = mx::mean(mx::square(h), -1, true);
        auto adjusted = mx::add(variance, mx::array(1e-20f));
        auto inv = mx::divide(mx::array(1.0f), mx::sqrt(adjusted));
        auto normalized = mx::multiply(h, inv);
        auto weighted = mx::multiply(mx::astype(weight, mx::float32), normalized);
        for (const auto& [name, value] : std::vector<std::pair<std::string, mx::array>>{
                {"variance", variance}, {"adjusted", adjusted}, {"inv", inv},
                {"normalized", normalized}, {"weighted", weighted}}) {
            mx::eval(value);
            stages[name].push_back(value);
        }
    }
    std::filesystem::create_directories(output);
    auto arrays = nlohmann::json::array();
    for (const auto& [stage, values] : stages) {
        const auto name = "encoder.layer0.qnorm_" + stage;
        auto value = mx::concatenate(values, 0);
        mx::eval(value);
        mx::save((output / (name + ".npy")).string(), value);
        arrays.push_back({{"name", name}, {"dtype", "float32"}, {"shape", value.shape()}});
    }
    for (const auto& [name, value] : std::vector<std::pair<std::string, mx::array>>{
            {"encoder.layer0.attn_qa", mx::concatenate(qa, 0)},
            {"encoder.layer0.attn_qr", mx::concatenate(qr, 0)}}) {
        mx::eval(value);
        mx::save((output / (name + ".npy")).string(), value);
        arrays.push_back({{"name", name}, {"dtype", "bfloat16"}, {"shape", {count, 1280}}});
    }
    nlohmann::json result = {{"schema_version", 1}, {ids, manifest.at(ids)},
        {"arrays", arrays}, {"revision", catalog.revision()},
        {"scope", norm_only ? "Synthetic Q normalization; not a token trace or full-model qualification" : "Native Q projection and norm on frozen attn_in; not full-model qualification"}};
    std::ofstream file(output / "manifest.json"); file << result.dump(2) << '\n';
    if (!file) throw std::runtime_error("manifest write failed");
    std::cout << "Completed native Q projection diagnostic\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
