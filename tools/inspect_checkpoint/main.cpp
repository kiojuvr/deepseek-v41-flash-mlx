#include "dsv41/checkpoint_atlas.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        dsv41::AtlasOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "dsv41-inspect-checkpoint --checkpoint DIR --output DIR "
                             "--manifest HF_JSON [--verify-payload]\n";
                return 0;
            }
            if (flag == "--verify-payload") { options.verify_payload = true; continue; }
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            if (flag == "--checkpoint") options.checkpoint = argv[++i];
            else if (flag == "--output") options.output = argv[++i];
            else if (flag == "--manifest") options.manifest = argv[++i];
            else throw std::runtime_error("unknown argument: " + flag);
        }
        if (options.checkpoint.empty() || options.output.empty() || options.manifest.empty())
            throw std::runtime_error("--checkpoint, --output, and --manifest are required");
        return dsv41::inspect_checkpoint(options);
    } catch (const std::exception& e) {
        std::cerr << "checkpoint atlas: " << e.what() << '\n';
        return 1;
    }
}
