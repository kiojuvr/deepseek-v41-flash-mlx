#include "bridge_executor.hpp"
#include "dsv41/text_generate.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
namespace dsv41 {
namespace {
class ModelExecutor final:public BridgeExecutor {
 // Declaration order keeps the catalog and metadata alive beyond the model.
 WeightCatalog catalog_;
 std::shared_ptr<const EngramMetadata> metadata_;
 std::unique_ptr<TextGenerationReference> model_;
 std::vector<uint32_t> stops_;
public:
 explicit ModelExecutor(const dsv41_model_config_t& config)
  :catalog_(config.checkpoint_path,config.summary_path){
  auto proof=read_json_file(config.provenance_path);
  std::ifstream file(config.metadata_path,std::ios::binary);
  if(!file)throw std::runtime_error("cannot open Engram metadata");
  std::string raw{std::istreambuf_iterator<char>(file),{}};
  if(sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())
   throw std::runtime_error("Engram metadata identity mismatch");
  metadata_=EngramMetadata::load(config.metadata_path);
  mlx::core::set_default_device(mlx::core::Device::gpu);
  model_=std::make_unique<TextGenerationReference>(catalog_,metadata_);
  if(config.stop_token_count)stops_.assign(config.stop_tokens,config.stop_tokens+config.stop_token_count);
 }
 GenerationResult generate(const dsv41_request_t& request,const GenerationControl& control) override{
  mlx::core::set_default_device(mlx::core::Device::gpu);
  return model_->generate({request.input_tokens,request.input_token_count},request.max_new_tokens,
                         {request.temperature,request.seed},stops_,control);
 }
};
}
std::unique_ptr<BridgeExecutor> make_bridge_executor(const dsv41_model_config_t& config){
 return std::make_unique<ModelExecutor>(config);
}
}
