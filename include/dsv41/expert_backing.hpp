#pragma once
#include "dsv41/weights.hpp"
#include <mlx/mlx.h>
#include <filesystem>
#include <memory>
#include <string>
namespace dsv41 {
struct ExpertBackingArrays {
 mlx::core::array weight,scale;
 std::size_t bytes=0;
};
class ExpertBackingStore {
public:
 ExpertBackingStore(const std::filesystem::path& root,const WeightCatalog& catalog);
 ~ExpertBackingStore();
 ExpertBackingStore(const ExpertBackingStore&)=delete;
 ExpertBackingStore& operator=(const ExpertBackingStore&)=delete;
 ExpertBackingArrays projection(int layer,const char* name,int n,int k) const;
private:
 struct Impl;std::unique_ptr<Impl> impl_;
};
// Creates/resumes <destination>.building and publishes destination only after
// all payloads and the complete manifest are durable. Checkpoint is read-only.
void prepare_expert_backing(WeightCatalog& catalog,const std::filesystem::path& destination);
// Repairs only the known temp->final rename ctime transition. Every other
// recorded identity field must still match; payload bytes/digests are untouched.
void repair_expert_backing_identity(const std::filesystem::path& root);
}
