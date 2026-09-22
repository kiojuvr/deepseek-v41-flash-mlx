#pragma once
#include <cstddef>
namespace dsv41 {
// Opt-in process-global Metal budget. Declare before every model-owned array.
// Candidate deliberately rejects concurrent backbone owners when enabled.
class RuntimeResidencyLease {
public:
 RuntimeResidencyLease();
 ~RuntimeResidencyLease() noexcept;
 RuntimeResidencyLease(const RuntimeResidencyLease&)=delete;
 RuntimeResidencyLease& operator=(const RuntimeResidencyLease&)=delete;
 std::size_t requested_bytes() const{return requested_;}
 void activate();
 void synchronize() const;
private:
 std::size_t requested_=0;
 bool active_=false;
};
// Declaration immediately after the expert atlas creates all no-copy views
// before one residency-set resize, while still preceding dense model members.
class RuntimeResidencyActivation {
public:
 explicit RuntimeResidencyActivation(RuntimeResidencyLease& lease){lease.activate();}
};
}
