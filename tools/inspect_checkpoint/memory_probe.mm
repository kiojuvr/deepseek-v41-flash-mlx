#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>
#include <iostream>
#include <nlohmann/json.hpp>

int main() {
    @autoreleasepool {
        std::uint64_t physical = 0;
        size_t length = sizeof(physical);
        if (sysctlbyname("hw.memsize", &physical, &length, nullptr, 0) != 0) {
            std::cerr << "cannot query hw.memsize\n";
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::cerr << "no Metal device visible\n"; return 1; }
        nlohmann::json result = {
            {"physical_memory_bytes",physical}, {"device_name",[[device name] UTF8String]},
            {"has_unified_memory",static_cast<bool>(device.hasUnifiedMemory)},
            {"recommended_max_working_set_bytes",device.recommendedMaxWorkingSetSize},
            {"max_buffer_length_bytes",device.maxBufferLength},
            {"current_allocated_bytes",device.currentAllocatedSize},
            {"os",[[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String]},
            {"note","Read-only device limits; no runtime allocation or residency qualification."}};
        std::cout << result.dump(2) << '\n';
    }
}
