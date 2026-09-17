// Audit-only process-local hook for Metal compute dispatches.
// It is built as a DYLD-inserted dylib and never linked into production.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

struct Shape {
  bool threads;
  MTLSize grid;
  MTLSize group;

  bool operator<(const Shape& other) const {
    return std::tie(threads, grid.width, grid.height, grid.depth, group.width,
                    group.height, group.depth) <
           std::tie(other.threads, other.grid.width, other.grid.height,
                    other.grid.depth, other.group.width, other.group.height,
                    other.group.depth);
  }
};

struct State {
  std::atomic<bool> enabled{true};
  std::atomic<bool> scoped{false};
  std::atomic<unsigned long long> thread_dispatches{0};
  std::atomic<unsigned long long> threadgroup_dispatches{0};
  std::mutex mutex;
  std::map<Shape, unsigned long long> shapes;
  IMP original_threads{nullptr};
  IMP original_threadgroups{nullptr};
};

State& state() {
  // Deliberately leaked so the destructor hook can safely write after static
  // C++ destruction has begun.
  static State* value = new State;
  return *value;
}

void record(bool threads, MTLSize grid, MTLSize group) {
  auto& s = state();
  if (!s.enabled.load(std::memory_order_relaxed)) {
    return;
  }
  if (threads) {
    s.thread_dispatches.fetch_add(1, std::memory_order_relaxed);
  } else {
    s.threadgroup_dispatches.fetch_add(1, std::memory_order_relaxed);
  }
  std::lock_guard<std::mutex> lock(s.mutex);
  ++s.shapes[{threads, grid, group}];
}

extern "C" __attribute__((visibility("default")))
void dsv41_metal_dispatch_counter_reset() {
  auto& s = state();
  s.thread_dispatches.store(0, std::memory_order_relaxed);
  s.threadgroup_dispatches.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(s.mutex);
  s.shapes.clear();
}

extern "C" __attribute__((visibility("default")))
void dsv41_metal_dispatch_counter_set_enabled(int enabled) {
  auto& s = state();
  s.scoped.store(true, std::memory_order_relaxed);
  s.enabled.store(enabled != 0, std::memory_order_relaxed);
}

void replacement_threads(id self, SEL command, MTLSize grid, MTLSize group) {
  record(true, grid, group);
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(
      state().original_threads)(self, command, grid, group);
}

void replacement_threadgroups(
    id self,
    SEL command,
    MTLSize grid,
    MTLSize group) {
  record(false, grid, group);
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(
      state().original_threadgroups)(self, command, grid, group);
}

void hook(Class encoder_class, SEL selector, IMP replacement, IMP& original) {
  Method method = class_getInstanceMethod(encoder_class, selector);
  if (method == nullptr) {
    return;
  }
  IMP implementation = method_getImplementation(method);
  if (implementation == replacement) {
    return;
  }
  if (original == nullptr) {
    original = implementation;
  }
  method_setImplementation(method, replacement);
}

__attribute__((constructor)) void install_hooks() {
  @autoreleasepool {
    if (const char* scoped = std::getenv("DSV41_METAL_DISPATCH_COUNTER_SCOPED");
        scoped != nullptr && std::string(scoped) == "1") {
      state().enabled.store(false, std::memory_order_relaxed);
      state().scoped.store(true, std::memory_order_relaxed);
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    auto& s = state();
    hook(object_getClass(encoder),
         @selector(dispatchThreads:threadsPerThreadgroup:),
         reinterpret_cast<IMP>(replacement_threads), s.original_threads);
    hook(object_getClass(encoder),
         @selector(dispatchThreadgroups:threadsPerThreadgroup:),
         reinterpret_cast<IMP>(replacement_threadgroups),
         s.original_threadgroups);
    [encoder endEncoding];
  }
}

__attribute__((destructor)) void write_json() {
  const char* path = std::getenv("DSV41_METAL_DISPATCH_COUNTER_OUTPUT");
  if (path == nullptr || *path == '\0') {
    return;
  }
  auto& s = state();
  std::ofstream output(path, std::ios::trunc);
  const auto threads = s.thread_dispatches.load(std::memory_order_relaxed);
  const auto groups =
      s.threadgroup_dispatches.load(std::memory_order_relaxed);
  output << "{\n  \"scope\": \"audit-only "
         << (s.scoped.load(std::memory_order_relaxed) ? "prefill-scoped" : "target-process")
         << " Metal compute dispatches\",\n"
         << "  \"dispatch_threads\": " << threads << ",\n"
         << "  \"dispatch_threadgroups\": " << groups << ",\n"
         << "  \"dispatch_total\": " << (threads + groups) << ",\n"
         << "  \"shapes\": [\n";
  bool first = true;
  std::lock_guard<std::mutex> lock(s.mutex);
  for (const auto& [shape, count] : s.shapes) {
    if (!first) {
      output << ",\n";
    }
    first = false;
    output << "    {\"kind\": \""
           << (shape.threads ? "threads" : "threadgroups") << "\", "
           << "\"grid\": [" << shape.grid.width << ", " << shape.grid.height
           << ", " << shape.grid.depth << "], \"group\": ["
           << shape.group.width << ", " << shape.group.height << ", "
           << shape.group.depth << "], \"count\": " << count << "}";
  }
  output << "\n  ]\n}\n";
}
