#include <cassert>
#include <cstdlib>

#include <atomic>
#include <chrono>
#include <memory>
#include <utility>

#include <benchmark/benchmark.h>

#include <folly/concurrency/AtomicSharedPtr.h>

#include "external/anthonywilliams/atomic_shared_ptr"
#include "external/vtyulb/atomic_shared_ptr.h"

#ifdef JUST_THREADS_AVAILABLE
#include <experimental/atomic>
#endif

#include "parlay/atomic_shared_ptr_custom.hpp"

#ifdef __cpp_lib_atomic_shared_ptr

// C++ standard library atomic support for shared ptrs
template<typename T>
using StlAtomicSharedPtr = std::atomic<std::shared_ptr<T>>;

#else

// Use free functions if std::atomic<shared_ptr> is not available. Much worse.
template<typename T>
struct StlAtomicSharedPtr {
  StlAtomicSharedPtr() = default;
  StlAtomicSharedPtr(std::shared_ptr<T> other) : sp(std::move(other)) { }
  std::shared_ptr<T> load() { return std::atomic_load(&sp); }
  void store(std::shared_ptr<T> r) { std::atomic_store(&sp, std::move(r)); }
  bool compare_exchange_strong(std::shared_ptr<T>& expected, std::shared_ptr<T> desired) {
    return atomic_compare_exchange_strong(&sp, &expected, std::move(desired));
  }
  bool compare_exchange_weak(std::shared_ptr<T>& expected, std::shared_ptr<T> desired) {
    return atomic_compare_exchange_weak(&sp, &expected, std::move(desired));
  }
  std::shared_ptr<T> sp;
};

#endif


constexpr auto compute_low = [](std::vector<double>& v) -> double {
  std::nth_element(v.begin(), v.begin() + v.size()/100, v.end());
  return v[v.size()/100];
};

constexpr auto compute_med = [](std::vector<double>& v) -> double {
  std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
  return v[v.size()/100];
};

constexpr auto compute_high = [](std::vector<double>& v) -> double {
  std::nth_element(v.begin(), v.begin() + v.size()*99/100, v.end());
  return v[v.size()*99/100];
};


template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_load(benchmark::State& state) {
  AtomicSharedPtr<int> src;
  src.store(SharedPtr<int>(new int(42)));

  std::vector<double> all_times;

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = src.load();
    auto finish = std::chrono::high_resolution_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_time.count());
    all_times.push_back(elapsed_time.count());
  }

  state.counters["1%"] = compute_low(all_times);
  state.counters["50%"] = compute_med(all_times);
  state.counters["99%"] = compute_high(all_times);
}

template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_store_delete(benchmark::State& state) {
  AtomicSharedPtr<int> src;
  src.store(SharedPtr<int>(new int(42)));

  std::vector<double> all_times;

  // These stores all overwrite the only copy of the pointer, so it will trigger destruction
  // of the managed object. This benchmark therefore also measures the cost of destruction.
  for (auto _ : state) {
    auto new_sp = SharedPtr<int>(new int(rand()));
    auto start = std::chrono::high_resolution_clock::now();
    src.store(std::move(new_sp));
    auto finish = std::chrono::high_resolution_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_time.count());
    all_times.push_back(elapsed_time.count());
  }

  state.counters["1%"] = compute_low(all_times);
  state.counters["50%"] = compute_med(all_times);
  state.counters["99%"] = compute_high(all_times);
}

template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_store_copy(benchmark::State& state) {
  AtomicSharedPtr<int> src;
  src.store(SharedPtr<int>(new int(42)));

  auto my_sp = SharedPtr<int>(new int(42));

  std::vector<double> all_times;

  // In this version, we keep a copy of our own pointer and store a copy of it, so it will
  // never be destroyed.  This version is therefore only testing the efficiency of store
  // without also testing the efficiency destruction.
  for (auto _ : state) {
    auto new_sp = my_sp;
    auto start = std::chrono::high_resolution_clock::now();
    src.store(std::move(new_sp));
    auto finish = std::chrono::high_resolution_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_time.count());
    all_times.push_back(elapsed_time.count());
  }

  state.counters["1%"] = compute_low(all_times);
  state.counters["50%"] = compute_med(all_times);
  state.counters["99%"] = compute_high(all_times);
}



#define SETUP_BENCHMARK(ptr_name, bench_name, bench)       \
  BENCHMARK(bench)                                         \
    ->Name(ptr_name "::" bench_name)                       \
    ->UseManualTime();

#define BENCH_PTR(name, atomic_sp, sp)                                        \
  SETUP_BENCHMARK(name, "load", (bench_load<atomic_sp, sp>));                 \
  SETUP_BENCHMARK(name, "store", (bench_store_copy<atomic_sp, sp>));          \
  SETUP_BENCHMARK(name, "store-del", (bench_store_delete<atomic_sp, sp>));


BENCH_PTR("STL", StlAtomicSharedPtr, std::shared_ptr);
BENCH_PTR("Folly", folly::atomic_shared_ptr, std::shared_ptr);
BENCH_PTR("Mine", parlay::atomic_shared_ptr, parlay::shared_ptr);
BENCH_PTR("JSS-Free", jss::atomic_shared_ptr, jss::shared_ptr);
BENCH_PTR("Vtyulb", LFStructs::AtomicSharedPtr, LFStructs::SharedPtr);

#ifdef JUST_THREADS_AVAILABLE
BENCH_PTR("JSS", std::experimental::atomic_shared_ptr, std::experimental::shared_ptr);
#endif
