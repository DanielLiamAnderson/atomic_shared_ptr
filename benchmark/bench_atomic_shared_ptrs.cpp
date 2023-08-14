#include <cassert>
#include <cstdlib>

#include <atomic>
#include <chrono>
#include <memory>
#include <utility>

#include <benchmark/benchmark.h>

#include <folly/concurrency/AtomicSharedPtr.h>

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

template<typename T, typename... Args>
inline extern T& get_singleton(Args&&... args) {
  static T t(std::forward<Args>(args)...);
  return t;
}

template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_load(benchmark::State& state) {
  AtomicSharedPtr<int>& src = get_singleton<AtomicSharedPtr<int>>(SharedPtr<int>(new int(42)));

  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    auto result = src.load();
    auto finish = std::chrono::steady_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_store_delete(benchmark::State& state) {
  AtomicSharedPtr<int>& src = get_singleton<AtomicSharedPtr<int>>(SharedPtr<int>(new int(42)));

  // These stores all overwrite the only copy of the pointer, so it will trigger destruction
  // of the managed object. This benchmark therefore also measures the cost of destruction.
  for (auto _ : state) {
    auto new_sp = SharedPtr<int>(new int(rand()));
    auto start = std::chrono::steady_clock::now();
    src.store(std::move(new_sp));
    auto finish = std::chrono::steady_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template<template<typename> typename AtomicSharedPtr, template<typename> typename SharedPtr>
static void bench_store_copy(benchmark::State& state) {
  AtomicSharedPtr<int>& src = get_singleton<AtomicSharedPtr<int>>(SharedPtr<int>(new int(42)));

  auto my_sp = SharedPtr<int>(new int(42));

  // In this version, we keep a copy of our own pointer and store a copy of it, so it will
  // never be destroyed.  This version is therefore only testing the efficiency of store
  // without also testing the efficiency destruction.
  for (auto _ : state) {
    auto new_sp = my_sp;
    auto start = std::chrono::steady_clock::now();
    src.store(std::move(new_sp));
    auto finish = std::chrono::steady_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

constexpr auto compute_low = [](const std::vector<double>& vv) -> double {
  auto v = vv;
  std::nth_element(v.begin(), v.end(), v.begin() + v.size()/100);
  return v[v.size()/100];
};

constexpr auto compute_high = [](const std::vector<double>& vv) -> double {
  auto v = vv;
  std::nth_element(v.begin(), v.end(), v.begin() + v.size()*99/100);
  return v[v.size()*99/100];
};

#define SETUP_BENCHMARK(bench)                    \
  BENCHMARK(bench)                                \
    ->Threads(1)                                  \
    ->UseManualTime()                             \
    ->ComputeStatistics("low", compute_low)       \
    ->ComputeStatistics("high", compute_high);

#define BENCH_PTR(atomic_sp, sp)                          \
  SETUP_BENCHMARK((bench_load<atomic_sp, sp>));           \
  SETUP_BENCHMARK((bench_store_delete<atomic_sp, sp>));   \
  SETUP_BENCHMARK((bench_store_copy<atomic_sp, sp>));

BENCH_PTR(StlAtomicSharedPtr, std::shared_ptr);
BENCH_PTR(folly::atomic_shared_ptr, std::shared_ptr);
BENCH_PTR(parlay::atomic_shared_ptr, parlay::shared_ptr);
