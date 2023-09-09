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

template<template<typename> typename SharedPtr, typename T, typename... Args>
auto dispatch_make_shared(Args... args) {
  if constexpr (std::is_same_v<SharedPtr<T>, std::shared_ptr<T>>) {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }
  else if constexpr (std::is_same_v<SharedPtr<T>, parlay::shared_ptr<T>>) {
    return parlay::make_shared<T>(std::forward<Args>(args)...);
  }
  else if constexpr (std::is_same_v<SharedPtr<T>, jss::shared_ptr<T>>) {
    return jss::make_shared<T>(std::forward<Args>(args)...);
  }
  else if constexpr (std::is_same_v<SharedPtr<T>, LFStructs::SharedPtr<T>>) {
    // No make_shared in Vtyulb's SharedPtr type
    return LFStructs::SharedPtr<T>(new T(std::forward<Args>(args)...));
  }
  else if constexpr (std::is_same_v<SharedPtr<T>, std::experimental::shared_ptr<T>>) {
    return std::experimental::make_shared<T>(std::forward<Args>(args)...);
  }
}


template<template<typename> typename SharedPtr>
static void bench_new(benchmark::State& state) {
  std::unique_ptr<SharedPtr<int>[]> sps = std::make_unique_for_overwrite<SharedPtr<int>[]>(1000000);
  size_t i = 0;
  for (auto _ : state) {
    new (&sps[i++]) SharedPtr<int>{new int(42)};
  }
}

template<template<typename> typename SharedPtr>
static void bench_make(benchmark::State& state) {
  std::unique_ptr<SharedPtr<int>[]> sps = std::make_unique_for_overwrite<SharedPtr<int>[]>(1000000);
  size_t i = 0;
  for (auto _ : state) {
    new (&sps[i++]) SharedPtr<int>{dispatch_make_shared<SharedPtr, int>(42)};
  }
}

template<template<typename> typename SharedPtr>
static void bench_copy(benchmark::State& state) {
  SharedPtr<int> sp{new int(42)};
  std::unique_ptr<SharedPtr<int>[]> sps = std::make_unique_for_overwrite<SharedPtr<int>[]>(1000000);
  size_t i = 0;
  for (auto _ : state) {
    new (&sps[i++]) SharedPtr<int>{sp};
  }
}

template<template<typename> typename SharedPtr>
static void bench_decrement(benchmark::State& state) {
  SharedPtr<int> sp{new int(42)};
  std::unique_ptr<SharedPtr<int>[]> sps = std::make_unique_for_overwrite<SharedPtr<int>[]>(1000000);
  for (size_t i = 0; i < 1000000; i++) {
    new (&sps[i++]) SharedPtr<int>{sp};
  }
  size_t i = 0;
  for (auto _ : state) {
    sps[i++].~SharedPtr<int>();
  }
  sps.release();
}

template<template<typename> typename SharedPtr>
static void bench_destroy(benchmark::State& state) {
  std::unique_ptr<SharedPtr<int>[]> sps = std::make_unique_for_overwrite<SharedPtr<int>[]>(1000000);
  for (size_t i = 0; i < 1000000; i++) {
    new (&sps[i++]) SharedPtr<int>{new int(i)};
  }
  size_t i = 0;
  for (auto _ : state) {
    sps[i++].~SharedPtr<int>();
  }
  sps.release();
}

#define SETUP_BENCHMARK(ptr_name, bench_name, bench)       \
  BENCHMARK(bench)                                         \
    ->Name(ptr_name "::" bench_name)                       \
    ->UseRealTime()                                        \
    ->Iterations(1000000);

#define BENCH_PTR(name, sp)                                 \
  SETUP_BENCHMARK(name, "new", (bench_new<sp>));            \
  SETUP_BENCHMARK(name, "make", (bench_make<sp>));          \
  SETUP_BENCHMARK(name, "decrement", (bench_make<sp>));     \
  SETUP_BENCHMARK(name, "destroy", (bench_make<sp>));


BENCH_PTR("STL", std::shared_ptr);
BENCH_PTR("Mine", parlay::shared_ptr);
BENCH_PTR("JSS-Free", jss::shared_ptr);
BENCH_PTR("Vtyulb", LFStructs::SharedPtr);

#ifdef JUST_THREADS_AVAILABLE
BENCH_PTR("JSS", std::experimental::shared_ptr);
#endif

