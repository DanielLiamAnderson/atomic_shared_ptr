#pragma once

#include <memory>

#include <folly/concurrency/AtomicSharedPtr.h>

#include "external/anthonywilliams/atomic_shared_ptr"
#include "external/vtyulb/atomic_shared_ptr.h"

#ifdef JUST_THREADS_AVAILABLE
#include <experimental/atomic>
#endif

#include "parlay/atomic_shared_ptr_custom.hpp"

#include "parlay/basic_atomic_shared_ptr.hpp"

#ifdef __cpp_lib_atomic_shared_ptr

// C++ standard library atomic support for shared ptrs
template<typename T>
using StlAtomicSharedPtr = std::atomic<std::shared_ptr<T>>;

#else

// Use free functions if std::atomic<shared_ptr> is not available. Much worse.
template<typename T>
struct StlAtomicSharedPtr {
  StlAtomicSharedPtr() = default;
  explicit(false) StlAtomicSharedPtr(std::shared_ptr<T> other) : sp(std::move(other)) { }   // NOLINT
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