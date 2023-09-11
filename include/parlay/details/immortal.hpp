#pragma once

#include <new>
#include <utility>

namespace parlay {
namespace detail {

// Wraps a value of type T whose destructor is never called.
template<typename T>
struct Immortal {

  template<typename... Args>
  explicit Immortal(Args&&... args) : storage() {
    ::new(storage) T(std::forward<Args>(args)...);
  }

  Immortal(const Immortal&) = delete;
  Immortal& operator=(const Immortal&) = delete;

  T& get() noexcept {
    return *std::launder(reinterpret_cast<T*>(&storage));
  }

  const T& get() const noexcept {
    return *std::launder(reinterpret_cast<const T*>(&storage));
  }

  /* implicit */ operator T&() & noexcept { return get(); }               // NOLINT
  /* implicit */ operator const T&() const & noexcept { return get(); }   // NOLINT

private:
  alignas(T) unsigned char storage[sizeof(T)];
};

}  // namespace detail
}  // namespace parlay
