// A bare minimal atomic_shared_ptr implementation that exists to teach the main algorithm.
// Not efficient and not feature complete. Just for demonstration!!
//
// In particular, the following are absent:
// - No make_shared,
// - No custom deleters/allocators,
// - No weak_ptr,
// - No alias pointers,
// - No enable_shared_from_this
// - No memory orders. All seq_cst.
//
// See atomic_shared_ptr_custom.hpp, shared_ptr.hpp for a feature-complete and optimized implementation!
//

#pragma once

#include <atomic>
#include <memory>

#include <folly/synchronization/Hazptr.h>

namespace parlay {

namespace basic {

template<typename T>
class shared_ptr;

template<typename T>
class atomic_shared_ptr;

}

namespace details {

template<typename T>
struct basic_control_block : public folly::hazptr_obj_base<basic_control_block<T>> {

  template<typename U>
  friend class basic::atomic_shared_ptr;

  template<typename... Args>
  explicit basic_control_block(T* ptr_) noexcept : ref_count(1), ptr(ptr_) { }

  basic_control_block(const basic_control_block &) = delete;
  basic_control_block& operator=(const basic_control_block&) = delete;

  ~basic_control_block() = default;

  // Increment the reference count.  The reference count must not be zero
  void increment_count() noexcept {
    ref_count.fetch_add(1);
  }

  // Increment the reference count if it is not zero.
  bool increment_if_nonzero() noexcept {
    auto cnt = ref_count.load();
    while (cnt > 0 && !ref_count.compare_exchange_weak(cnt, cnt + 1)) { }
    return cnt > 0;
  }

  // Release a reference to the object.
  void decrement_count() noexcept {
    if (ref_count.fetch_sub(1) == 1) {
      delete ptr;
      this->retire();
    }
  }

  std::atomic<long> ref_count;
  T* ptr;
};

}

namespace basic {


template<typename T>
class shared_ptr {

  template<typename U>
  friend class atomic_shared_ptr;

  // Private constructor used by atomic_shared_ptr::load
  explicit shared_ptr(details::basic_control_block<T>* control_block_) : control_block(control_block_) {}

public:

  using element_type = T;

  // Decrement the reference count on destruction.
  ~shared_ptr() noexcept {
    decrement();
  }

  constexpr shared_ptr() noexcept = default;

  constexpr explicit(false) shared_ptr(std::nullptr_t) noexcept {}      // NOLINT(google-explicit-constructor)

  explicit shared_ptr(T* p) {
    std::unique_ptr<T> up(p);     // Hold inside a unique_ptr so that p is deleted if the allocation throws
    control_block = new details::basic_control_block<T>(p);
    up.release();
  }

  shared_ptr(const shared_ptr& other) noexcept : control_block(other.control_block) {
    increment();
  }

  shared_ptr(shared_ptr&& other) noexcept : control_block(std::exchange(other.control_block, nullptr)) { }

  shared_ptr& operator=(const shared_ptr& other) noexcept {
    shared_ptr(other).swap(*this);
    return *this;
  }

  shared_ptr& operator=(shared_ptr&& other) noexcept {
    shared_ptr(std::move(other)).swap(*this);
    return *this;
  }

  void swap(shared_ptr& other) noexcept {
    std::swap(control_block, other.control_block);
  }

  void reset() noexcept {
    shared_ptr().swap(*this);
  }

  void reset(std::nullptr_t) noexcept {
    shared_ptr().swap(*this);
  }

  void reset(T* p) {
    shared_ptr(p).swap(*this);
  }

  T* get() noexcept {
    return control_block ? control_block->ptr : nullptr;
  }

  const T* get() const noexcept {
    return control_block ? control_block->ptr : nullptr;
  }

  [[nodiscard]] T& operator*() noexcept requires (!std::is_void_v<T>) {
    return *(this->get());
  }

  [[nodiscard]] const T& operator*() const noexcept requires (!std::is_void_v<T>) {
    return *(this->get());
  }

  [[nodiscard]] T* operator->() const noexcept {
    return this->get();
  }

  explicit operator bool() const noexcept {
    return this->get() != nullptr;
  }

  [[nodiscard]] long use_count() const noexcept {
    return control_block ? control_block->ref_count.load() : 0;
  }

private:

  void increment() noexcept {
    if (control_block) {
      control_block->increment_count();
    }
  }

  void decrement() noexcept {
    if (control_block) {
      control_block->decrement_count();
    }
  }

  details::basic_control_block<T>* control_block{nullptr};
};

template<typename T1, typename T2>
auto operator<=>(const shared_ptr<T1>& left, const shared_ptr<T2>& right) noexcept {
  return left.get() <=> right.get();
}

template<typename T0>
auto operator<=>(const shared_ptr<T0>& left, std::nullptr_t) noexcept {
  return left.get() <=> static_cast<shared_ptr<T0>::element_type*>(nullptr);
}

template<typename T0>
auto operator<=>(std::nullptr_t, const shared_ptr<T0>& right) noexcept {
  return static_cast<shared_ptr<T0>::element_type*>(nullptr) <=> right.get();
}

template<typename T1, typename T2>
auto operator==(const shared_ptr<T1>& left, const shared_ptr<T2>& right) noexcept {
  return left.get() == right.get();
}

template<typename T0>
auto operator==(const shared_ptr<T0>& left, std::nullptr_t) noexcept {
  return left.get() == static_cast<shared_ptr<T0>::element_type*>(nullptr);
}

template<typename T0>
auto operator==(std::nullptr_t, const shared_ptr<T0>& right) noexcept {
  return static_cast<shared_ptr<T0>::element_type*>(nullptr) == right.get();
}

template<typename T>
class atomic_shared_ptr {
  using shared_ptr_type = shared_ptr<T>;
  using control_block_type = details::basic_control_block<T>;

public:

  constexpr atomic_shared_ptr() noexcept = default;
  constexpr explicit(false) atomic_shared_ptr(std::nullptr_t) noexcept    // NOLINT(google-explicit-constructor)
    : control_block{nullptr} { }

  atomic_shared_ptr(shared_ptr_type desired) {                            // NOLINT(google-explicit-constructor)
    control_block.store(std::exchange(desired.control_block, nullptr));
  }

  atomic_shared_ptr(const atomic_shared_ptr&) = delete;
  atomic_shared_ptr& operator=(const atomic_shared_ptr&) = delete;

  ~atomic_shared_ptr() { store(nullptr); }

  bool is_lock_free() const noexcept {
    return control_block.is_lock_free();
  }

  constexpr static bool is_always_lock_free = std::atomic<control_block_type*>::is_always_lock_free;

  [[nodiscard]] shared_ptr_type load() const {

    folly::hazptr_holder hp = folly::make_hazard_pointer();
    control_block_type* current_control_block = nullptr;

    do {
      current_control_block = hp.protect(control_block);
    } while (current_control_block != nullptr && !current_control_block->increment_if_nonzero());

    return shared_ptr<T>(current_control_block);
  }

  void store(shared_ptr_type desired) {
    auto new_control_block = std::exchange(desired.control_block, nullptr);
    auto old_control_block = control_block.exchange(new_control_block);
    if (old_control_block) {
      old_control_block->decrement_count();
    }
  }

  shared_ptr_type exchange(shared_ptr_type desired) noexcept {
    auto new_control_block = std::exchange(desired.control_block, nullptr);
    auto old_control_block = control_block.exchange(new_control_block);
    return shared_ptr_type(old_control_block);
  }

  bool compare_exchange_weak(shared_ptr_type& expected, shared_ptr_type desired) {
    auto expected_ctrl_block = expected.control_block;
    auto desired_ctrl_block = desired.control_block;

    if (control_block.compare_exchange_weak(expected_ctrl_block, desired_ctrl_block)) {
      if (expected_ctrl_block) {
        expected_ctrl_block->decrement_count();
      }
      desired.control_block = nullptr;
      return true;
    }
    else {
      expected = load();   // It's possible that expected ABAs and stays the same on failure, hence
      return false;        // why this algorithm can not be used to implement compare_exchange_strong
    }
  }

  bool compare_exchange_strong(shared_ptr_type& expected, shared_ptr_type desired) {
    auto expected_ctrl_block = expected.control_block;

    // If expected changes then we have completed the operation (unsuccessfully), we only
    // have to loop in case expected ABAs or the weak operation fails spuriously.
    do {
      if (compare_exchange_weak(expected, desired)) {
        return true;
      }
    } while (expected_ctrl_block == expected.control_block);

    return false;
  }

private:
  mutable std::atomic<control_block_type*> control_block;
};

}  // namespace basic
}  // namespace parlay
