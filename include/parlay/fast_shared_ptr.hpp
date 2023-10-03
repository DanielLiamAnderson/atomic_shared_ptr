// A faster shared_ptr that omits most of the bells and whistles in order
// to make the control block smaller and remove all type erasure.
//
// In particular, the following are absent:
// - No make_shared,
// - No custom deleters/allocators,
// - No weak_ptr,
// - No alias pointers,
// - No enable_shared_from_this
//
// The benefit is that the control block is only 16 bytes at minimum
// because it has no weak ref count and no vtable pointer.
//
// See shared_ptr.hpp for a feature-complete implementation!
//
#pragma once

#include <atomic>
#include <memory>
#include <variant>

#include "details/hazard_pointers.hpp"
#include "details/wait_free_counter.hpp"

#include <parlay/alloc.h>

namespace parlay {

template<typename T>
class atomic_shared_ptr;

template<typename T>
class shared_ptr;

namespace details {


using ref_cnt_type = uint32_t;


// Minimal, optimized control block.  No alias support, or custom deleter, or custom allocator.
template<typename T>
struct fast_control_block {

  static void* operator new(std::size_t sz) {
    assert(sz == 1);
    return parlay::type_allocator<fast_control_block>::alloc();
  }

  static void operator delete(void* ptr) {
    parlay::type_allocator<fast_control_block>::free(static_cast<fast_control_block*>(ptr));
  }

  struct inline_tag {};

  template<typename U>
  friend class atomic_shared_ptr;

  fast_control_block(T* ptr_) : strong_count(1), ptr(ptr_), inline_alloc(false) { }    // NOLINT

  template<typename... Args>
  fast_control_block(inline_tag, Args&&... args)                                 // NOLINT
    : strong_count(1), object(std::forward<Args>(args)...), inline_alloc(true) { }


  fast_control_block(const fast_control_block &) = delete;
  fast_control_block& operator=(const fast_control_block&) = delete;

  ~fast_control_block() { }

  // Destroy the managed object.  Called when the strong count hits zero
  void dispose() noexcept {
    if (inline_alloc) {
      object.~T();
    }
    else {
      delete ptr;
    }
  }

  // Destroy the control block.  dispose() must have been called prior to
  // calling destroy.  Called when the weak count hits zero.
  void destroy() noexcept {
    delete this;
  }

  // Increment the strong reference count.  The strong reference count must not be zero
  void increment_strong_count() noexcept {
    assert(strong_count.load(std::memory_order_relaxed) > 0);
    [[maybe_unused]] auto success = strong_count.increment(1, std::memory_order_relaxed);
    assert(success);
  }

  // Increment the strong reference count if it is not zero. Return true if successful,
  // otherwise return false indicating that the strong reference count is zero.
  bool increment_strong_count_if_nonzero() noexcept {
    return strong_count.increment(1, std::memory_order_relaxed);
  }

  void decrement_strong_count() noexcept {
    if (strong_count.decrement(1, std::memory_order_release)) {
      std::atomic_thread_fence(std::memory_order_acquire);
      dispose();
      get_hazard_list<fast_control_block>().retire(this);
    }
  }

  fast_control_block* get_next() const noexcept { return next_; }
  void set_next(fast_control_block* next) noexcept { next_ = next; }

  T* get_ptr() const noexcept {
    if (inline_alloc) return const_cast<T*>(std::addressof(object));
    else return const_cast<T*>(ptr);
  }

  auto get_use_count() const noexcept { return strong_count.load(std::memory_order_relaxed); }

private:

  WaitFreeCounter<ref_cnt_type> strong_count;
  const bool inline_alloc;

  union {
    std::monostate empty;
    fast_control_block* next_;          // Intrusive ptr used for garbage collection by Hazard pointers
    T* ptr;                             // Pointer to the managed object while it is alive
    T object;
  };

};

static_assert(sizeof(fast_control_block<int>) == 16);
static_assert(sizeof(fast_control_block<uintptr_t>) == 16);

// Base class for shared_ptr and weak_ptr
template<typename T>
class smart_ptr_base {

  template<typename U>
  friend class atomic_shared_ptr;

public:
  using element_type = T;

  [[nodiscard]] long use_count() const noexcept {
    return control_block ? control_block->get_use_count() : 0;
  }

  // Comparator for sorting shared pointers.  Ordering is based on the address of the control blocks.
  template<typename T2>
  [[nodiscard]] bool owner_before(const smart_ptr_base<T2>& other) const noexcept {
    return control_block < other.control_block;
  }

  smart_ptr_base& operator=(const smart_ptr_base&) = delete;

  [[nodiscard]] element_type* get() const noexcept {
    return control_block ? control_block->get_ptr() : nullptr;
  }

protected:

  constexpr smart_ptr_base() noexcept = default;

  smart_ptr_base(fast_control_block<T>* control_block_) noexcept
    : control_block(control_block_)  {

  }


  explicit smart_ptr_base(const smart_ptr_base& other) noexcept
    :  control_block(other.control_block) {

  }


  explicit smart_ptr_base(smart_ptr_base&& other) noexcept
    : control_block(std::exchange(other.control_block, nullptr)) {

  }

  ~smart_ptr_base() = default;

  void swap_ptrs(smart_ptr_base& other) noexcept {
    //std::swap(ptr, other.ptr);
    std::swap(control_block, other.control_block);
  }

  void increment_strong() const noexcept {
    if (control_block) {
      control_block->increment_strong_count();
    }
  }

  [[nodiscard]] bool increment_if_nonzero() const noexcept {
    return control_block && control_block->increment_strong_count_if_nonzero();
  }

  void decrement_strong() noexcept {
    if (control_block) {
      control_block->decrement_strong_count();
    }
  }

  //element_type* ptr{nullptr};
  fast_control_block<T>* control_block{nullptr};
};

}  // namespace details

template<typename T>
class shared_ptr : public details::smart_ptr_base<T> {

  using base = details::smart_ptr_base<T>;

  template<typename U>
  friend class atomic_shared_ptr;

  template<typename T0>
  friend class shared_ptr;

  // Private constructor used by atomic_shared_ptr::load and weak_ptr::lock
  shared_ptr([[maybe_unused]] T* ptr_, details::fast_control_block<T>* control_block_) : base(control_block_) {
    assert(ptr_ == control_block_->get_ptr() && "This shared_ptr does not support alias pointers.");
  }

public:
  using typename base::element_type;

  // Decrement the reference count on destruction.  Resource cleanup is all
  // handled internally by the control block (including deleting itself!)
  ~shared_ptr() noexcept {
    this->decrement_strong();
  }

  // ==========================================================================================
  //                              INITIALIZING AND NULL CONSTRUCTORS
  // ==========================================================================================

  constexpr shared_ptr() noexcept = default;

  constexpr explicit(false) shared_ptr(std::nullptr_t) noexcept {}      // NOLINT(google-explicit-constructor)

  explicit shared_ptr(T* p) {
    std::unique_ptr<T> up(p);     // Hold inside a unique_ptr so that p is deleted if the allocation throws
    auto control_block = new details::fast_control_block<T>(p);
    this->set_ptrs_and_esft(up.release(), control_block);
  }


  // ==========================================================================================
  //                                  COPY CONSTRUCTORS
  // ==========================================================================================

  shared_ptr(const shared_ptr& other) noexcept : base(other) {
    this->increment_strong();
  }

  // ==========================================================================================
  //                                  MOVE CONSTRUCTORS
  // ==========================================================================================

  shared_ptr(shared_ptr&& other) noexcept : base(std::exchange(other.control_block, nullptr)) { }

  // ==========================================================================================
  //                                  ASSIGNMENT OPERATORS
  // ==========================================================================================

  shared_ptr& operator=(const shared_ptr& other) noexcept {
    shared_ptr(other).swap(*this);
    return *this;
  }

  shared_ptr& operator=(shared_ptr&& other) noexcept {
    shared_ptr(std::move(other)).swap(*this);
    return *this;
  }

  // ==========================================================================================
  //                                    SWAP, RESET
  // ==========================================================================================

  void swap(shared_ptr& other) noexcept {
    this->swap_ptrs(other);
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

  // ==========================================================================================
  //                                    ACCESS, DEREFERENCE
  // ==========================================================================================

  [[nodiscard]] T& operator*() const noexcept requires (!std::is_void_v<T>) {
    return *(this->get());
  }

  [[nodiscard]] T* operator->() const noexcept {
    return this->get();
  }

  explicit operator bool() const noexcept {
    return this->get() != nullptr;
  }

  // ==========================================================================================
  //                                       FACTORIES
  // ==========================================================================================

  template<typename T0, typename... Args>
  //  requires std::constructible_from<T, Args...>
  friend shared_ptr<T0> make_shared(Args&&... args);

private:

  void set_ptrs_and_esft(T* ptr_, details::fast_control_block<T>* control_block_) {
    //this->ptr = ptr_;
    this->control_block = control_block_;
  }

  // Release the ptr and control_block to the caller.  Does not modify the reference count,
  // so the caller is responsible for taking over the reference count owned by this copy
  std::pair<T*, details::fast_control_block<T>*> release_internals() noexcept {
    auto p = this->control_block ? this->control_block->get_ptr() : nullptr;
    return std::make_pair(p, std::exchange(this->control_block, nullptr));
  }

};

// ==========================================================================================
//                    IMPLEMENTATIONS OF PREDECLARED FRIEND FUNCTIONS
// ==========================================================================================

template<typename Deleter, typename T>
Deleter* get_deleter(const shared_ptr<T>& sp) noexcept {
  if (sp.control_block) {
    return static_cast<Deleter*>(sp.control_block.get_deleter(typeid(Deleter)));
  }
  return nullptr;
}

template<typename T, typename... Args>
[[nodiscard]] shared_ptr<T> make_shared(Args&&... args) {
  const auto control_block = new details::fast_control_block<T>(typename details::fast_control_block<T>::inline_tag{}, std::forward<Args>(args)...);
  assert(control_block != nullptr);
  assert(control_block->get_ptr() != nullptr);
  shared_ptr<T> result(control_block->get_ptr(), control_block);
  return result;
}


// ==========================================================================================
//                                       COMPARISON
// ==========================================================================================

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


}  // namespace parlay
