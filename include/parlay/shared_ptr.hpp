// A custom modern shared_ptr and weak_ptr implementation used by the atomic_shared_ptr.

// It tries to closely match the standard library std::shared_ptr as much as possible. Most 
// of the code roughly follows the same implementation strategies as libstdc++, libc++, and
// Microsoft STL.  The main difference is using Hazard Pointer deferred reclaimation on the
// control block to allow atomic_shared_ptr to be lock free and not require a split reference
// count.
//
// No support for std::shared_ptr<T[]>, i.e., shared pointers of arrays. They should not exist.
//

#pragma once

#include <atomic>
#include <memory>

#include "details/hazard_pointers.hpp"
#include "details/wait_free_counter.hpp"

namespace parlay {

template<typename T>
class atomic_shared_ptr;

template<typename T>
class shared_ptr;

template<typename T>
class weak_ptr;

namespace details {


using ref_cnt_type = uint32_t;


// Base class of all control blocks used by smart pointers
struct control_block_base {
  
  template<typename T>
  friend class atomic_shared_ptr;

  template<typename... Args>
  explicit control_block_base(void* ptr_) noexcept : strong_count(1), weak_count(1), ptr(ptr_) { }

  control_block_base(const control_block_base &) = delete;
  control_block_base& operator=(const control_block_base&) = delete;

  virtual ~control_block_base() = default;
  
  // Destroy the managed object.  Called when the strong count hits zero
  virtual void dispose() noexcept = 0;
  
  // Destroy the control block.  dispose() must have been called prior to
  // calling destroy.  Called when the weak count hits zero.
  virtual void destroy() noexcept = 0;

  // Return the custom deleter for this object if the deleter has the type,
  // indicated by the argument, otherwise return nullptr
  virtual void* get_deleter(std::type_info&) const noexcept { return nullptr; }
  
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
  
  // Release a strong reference to the object. If the strong reference count hits zero,
  // the object is disposed and the weak reference count is decremented. If the weak
  // reference count also reaches zero, the object is immediately destroyed.
  void decrement_strong_count() noexcept {

    // A decrement-release + an acquire fence is recommended by Boost's documentation:
    // https://www.boost.org/doc/libs/1_57_0/doc/html/atomic/usage_examples.html
    // Alternatively, an acquire-release decrement would work, but might be less efficient
    // since the acquire is only relevant if the decrement zeros the counter.
    if (strong_count.decrement(1, std::memory_order_release)) {
      std::atomic_thread_fence(std::memory_order_acquire);
      
      // The strong reference count has hit zero, so the managed object can be disposed of.
      dispose();
      decrement_weak_count();
    }
  }
  
  // Increment the weak reference count.
  void increment_weak_count() noexcept {
    weak_count.fetch_add(1, std::memory_order_relaxed);
  }

  // Release weak references to the object. If this causes the weak reference count
  // to hit zero, the control block is ready to be destroyed.  We delay the destroy
  // using hazard pointers in case there are in in-flight increments.
  void decrement_weak_count() noexcept {
    if (weak_count.fetch_sub(1, std::memory_order_release) == 1) {
      auto& hazptr = get_hazard_list<control_block_base>();
      hazptr.retire(this);
    }
  }

  control_block_base*& get_next() noexcept { return next_; }
  void* get_ptr() const noexcept { return const_cast<void*>(ptr); }

  auto get_use_count() const noexcept { return strong_count.load(std::memory_order_relaxed); }
  auto get_weak_count() const noexcept { return weak_count.load(std::memory_order_relaxed); }

 protected:
  void set_ptr(void* ptr_) noexcept { ptr = ptr_; }

 private:
  WaitFreeCounter<ref_cnt_type> strong_count;
  std::atomic<ref_cnt_type> weak_count;

  union {
    control_block_base* next_;     // Used for garbage collection by Hazard pointers
    void* ptr;                     // Pointer to the managed object while it is alive
  };
};


// Diambiguate make_shared and make_shared_for_overwrite
struct for_overwrite_tag {};

// Shared base class for control blocks that store the object directly inside
template<typename T>
struct control_block_inplace_base : public control_block_base {
  
  control_block_inplace_base() : control_block_base(nullptr), empty{} { }

  T* get() const noexcept { return const_cast<T*>(std::addressof(object)); }

  // Store the object inside a union, so we get precise control over its lifetime
  union {
    T object;
    char empty;
  };
};


template<typename T>
struct control_block_inplace final : public control_block_inplace_base<T> {

  explicit control_block_inplace(for_overwrite_tag) {
    ::new(static_cast<void*>(this->get())) T;   // Default initialization when using make_shared_for_overwrite
    this->set_ptr(this->get());
  }

  template<typename... Args>
    requires (!(std::is_same_v<for_overwrite_tag, Args> && ...))
  explicit control_block_inplace(Args&&... args) {
    ::new(static_cast<void*>(this->get())) T(std::forward<Args>(args)...);
    this->set_ptr(this->get());
  }
  
  ~control_block_inplace() noexcept = default;
  
  void dispose() noexcept override {
    this->get()->~T();
  }
  
  void destroy() noexcept override {
    delete this;
  }
};

template<typename T, typename Allocator>
struct control_block_inplace_allocator final : public control_block_inplace_base<T> {
  
  using cb_allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<control_block_inplace_allocator>;
  using object_allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<std::remove_cv_t<T>>;

  control_block_inplace_allocator(Allocator alloc_, for_overwrite_tag) {
    ::new(static_cast<void*>(this->get())) T;   // Default initialization when using make_shared_for_overwrite
    this->set_ptr(this->get());                 // Unfortunately not possible via the allocator since the C++
                                                // standard forgot about this case, apparently.
  }

  template<typename... Args>
    requires (!(std::is_same_v<for_overwrite_tag, Args> && ...))
  explicit control_block_inplace_allocator(Allocator alloc_, Args&&... args) : alloc(alloc_) {
    std::allocator_traits<object_allocator_t>::construct(alloc, this->get(), std::forward<Args>(args)...);
    this->set_ptr(this->get());
  }
  
  ~control_block_inplace_allocator() noexcept = default;
  
  void dispose() noexcept override {
    std::allocator_traits<object_allocator_t>::destroy(alloc, this->get());
  }
  
  void destroy() noexcept override {
    cb_allocator_t a{alloc};
    this->~control_block_inplace_allocator();
    std::allocator_traits<cb_allocator_t>::deallocate(a, this, 1);
  }
  
  [[no_unique_address]] object_allocator_t alloc;
};


// A control block pointing to a dynamically allocated object without a custom allocator or custom deleter
template<typename T>
struct control_block_with_ptr : public control_block_base {
  
  using base = control_block_base;
  
  explicit control_block_with_ptr(T* ptr_) : base(ptr_) { }
  
  void dispose() noexcept override {
    delete get();
  }
  
  void destroy()  noexcept override {
    delete this;
  }
  
  T* get() const noexcept {
    return static_cast<T*>(this->get_ptr());
  }
};

// A control block pointering to a dynamically allocated object with a custom deleter
template<typename T, typename Deleter>
struct control_block_with_deleter : public control_block_with_ptr<T> {
  
  using base = control_block_with_ptr<T>;
  
  control_block_with_deleter(T* ptr_, Deleter deleter_) : base(ptr_), deleter(std::move(deleter_)) { }
  
  ~control_block_with_deleter() noexcept override = default;
  
  // Get a pointer to the custom deleter if it is of the request type indicated by the argument
  [[nodiscard]] void* get_deleter(const std::type_info& type) const noexcept override {
    if (type == typeid(Deleter)) {
      return const_cast<Deleter*>(std::addressof(deleter));
    }
    else {
      return nullptr;
    }
  }
  
  // Dispose of the managed object using the provided custom deleter
  void dispose() noexcept override {
    deleter(this->ptr);
  }
  
  [[no_unique_address]] Deleter deleter;
};


// A control block pointering to a dynamically allocated object with a custom deleter and custom allocator
template<typename T, typename Deleter, typename Allocator>
struct control_block_with_allocator final : public control_block_with_deleter<T, Deleter> {

  using base = control_block_with_deleter<T, Deleter>;
  using allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<control_block_with_allocator>;

  control_block_with_allocator(T* ptr_, Deleter deleter_, const Allocator& alloc_) :
    base(ptr_, std::move(deleter_)), alloc(alloc_) { }
  
  ~control_block_with_allocator() noexcept override = default;
  
  // Deallocate the control block using the provided custom allocator
  void destroy() noexcept override {
    allocator_t a{alloc};                     // We must copy the allocator otherwise it gets destroyed
    this->~control_block_with_allocator();    // on the next line, then we can't use it on the final line
    std::allocator_traits<allocator_t>::deallocate(a, this, 1);
  }
  
  [[no_unique_address]] allocator_t alloc;
};


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
  
  smart_ptr_base(const smart_ptr_base&) = delete;
  smart_ptr_base& operator=(const smart_ptr_base&) = delete;
  
  [[nodiscard]] element_type* get() const noexcept {
    return ptr;
  }
  
 protected:

  constexpr smart_ptr_base() noexcept = default;
  
  smart_ptr_base(element_type* ptr_, control_block_base* control_block_) noexcept : ptr(ptr_), control_block(control_block_)  { }
  
  ~smart_ptr_base() = default;
  
  void swap_ptrs(smart_ptr_base& other) noexcept {
    std::swap(ptr, other.ptr);
    std::swap(control_block, other.control_block);
  }
  
  void increment_strong() const noexcept {
    if (control_block) {
      control_block->increment_strong_count();
    }
  }
  
  [[nodiscard]] bool increment_if_nonzero() const noexcept {
    if (control_block) {
      return control_block->increment_strong_count_if_nonzero();
    }
    return false;
  }
  
  void decrement_strong() noexcept {
    if (control_block) {
      control_block->decrement_strong_count();
    }
  }
  
  void increment_weak() const noexcept {
    if (control_block) {
      control_block->increment_weak_count();
    }
  }
  
  void decrement_weak() noexcept {
    if (control_block) {
      control_block->decrement_weak_count();
    }
  }
  
  void set_ptr_and_control_block(element_type* ptr_, control_block_base* control_block_) {
    ptr = ptr_;
    control_block = control_block_;
  }

  template<typename Deleter, typename TT>
  friend Deleter* get_deleter(const shared_ptr<TT>& sp) noexcept;

  element_type* ptr{nullptr};
  control_block_base* control_block{nullptr};
};

}  // namespace details

template<typename T>
class shared_ptr : public details::smart_ptr_base<T> {
  
  using base = details::smart_ptr_base<T>;  
  
  template<typename U>
  friend class atomic_shared_ptr;
  
  template<typename T0>
  friend class shared_ptr;
  
  template<typename T0>
  friend class weak_ptr;
  
  // Private constructor used by atomic_shared_ptr::load
  shared_ptr(T* ptr_, details::control_block_base* control_block_) : base(ptr_, control_block_) {}
  
 public:
  using typename base::element_type;
  using weak_type = weak_ptr<T>;
  
  // Decrement the reference count on destruction.  Resource cleanup is all 
  // handled internally by the control block (including deleting itself!)
  ~shared_ptr() noexcept {
    this->decrement_strong();
  }
  
  // ==========================================================================================
  //                              INITIALIZING AND NULL CONSTRUCTORS
  // ==========================================================================================
  
  constexpr shared_ptr() noexcept = default;
  
  constexpr shared_ptr(std::nullptr_t) noexcept {}
  
  template<typename U>
    requires std::convertible_to<U*, T*>
  explicit shared_ptr(U* p) {
    std::unique_ptr<U> up(p);     // Hold inside a unique_ptr so that p is deleted if the allocation throws
    auto control_block = new details::control_block_with_ptr<U>(p);
    this->set_ptr_and_control_block(up.release(), control_block);
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(U* p, Deleter deleter) {
    std::unique_ptr<U, Deleter> up(p, deleter);
    auto control_block = new details::control_block_with_deleter<U, Deleter>(p, std::move(deleter));
    this->set_ptr_and_control_block(up.release(), control_block);
  }
  
  template<typename U, typename Deleter, typename Allocator>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(U* p, Deleter deleter, Allocator alloc) {
    using cb_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<details::control_block_with_allocator<U, Deleter, Allocator>>;
    
    std::unique_ptr<U, Deleter> up(p, deleter);
    cb_alloc_t a{alloc};
    auto control_block = std::allocator_traits<cb_alloc_t>::allocate(a, 1);
    std::allocator_traits<cb_alloc_t>::construct(a, control_block, p, std::move(deleter), a);
    this->set_ptr_and_control_block(up.release(), control_block);
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(std::nullptr_t, Deleter deleter) {
    std::unique_ptr<U, Deleter> up(nullptr, deleter);
    auto control_block = new details::control_block_with_deleter<U, Deleter>(nullptr, std::move(deleter));
    this->set_ptr_and_control_block(nullptr, control_block);
  }
  
  template<typename U, typename Deleter, typename Allocator>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(std::nullptr_t, Deleter deleter, Allocator alloc) {
    using cb_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<details::control_block_with_allocator<U, Deleter, Allocator>>;
    
    std::unique_ptr<U, Deleter> up(nullptr, deleter);
    cb_alloc_t a{alloc};
    auto control_block = std::allocator_traits<cb_alloc_t>::allocate(a, 1);
    std::allocator_traits<cb_alloc_t>::construct(a, control_block, nullptr, std::move(deleter), a);
    this->set_ptr_and_control_block(up.release(), control_block);
  }
  
  // ==========================================================================================
  //                                  ALIASING CONSTRUCTORS
  // ==========================================================================================
  
  template<typename T2>
  shared_ptr(const shared_ptr<T2>& other, element_type* p) noexcept {
    other.increment_strong();
    this->set_ptr_and_control_block(p, other.control_block);
  }
  
  
  template<typename T2>
  shared_ptr(shared_ptr<T2>&& other, element_type* p) noexcept {
    this->set_ptr_and_control_block(p, other.control_block);
    other.p = nullptr;
    other.control_block = nullptr;
  }
  
  // ==========================================================================================
  //                                  COPY CONSTRUCTORS
  // ==========================================================================================
  
  shared_ptr(const shared_ptr& other) noexcept : base(other.ptr, other.control_block) {
    other.increment_strong();
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  shared_ptr(const shared_ptr<T2>& other) noexcept {
    other.increment_strong();
    this->set_ptr_and_control_block(other.ptr, other.control_block);
  }
  
  // ==========================================================================================
  //                                  MOVE CONSTRUCTORS
  // ==========================================================================================
  
  shared_ptr(shared_ptr&& other) noexcept {
    this->set_ptr_and_control_block(other.ptr, other.control_block);
    other.ptr = nullptr;
    other.control_block = nullptr;
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  shared_ptr(shared_ptr<T2>&& other) noexcept {
    this->set_ptr_and_control_block(other.ptr, other.control_block);
    other.ptr = nullptr;
    other.control_block = nullptr;
  }
  
  // ==========================================================================================
  //                                  CONVERTING CONSTRUCTORS
  // ==========================================================================================
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  shared_ptr(const weak_ptr<T2>& other) {
    if (other.increment_if_nonzero()) {
      this->set_ptr_and_control_block(other.ptr, other.control_block);
    }
    else {
      throw std::bad_weak_ptr();
    }
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::convertible_to<typename std::unique_ptr<U, Deleter>::pointer, T*>
  shared_ptr(std::unique_ptr<U, Deleter>&& other) {
    using ptr_type = typename std::unique_ptr<U, Deleter>::pointer;
    using deleter_type = std::conditional_t<std::is_reference_v<Deleter>, decltype(std::ref(other.get_deleter())), Deleter>;
    
    if (other) {
      auto control_block = new details::control_block_with_deleter<ptr_type, deleter_type>
        (other.get(), std::forward<decltype(other.get_deleter())>(other.get_deleter()));
      this->set_ptr_and_control_block(other.release(), control_block);
    }
  }
  
  // ==========================================================================================
  //                                  ASSIGNMENT OPERATORS
  // ==========================================================================================  
  
  shared_ptr& operator=(const shared_ptr& other) noexcept {
    shared_ptr(other).swap(*this);
    return *this;
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  shared_ptr& operator=(const shared_ptr<T2>& other) noexcept {
    shared_ptr(other).swap(*this);
    return *this;
  }
  
  shared_ptr& operator=(shared_ptr&& other) noexcept {
    shared_ptr(std::move(other)).swap(*this);
    return *this;
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  shared_ptr& operator=(shared_ptr<T2>&& other) noexcept {
    shared_ptr(std::move(other)).swap(*this);
    return *this;
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::convertible_to<typename std::unique_ptr<U, Deleter>::pointer, T*>
  shared_ptr& operator=(std::unique_ptr<U, Deleter>&& other) {
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

  template<typename Deleter>
    requires std::copy_constructible<Deleter> && std::invocable<Deleter&, std::nullptr_t>
  void reset(std::nullptr_t, Deleter deleter) {
    shared_ptr(nullptr, deleter).swap(*this);
  }

  template<typename Deleter, typename Allocator>
    requires std::copy_constructible<Deleter> && std::invocable<Deleter&, std::nullptr_t>
  void reset(std::nullptr_t, Deleter deleter, Allocator alloc) {
    shared_ptr(nullptr, deleter, alloc).swap(*this);
  }

  template<typename U>
    requires std::convertible_to<U*, T*>
  void reset(U* p) {
    shared_ptr(p).swap(*this);
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  void reset(U* p, Deleter deleter) {
    shared_ptr(p, deleter).swap(*this);
  }
  
  template<typename U, typename Deleter, typename Allocator>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  void reset(U* p, Deleter deleter, Allocator alloc) {
    shared_ptr(p, deleter, alloc).swap(*this);
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
  
  template<typename T0, typename... Args>
    requires std::constructible_from<T0, Args...>
  friend shared_ptr<T0> make_shared_for_overwrite();
  
  template<typename T0, typename Allocator, typename... Args>
    requires std::constructible_from<T0, Args...>
  friend shared_ptr<T0> allocate_shared(const Allocator& allocator, Args&&... args);
  
  template<typename T0, typename Allocator, typename... Args>
    requires std::constructible_from<T0, Args...>
  friend shared_ptr<T0> allocate_shared_for_overwrite(const Allocator& allocator);
  
 private:
 
  std::pair<T*, details::control_block_base*> release_internals() noexcept {
    return std::make_pair(std::exchange(this->ptr, nullptr), std::exchange(this->control_block, nullptr));
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
  const auto control_block = new details::control_block_inplace<T>(std::forward<Args>(args)...);
  shared_ptr<T> result(control_block->get(), control_block);
  return result;
}

template<typename T, typename... Args>
[[nodiscard]] shared_ptr<T> make_shared_for_overwrite() {
  const auto control_block = new details::control_block_inplace<T>(details::for_overwrite_tag{});
  shared_ptr<T> result;
  result.set_ptr_and_control_block(control_block.get(), control_block);
  return result;
}

template<typename T, typename Allocator, typename... Args>
[[nodiscard]] shared_ptr<T> allocate_shared(const Allocator& allocator, Args&&... args) {
  using control_block_type = details::control_block_inplace_allocator<std::remove_cv_t<T>, Allocator>;
  using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<control_block_type>;
  
  allocator_type a{allocator};
  const auto control_block = std::allocator_traits<allocator_type>::allocate(a, 1);
  std::allocator_traits<allocator_type>::construct(a, control_block, a, std::forward<Args>(args)...);
  shared_ptr<T> result;
  result.set_ptr_and_control_block(control_block.get(), control_block);
  return result;
}

template<typename T, typename Allocator, typename... Args>
[[nodiscard]] shared_ptr<T> allocate_shared_for_overwrite(const Allocator& allocator) {
  using control_block_type = details::control_block_inplace_allocator<std::remove_cv_t<T>, Allocator>;
  using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<control_block_type>;
  
  allocator_type a{allocator};
  const auto control_block = std::allocator_traits<allocator_type>::allocate(a, 1);
  std::allocator_traits<allocator_type>::construct(a, control_block, a, details::for_overwrite_tag{});
  shared_ptr<T> result;
  result.set_ptr_and_control_block(control_block.get(), control_block);
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

template<typename T>
class weak_ptr : public details::smart_ptr_base<T> {
  // TODO
};


}  // namespace parlay

