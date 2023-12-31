// A custom modernized C++20 shared_ptr and weak_ptr implementation used by the atomic_shared_ptr.
//
// It tries to closely match the standard library std::shared_ptr as much as possible. Most  of the
// code roughly follows the same implementation strategies as libstdc++, libc++, and Microsoft STL.
//
// The main difference is using Hazard Pointer deferred reclaimation on the control block to
// allow atomic_shared_ptr to be lock free and not require a split reference count.
//
// No support for std::shared_ptr<T[]>, i.e., shared pointers of arrays. Everything else should
// be supported, including custom deleters, allocators, weak_ptrs, enable_shared_from_this, etc.
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

template<typename T>
class weak_ptr;

template<typename T>
class enable_shared_from_this;

template<typename Deleter, typename T>
Deleter* get_deleter(const shared_ptr<T>&) noexcept;

namespace details {

// Very useful explanation from Raymond Chen's blog:
// https://devblogs.microsoft.com/oldnewthing/20230816-00/?p=108608
template<typename T>
concept SupportsESFT = requires() {
  typename T::esft_detector;                                                    // Class should derive from ESFT
  requires std::same_as<typename T::esft_detector, enable_shared_from_this<T>>;
  requires std::convertible_to<T*, enable_shared_from_this<T>*>;                // Inheritance is unambiguous
};

using ref_cnt_type = uint32_t;


// Base class of all control blocks used by smart pointers.  This base class is agnostic
// to the type of the managed object, so all type-specific operations are implemented
// by virtual functions in the derived classes.
struct control_block_base {
  
  template<typename T>
  friend class atomic_shared_ptr;

  explicit control_block_base() noexcept : strong_count(1), weak_count(1) { }

  control_block_base(const control_block_base &) = delete;
  control_block_base& operator=(const control_block_base&) = delete;

  virtual ~control_block_base() = default;
  
  // Destroy the managed object.  Called when the strong count hits zero
  virtual void dispose() noexcept = 0;
  
  // Destroy the control block.  dispose() must have been called prior to
  // calling destroy.  Called when the weak count hits zero.
  virtual void destroy() noexcept = 0;

  // Delay the destroy using hazard pointers in case there are in in-flight increments.
  void retire() noexcept {
    // Defer destruction of the control block using hazard pointers
    get_hazard_list<control_block_base>().retire(this);
  }

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
  // to hit zero, the control block is ready to be destroyed.
  void decrement_weak_count() noexcept {
    if (weak_count.fetch_sub(1, std::memory_order_release) == 1) {
      retire();
    }
  }

  [[nodiscard]] virtual control_block_base* get_next() const noexcept = 0;
  virtual void set_next(control_block_base* next) noexcept = 0;

  [[nodiscard]] virtual void* get_ptr() const noexcept = 0;

  auto get_use_count() const noexcept { return strong_count.load(std::memory_order_relaxed); }
  auto get_weak_count() const noexcept { return weak_count.load(std::memory_order_relaxed); }

 private:
  WaitFreeCounter<ref_cnt_type> strong_count;
  std::atomic<ref_cnt_type> weak_count;
};


// Diambiguate make_shared and make_shared_for_overwrite
struct for_overwrite_tag {};

// Shared base class for control blocks that store the object directly inside
template<typename T>
struct control_block_inplace_base : public control_block_base {
  
  control_block_inplace_base() : control_block_base(), empty{} { }

  T* get() const noexcept { return const_cast<T*>(std::addressof(object)); }

  void* get_ptr() const noexcept override {
    return static_cast<void*>(get());
  }

  // Expose intrusive pointers used by Hazard Pointers
  [[nodiscard]] control_block_base* get_next() const noexcept override { return next_; }
  void set_next(control_block_base* next) noexcept override { next_ = next; }

  ~control_block_inplace_base() override { }


  union {
    std::monostate empty{};
    T object;                       // Since the object is inside a union, we get precise control over its lifetime
    control_block_base* next_;      // Intrusive ptr used for garbage collection by Hazard Pointers
  };
};


template<typename T>
struct control_block_inplace final : public control_block_inplace_base<T> {

  // TODO: Don't hardcode an allocator override here.  Should just
  // use allocate_shared and pass in an appropriate allocator.
  static void* operator new(std::size_t sz) {
    assert(sz == sizeof(control_block_inplace));
    return parlay::type_allocator<control_block_inplace>::alloc();
  }

  static void operator delete(void* ptr) {
    parlay::type_allocator<control_block_inplace>::free(static_cast<control_block_inplace*>(ptr));
  }

  explicit control_block_inplace(for_overwrite_tag) {
    ::new(static_cast<void*>(this->get())) T;   // Default initialization when using make_shared_for_overwrite
  }

  template<typename... Args>
    requires (!(std::is_same_v<for_overwrite_tag, Args> || ...))
  explicit control_block_inplace(Args&&... args) {
    ::new(static_cast<void*>(this->get())) T(std::forward<Args>(args)...);
  }
  
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

  control_block_inplace_allocator(Allocator, for_overwrite_tag) {
    ::new(static_cast<void*>(this->get())) T;   // Default initialization when using make_shared_for_overwrite
                                                // Unfortunately not possible via the allocator since the C++
                                                // standard forgot about this case, apparently.
  }

  template<typename... Args>
    requires (!(std::is_same_v<for_overwrite_tag, Args> && ...))
  explicit control_block_inplace_allocator(Allocator alloc_, Args&&... args) : alloc(alloc_) {
    std::allocator_traits<object_allocator_t>::construct(alloc, this->get(), std::forward<Args>(args)...);
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
  
  explicit control_block_with_ptr(T* ptr_) : ptr(ptr_) { }
  
  void dispose() noexcept override {
    delete get();
  }
  
  void destroy()  noexcept override {
    delete this;
  }

  void* get_ptr() const noexcept override {
    return static_cast<void*>(get());
  }

  T* get() const noexcept {
    return const_cast<T*>(ptr);
  }

  // Expose intrusive pointers used by Hazard Pointers
  [[nodiscard]] control_block_base* get_next() const noexcept override { return next_; }
  void set_next(control_block_base* next) noexcept override { next_ = next; }

  union {
    control_block_base* next_;     // Intrusive ptr used for garbage collection by Hazard pointers
    T* ptr;                        // Pointer to the managed object while it is alive
  };
};

// A control block pointing to a dynamically allocated object with a custom deleter
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


// A control block pointing to a dynamically allocated object with a custom deleter and custom allocator
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

  smart_ptr_base& operator=(const smart_ptr_base&) = delete;
  
  [[nodiscard]] element_type* get() const noexcept {
    return ptr;
  }
  
 protected:

  constexpr smart_ptr_base() noexcept = default;
  
  smart_ptr_base(element_type* ptr_, control_block_base* control_block_) noexcept
      : ptr(ptr_), control_block(control_block_)  {
    assert(control_block != nullptr || ptr == nullptr);   // Can't have non-null ptr and null control_block
  }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit smart_ptr_base(const smart_ptr_base<T2>& other) noexcept
      : ptr(other.ptr), control_block(other.control_block) {
    assert(control_block != nullptr || ptr == nullptr);   // Can't have non-null ptr and null control_block
  }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit smart_ptr_base(smart_ptr_base<T2>&& other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)), control_block(std::exchange(other.control_block, nullptr)) {
    assert(control_block != nullptr || ptr == nullptr);   // Can't have non-null ptr and null control_block
  }
  
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
    return control_block && control_block->increment_strong_count_if_nonzero();
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

  template<typename Deleter, typename TT>
  friend Deleter* ::parlay::get_deleter(const shared_ptr<TT>&) noexcept;

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
  
  // Private constructor used by atomic_shared_ptr::load and weak_ptr::lock
  shared_ptr(T* ptr_, details::control_block_base* control_block_) : base(ptr_, control_block_) { }
  
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
  
  constexpr explicit(false) shared_ptr(std::nullptr_t) noexcept {}      // NOLINT(google-explicit-constructor)
  
  template<typename U>
    requires std::convertible_to<U*, T*>
  explicit shared_ptr(U* p) {
    std::unique_ptr<U> up(p);     // Hold inside a unique_ptr so that p is deleted if the allocation throws
    auto control_block = new details::control_block_with_ptr<U>(p);
    this->set_ptrs_and_esft(up.release(), control_block);
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(U* p, Deleter deleter) {
    std::unique_ptr<U, Deleter> up(p, deleter);
    auto control_block = new details::control_block_with_deleter<U, Deleter>(p, std::move(deleter));
    this->set_ptrs_and_esft(up.release(), control_block);
  }
  
  template<typename U, typename Deleter, typename Allocator>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(U* p, Deleter deleter, Allocator alloc) {
    using cb_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<details::control_block_with_allocator<U, Deleter, Allocator>>;
    
    std::unique_ptr<U, Deleter> up(p, deleter);
    cb_alloc_t a{alloc};
    auto control_block = std::allocator_traits<cb_alloc_t>::allocate(a, 1);
    std::allocator_traits<cb_alloc_t>::construct(a, control_block, p, std::move(deleter), a);
    this->set_ptrs_and_esft(up.release(), control_block);
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(std::nullptr_t, Deleter deleter) {
    std::unique_ptr<U, Deleter> up(nullptr, deleter);
    auto control_block = new details::control_block_with_deleter<U, Deleter>(nullptr, std::move(deleter));
    this->set_ptrs_and_esft(nullptr, control_block);
  }
  
  template<typename U, typename Deleter, typename Allocator>
    requires std::convertible_to<U*, T*> && std::copy_constructible<Deleter> && std::invocable<Deleter&, U*>
  shared_ptr(std::nullptr_t, Deleter deleter, Allocator alloc) {
    using cb_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<details::control_block_with_allocator<U, Deleter, Allocator>>;
    
    std::unique_ptr<U, Deleter> up(nullptr, deleter);
    cb_alloc_t a{alloc};
    auto control_block = std::allocator_traits<cb_alloc_t>::allocate(a, 1);
    std::allocator_traits<cb_alloc_t>::construct(a, control_block, nullptr, std::move(deleter), a);
    this->set_ptrs_and_esft(up.release(), control_block);
  }
  
  // ==========================================================================================
  //                                  ALIASING CONSTRUCTORS
  // ==========================================================================================
  
  template<typename T2>
  shared_ptr(const shared_ptr<T2>& other, element_type* p) noexcept : base(p, other.control_block) {
    this->increment_strong();
  }

  template<typename T2>
  shared_ptr(shared_ptr<T2>&& other, element_type* p) noexcept : base(p, other.control_block) {
    other.ptr = nullptr;
    other.control_block = nullptr;
  }
  
  // ==========================================================================================
  //                                  COPY CONSTRUCTORS
  // ==========================================================================================
  
  shared_ptr(const shared_ptr& other) noexcept : base(other) {
    this->increment_strong();
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit(false) shared_ptr(const shared_ptr<T2>& other) noexcept {        // NOLINT(google-explicit-constructor)
    other.increment_strong();
    this->set_ptrs_and_esft(other.ptr, other.control_block);
  }
  
  // ==========================================================================================
  //                                  MOVE CONSTRUCTORS
  // ==========================================================================================
  
  shared_ptr(shared_ptr&& other) noexcept {
    this->set_ptrs_and_esft(other.ptr, other.control_block);
    other.ptr = nullptr;
    other.control_block = nullptr;
  }
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit(false) shared_ptr(shared_ptr<T2>&& other) noexcept {         // NOLINT(google-explicit-constructor)
    this->set_ptrs_and_esft(other.ptr, other.control_block);
    other.ptr = nullptr;
    other.control_block = nullptr;
  }
  
  // ==========================================================================================
  //                                  CONVERTING CONSTRUCTORS
  // ==========================================================================================
  
  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit(false) shared_ptr(const weak_ptr<T2>& other) {               // NOLINT(google-explicit-constructor)
    if (other.increment_if_nonzero()) {
      this->set_ptrs_and_esft(other.ptr, other.control_block);
    }
    else {
      throw std::bad_weak_ptr();
    }
  }
  
  template<typename U, typename Deleter>
    requires std::convertible_to<U*, T*> && std::convertible_to<typename std::unique_ptr<U, Deleter>::pointer, T*>
  explicit(false) shared_ptr(std::unique_ptr<U, Deleter>&& other) {     // NOLINT(google-explicit-constructor)
    using ptr_type = typename std::unique_ptr<U, Deleter>::pointer;
    
    if (other) {
      // [https://en.cppreference.com/w/cpp/memory/shared_ptr/shared_ptr]
      // If Deleter is a reference type, it is equivalent to shared_ptr(r.release(), std::ref(r.get_deleter()).
      // Otherwise, it is equivalent to shared_ptr(r.release(), std::move(r.get_deleter()))
      if constexpr (std::is_reference_v<Deleter>) {
        auto control_block = new details::control_block_with_deleter<ptr_type, decltype(std::ref(other.get_deleter()))>
          (other.get(), std::ref(other.get_deleter()));
        this->set_ptrs_and_esft(other.release(), control_block);
      }
      else {
        auto control_block = new details::control_block_with_deleter<ptr_type, Deleter>
          (other.get(), std::move(other.get_deleter()));
        this->set_ptrs_and_esft(other.release(), control_block);
      }
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

  template<typename U>
  void set_ptrs_and_esft(U* ptr_, details::control_block_base* control_block_) {
    static_assert(std::convertible_to<U*, T*>);

    this->ptr = ptr_;
    this->control_block = control_block_;

    if constexpr(details::SupportsESFT<element_type>) {
      if (this->ptr && this->ptr->weak_this.expired()) {
        this->ptr->weak_this = shared_ptr<std::remove_cv_t<U>>(*this, const_cast<std::remove_cv_t<U>*>(this->ptr));
      }
    }
  }

  // Release the ptr and control_block to the caller.  Does not modify the reference count,
  // so the caller is responsible for taking over the reference count owned by this copy
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
  result.set_ptrs_and_esft(control_block.get(), control_block);
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
  result.set_ptrs_and_esft(control_block.get(), control_block);
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
  result.set_ptrs_and_esft(control_block.get(), control_block);
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

  using base = details::smart_ptr_base<T>;

 public:

// ==========================================================================================
//                                       CONSTRUCTORS
// ==========================================================================================

    constexpr weak_ptr() noexcept = default;

    weak_ptr(const weak_ptr& other) noexcept : base(other) { }

    template<typename T2>
      requires std::convertible_to<T2*, T*>
    explicit(false) weak_ptr(const shared_ptr<T2>& other) noexcept          // NOLINT(google-explicit-constructor)
        : base(other) {
      this->increment_weak();
    }

    template<typename T2>
      requires std::convertible_to<T2*, T*> && std::convertible_to<T*, const T2*>
    explicit(false) weak_ptr(const weak_ptr<T2>& other) noexcept            // NOLINT(google-explicit-constructor)
        : base(other) {
      this->increment_weak();
    }

    template<typename T2>
      requires std::convertible_to<T2*, T*>
    explicit(false) weak_ptr(const weak_ptr<T2>& other) noexcept            // NOLINT(google-explicit-constructor)
      : base{} {

      // This case is subtle.  If T2 virtually inherits T, then it might require RTTI to
      // convert from T2* to T*.  If other.ptr is expired, the vtable may have been
      // destroyed, which is very bad.  Furthermore, other.ptr could expire concurrently
      // at any point by another thread, so we can not just check. So, we increment the
      // strong ref count to prevent other from being destroyed while we copy.
      if (other.control_block) {
        this->control_block = other.control_block;
        this->control_block->increment_weak_count();

        if (this->increment_if_nonzero()) {
          this->ptr = other.ptr;    // Now that we own a strong ref, it is safe to copy the ptr
          this->control_block->decrement_strong_count();
        }
      }
    }

    weak_ptr(weak_ptr&& other) noexcept : base(std::move(other)) { }

  template<typename T2>
    requires std::convertible_to<T2*, T*> && std::convertible_to<T*, const T2*>
  explicit(false) weak_ptr(weak_ptr<T2>&& other) noexcept                   // NOLINT(google-explicit-constructor)
    : base(std::move(other)) { }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  explicit(false) weak_ptr(weak_ptr<T2>&& other) noexcept : base{} {        // NOLINT(google-explicit-constructor)
    this->control_block = std::exchange(other.control_block, nullptr);

    // See comment in copy constructor.  Same subtlety applies.
    if (this->increment_if_nonzero()) {
      this->ptr = other.ptr;
      this->control_block->decrement_strong_count();
    }

    other.ptr = nullptr;
  }

  ~weak_ptr() {
    this->decrement_weak();
  }

// ==========================================================================================
//                                       ASSIGNMENT OPERATORS
// ==========================================================================================

  weak_ptr& operator=(const weak_ptr& other) noexcept {
    weak_ptr(other).swap(*this);
    return *this;
  }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  weak_ptr& operator=(const weak_ptr<T2>& other) noexcept {
    weak_ptr(other).swap(*this);
    return *this;
  }

  weak_ptr& operator=(weak_ptr&& other) noexcept {
    weak_ptr(std::move(other)).swap(*this);
    return *this;
  }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  weak_ptr& operator=(weak_ptr<T2>&& other) noexcept {
    weak_ptr(std::move(other)).swap(*this);
    return *this;
  }

  template<typename T2>
    requires std::convertible_to<T2*, T*>
  weak_ptr& operator=(const shared_ptr<T2>& other) noexcept {
    weak_ptr(other).swap(*this);
    return *this;
  }

  void swap(weak_ptr& other) noexcept {
    this->swap_ptrs(other);
  }

  [[nodiscard]] bool expired() const noexcept {
    return this->use_count() == 0;
  }

  [[nodiscard]] shared_ptr<T> lock() const noexcept {
    if (this->increment_if_nonzero()) {
      return shared_ptr<T>{this->ptr, this->control_block};
    }
    return {nullptr};
  }

};


// ==========================================================================================
//                                       shared_from_this
// ==========================================================================================

template<typename T>
class enable_shared_from_this {
protected:
  constexpr enable_shared_from_this() noexcept : weak_this{} {}

  enable_shared_from_this(enable_shared_from_this const&) noexcept : weak_this{} {}

  enable_shared_from_this& operator=(enable_shared_from_this const&) noexcept { return *this; }

  ~enable_shared_from_this() = default;

public:
  using esft_detector = enable_shared_from_this;

  [[nodiscard]] weak_ptr<T> weak_from_this() {
    return weak_this;
  }

  [[nodiscard]] weak_ptr<const T> weak_from_this() const {
    return weak_this;
  }

  [[nodiscard]] shared_ptr<T> shared_from_this() {
    return shared_ptr<T>{weak_this};
  }

  [[nodiscard]] shared_ptr<const T> shared_from_this() const {
    return shared_ptr<const T>{weak_this};
  }

  mutable weak_ptr<T> weak_this;
};

}  // namespace parlay
