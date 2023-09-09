
#pragma once

#include <cstddef>

#include <deque>
#include <new>
#include <type_traits>
#include <unordered_set>

#include <folly/synchronization/AsymmetricThreadFence.h>

#include <folly/container/F14Set.h>

namespace parlay {

// PARLAY_PREFETCH: Prefetch data into cache
#if defined(__GNUC__)
#define PARLAY_PREFETCH(addr, rw, locality) __builtin_prefetch ((addr), (rw), (locality))
#elif defined(_WIN32)
#define PARLAY_PREFETCH(addr, rw, locality)                                                 \
  PreFetchCacheLine(((locality) ? PF_TEMPORAL_LEVEL_1 : PF_NON_TEMPORAL_LEVEL_ALL), (addr))
#else
#define PARLAY_PREFETCH(addr, rw, locality)
#endif


#ifdef __cpp_lib_hardware_interference_size

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"

inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 2 * std::hardware_destructive_interference_size;

#pragma GCC diagnostic pop

#else
inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 128;
#endif


template<typename T>
concept GarbageCollectible = requires(T t, T *tp) {
  { t.get_next() } -> std::convertible_to<T *>;     // The object should expose an intrusive next ptr
  { t.set_next(tp) };
  { t.destroy() };                                 // The object should be destructible on demand
};

template<typename GarbageType> requires GarbageCollectible<GarbageType>
class HazardList;

template<typename GarbageType>
extern inline HazardList<GarbageType> *get_hazard_list();

template<typename GarbageType> requires GarbageCollectible<GarbageType>
class HazardList {

  constexpr static std::size_t cleanup_threshold = 1000;

  using garbage_type = GarbageType;
  using protected_set_type = folly::F14FastSet<garbage_type *>;

  // The retired list is an intrusive linked list of retired blocks. It takes advantage
  // of the available managed object pointer in the control block to store the next pointers.
  // (since, after retirement, it is guaranteed that the object has been freed, and thus
  // the managed object pointer is no longer used.  Furthermore, it does not have to be
  // kept as null since threads never read the pointer unless they own a reference count.)
  //
  struct RetiredList {

    constexpr RetiredList() noexcept = default;

    ~RetiredList() {
      cleanup([](auto &&) { return false; });
    }

    void push(garbage_type *p) noexcept {
      p->set_next(std::exchange(head, p));
      size++;
    }

    [[nodiscard]] std::size_t get_size() const noexcept {
      return size;
    }

    // For each element x currently in the retired list, if is_protected(x) == false,
    // then x->destroy() and remove x from the retired list.  Otherwise, keep x on
    // the retired list for the next cleanup.
    template<typename F>
    void cleanup(F &&is_protected) {

      while (head && !is_protected(head)) {
        garbage_type *old = std::exchange(head, head->get_next());
        old->destroy();
        size--;
      }

      if (head) {
        garbage_type *prev = head;
        garbage_type *current = head->get_next();
        while (current) {
          if (!is_protected(current)) {
            garbage_type *old = std::exchange(current, current->get_next());
            old->destroy();
            prev->set_next(current);
            size--;
          } else {
            prev = std::exchange(current, current->get_next());
          }
        }
      }
    }

  private:
    garbage_type *head = nullptr;
    std::size_t size = 0;
  };

  // Each thread owns a hazard entry slot which contains a single hazard pointer
  // (called protected_pointer) and the thread's local retired list.
  //
  // The slots are linked together to form a linked list so that threads can scan
  // for the set of currently protected pointers.
  //
  struct alignas(CACHE_LINE_ALIGNMENT) HazardSlot {
    explicit HazardSlot(bool in_use_) : in_use(in_use_) {}

    // The *actual* "Hazard Pointer" that protects the object that it points to.
    // Other threads scan for the set of all such pointers before they clean up.
    std::atomic<garbage_type *> protected_ptr{nullptr};

    // Link together all existing slots into a big global linked list
    std::atomic<HazardSlot *> next{nullptr};

    // True if a thread owns this slot, else false.
    std::atomic<bool> in_use;

    // (Intrusive) linked list of retired objects.  Does not allocate memory since it
    // just uses the next pointer from inside the retired block.
    RetiredList retired{};

    // Set of protected objects used by cleanup().  Re-used between cleanups so that
    // we don't have to allocate new memory unless the table gets full, which would
    // only happen if the user spawns substantially more threads than were active
    // during the previous call to cleanup().
    protected_set_type protected_set{2 * std::thread::hardware_concurrency()};
  };

  // Pre-populate the slot list with P slots, one for each hardware thread
  HazardList() : global_list_head(new HazardSlot{false}) {
    auto current = global_list_head;
    for (unsigned i = 1; i < std::thread::hardware_concurrency(); i++) {
      current->next = new HazardSlot{false};
      current = current->next;
    }
  }

  // Find an available hazard slot, or allocate a new one if none available.
  HazardSlot *get_slot() {
    auto current = global_list_head;
    while (true) {
      if (!current->in_use.load() && !current->in_use.exchange(true)) {
        return current;
      }
      if (current->next.load() == nullptr) {
        auto my_slot = new HazardSlot{true};
        HazardSlot *next = nullptr;
        while (!current->next.compare_exchange_weak(next, my_slot)) {
          current = next;
          next = nullptr;
        }
        return my_slot;
      } else {
        current = current->next.load();
      }
    }
  }

  // Give a slot back to the world so another thread can re-use it
  void relinquish_slot(HazardSlot *slot) {
    slot->in_use.store(false);
  }

  // A HazardSlotOwner owns exactly one HazardSlot entry in the global linked list
  // of HazardSlots.  On creation, it acquires a free slot from the list, or appends
  // a new slot if all of them are in use.  On destruction, it makes the slot available
  // for another thread to pick up.
  struct HazardSlotOwner {
    explicit HazardSlotOwner(HazardList<GarbageType> *list_) : list(list_), my_slot(list->get_slot()) {}

    ~HazardSlotOwner() {
      list->relinquish_slot(my_slot);
    }

  private:
    HazardList<GarbageType> *const list;
  public:
    HazardSlot *const my_slot;
  };

public:

  // Leak on purpose since we don't want static destruction order to ruin our day
  ~HazardList() = delete;

  // Apply the function f to all currently announced hazard pointers
  template<typename F>
  void scan_hazard_pointers(F &&f) noexcept(std::is_nothrow_invocable_v<F &, garbage_type *>) {
    auto current = global_list_head;
    while (current) {
      auto p = current->protected_ptr.load();
      if (p) {
        f(p);
      }
      current = current->next.load();
    }
  }

  // Protect the object pointed to by the pointer currently stored at src.
  //
  // The second argument allows the protected pointer to be deduced from
  // the value stored at src, for example, if src stores a pair containing
  // the pointer to protect and some other value. In this case, the value of
  // f(ptr) is protected instead, but the full value *ptr is still returned.
  template<template<typename> typename Atomic, typename U, typename F>
  U protect(const Atomic<U> &src, F &&f) {
    static_assert(std::is_convertible_v<std::invoke_result_t<F, U>, garbage_type *>);
    auto &slot = local_slot.my_slot->protected_ptr;

    U result = src.load(std::memory_order_relaxed);

    while (true) {
      PARLAY_PREFETCH(f(result), 0, 0);
      slot.store(f(result), std::memory_order_relaxed);
      folly::asymmetric_thread_fence_light(std::memory_order_seq_cst);    /*  Fast-side fence  */
      U current_value = src.load(std::memory_order_acquire);
      if (current_value == result) [[likely]] {
        return result;
      } else {
        result = std::move(current_value);
      }
    }
  }

  // Protect the object pointed to by the pointer currently stored at src.
  template<template<typename> typename Atomic, typename U>
  U protect(const Atomic<U> &src) {
    return protect(src, [](auto &&x) { return std::forward<decltype(x)>(x); });
  }

  // Unprotect the currently protected object
  void release() {
    local_slot.my_slot->protected_ptr.store(nullptr, std::memory_order_release);
  }

  // Retire the given object
  //
  // The object managed by p must have reference count zero.
  void retire(garbage_type *p) noexcept {
    RetiredList &retired_list = local_slot.my_slot->retired;
    retired_list.push(p);

    if (retired_list.get_size() >= cleanup_threshold) [[unlikely]] {
      cleanup(retired_list, local_slot.my_slot->protected_set);
    }
  }

  FOLLY_NOINLINE void cleanup(RetiredList &retired_list, protected_set_type &protected_set) {
    folly::asymmetric_thread_fence_heavy(std::memory_order_seq_cst);

    scan_hazard_pointers([&](garbage_type *p) {
      protected_set.insert(p);
    });
    retired_list.cleanup([&](garbage_type *p) {
      return protected_set.count(p) > 0;
    });

    protected_set.clear();  // Does not free memory, only clears contents
  }

private:
  template<typename U>
  friend HazardList<U> *get_hazard_list();

  HazardSlot *const global_list_head;
  static inline const thread_local HazardSlotOwner local_slot{get_hazard_list<garbage_type>()};
};


// Global singleton containing the list of hazard pointers.  We leak it on
// purpose to avoid falling victim to the woes of static destruction order
//
// (i.e., a detached thread might grab a HazardSlot entry and not relinquish
// it until static destruction, at which point this global static would have
// already been destroyed.)
template<typename GarbageType>
HazardList<GarbageType> *get_hazard_list() {
  static auto *list = new HazardList<GarbageType>{};
  return list;
}

}  // namespace parlay
