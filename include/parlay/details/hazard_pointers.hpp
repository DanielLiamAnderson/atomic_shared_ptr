
#pragma once

#include <cassert>
#include <cstddef>

#include <atomic>
#include <deque>
#include <memory>
#include <new>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <folly/container/F14Set.h>

#include <folly/synchronization/AsymmetricThreadFence.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "immortal.hpp"

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

enum class ReclamationMethod {
  amortized_reclamation,                // Reclamation happens in bulk in the retiring thread
  background_thread_reclamation,        // Reclamation happens asynchronously in a background thread
};

template<typename T>
concept GarbageCollectible = requires(T* t, T* tp) {
  { t->get_next() } -> std::convertible_to<T*>;     // The object should expose an intrusive next ptr
  { t->set_next(tp) };
  { t->destroy() };                                 // The object should be destructible on demand
};

template<typename GarbageType> requires GarbageCollectible<GarbageType>
class HazardPointers;

template<typename GarbageType>
extern inline HazardPointers<GarbageType>& get_hazard_list();

// A simple and efficient implementation of Hazard Pointer deferred reclamation
//
// Each live thread owns *exactly one* Hazard Pointer, which is sufficient for most
// (but not all) algorithms that use them. In particular, it is sufficient for lock-
// free atomic shared ptrs. This makes it much simpler and slightly more efficient
// than a general-purpose Hazard Pointer implementation, like the one in Folly, which
// supports each thread having an arbitrary number of Hazard Pointers.
//
// Each thread keeps a local retired list of objects that are pending deletion.
// This means that a stalled thread can delay the destruction of its retired objects
// indefinitely, however, since each thread is only allowed to protect a single object
// at a time, it is guaranteed that there are at most O(P^2) total unreclaimed objects
// at any given point, so the memory usage is theoretically bounded. This is assuming
// cleanup is performed by the retiring thread.  If using background thread reclamation,
// technically there is no guarantee of memory being freed in a timely manner since the
// background thread could stall and not actually free anything.
template<typename GarbageType> requires GarbageCollectible<GarbageType>
class HazardPointers {

  // After this many retires, a thread will attempt to clean up the contents of
  // its local retired list, deleting any retired objects that are not protected.
  constexpr static std::size_t cleanup_threshold = 1000;

  using garbage_type = GarbageType;
  using protected_set_type = folly::F14FastSet<garbage_type*>;

  // The retired list is an intrusive linked list of retired blocks. It takes advantage
  // of the available managed object pointer in the control block to store the next pointers.
  // (since, after retirement, it is guaranteed that the object has been freed, and thus
  // the managed object pointer is no longer used.  Furthermore, it does not have to be
  // kept as null since threads never read the pointer unless they own a reference count.)
  //
  struct RetiredList {

    constexpr RetiredList() noexcept = default;

    explicit RetiredList(garbage_type* head_) : head(head_) { }

    RetiredList& operator=(garbage_type* head_) {
      assert(head == nullptr);
      head = head_;
    }

    RetiredList(const RetiredList&) = delete;

    RetiredList(RetiredList&& other) noexcept : head(std::exchange(other.head, nullptr)) { }

    ~RetiredList() {
      cleanup([](auto&&) { return false; });
    }

    void push(garbage_type* p) noexcept {
      p->set_next(std::exchange(head, p));
      if (p->get_next() == nullptr) [[unlikely]] tail = p;
    }

    // For each element x currently in the retired list, if is_protected(x) == false,
    // then x->destroy() and remove x from the retired list.  Otherwise, keep x on
    // the retired list for the next cleanup.
    template<typename F>
    void cleanup(F&& is_protected) {

      while (head && !is_protected(head)) {
        garbage_type* old = std::exchange(head, head->get_next());
        old->destroy();
      }

      if (head) {
        garbage_type* prev = head;
        garbage_type* current = head->get_next();
        while (current) {
          if (!is_protected(current)) {
            garbage_type* old = std::exchange(current, current->get_next());
            old->destroy();
            prev->set_next(current);
          } else {
            prev = std::exchange(current, current->get_next());
          }
        }
      }
    }

    // For each element x currently in the retired list, if is_protected(x) == false,
    // then x->destroy() and remove x from the retired list.  Otherwise, move x onto
    // the leftovers list.  Leaves the current list empty.
    template<typename F>
    void cleanup_and_move(RetiredList& leftovers, F&& is_protected) {

      while (head) {
        garbage_type* current = std::exchange(head, head->get_next());
        if (is_protected(current)) {
          leftovers.push(current);
        }
        else {
          current->destroy();
        }
      }
    }

    garbage_type* head{nullptr};
    garbage_type* tail{nullptr};
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
    std::atomic<garbage_type*> protected_ptr{nullptr};

    // Link together all existing slots into a big global linked list
    std::atomic<HazardSlot*> next{nullptr};

    // (Intrusive) linked list of retired objects.  Does not allocate memory since it
    // just uses the next pointer from inside the retired block.
    RetiredList retired_list{};

    // Count the number of retires since the last cleanup. When this value exceeds
    // cleanup_threshold, we will perform cleanup.
    std::size_t num_retires_since_cleanup{0};

    // Set of protected objects used by cleanup().  Re-used between cleanups so that
    // we don't have to allocate new memory unless the table gets full, which would
    // only happen if the user spawns substantially more threads than were active
    // during the previous call to cleanup().
    protected_set_type protected_set{2 * std::thread::hardware_concurrency()};

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //   Above: Single-writier thread-local variables. Written by the owning thread
    // =============================================================================
    //   Below: Data shared with other threads and used by the background reclaimer
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // True if a thread owns this slot, else false.
    alignas(CACHE_LINE_ALIGNMENT) std::atomic<bool> in_use;

    // A synchronized location for the main thread to hand off garbage
    // to the background reclaimer thread.
    std::atomic<garbage_type*> available_garbage;

    // Local (unsynchronized) variables used by the background reclaimer thread.
    garbage_type* old_garbage;
    garbage_type* prev_garbage;
  };

  // Find an available hazard slot, or allocate a new one if none available.
  HazardSlot *get_slot() {
    auto current = list_head;
    while (true) {
      if (!current->in_use.load() && !current->in_use.exchange(true)) {
        return current;
      }
      if (current->next.load() == nullptr) {
        auto my_slot = new HazardSlot{true};
        HazardSlot* next = nullptr;
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
  void relinquish_slot(HazardSlot* slot) {
    slot->in_use.store(false);
  }

  // A HazardSlotOwner owns exactly one HazardSlot entry in the global linked list
  // of HazardSlots.  On creation, it acquires a free slot from the list, or appends
  // a new slot if all of them are in use.  On destruction, it makes the slot available
  // for another thread to pick up.
  struct HazardSlotOwner {
    explicit HazardSlotOwner(HazardPointers<GarbageType>& list_) : list(list_), my_slot(list.get_slot()) {}

    ~HazardSlotOwner() {
      list.relinquish_slot(my_slot);
    }

  private:
    HazardPointers<GarbageType>& list;
  public:
    HazardSlot* const my_slot;
  };

public:

  // Pre-populate the slot list with P slots, one for each hardware thread
  HazardPointers() : list_head(new HazardSlot{false}) {
    auto current = list_head;
    for (unsigned i = 1; i < std::thread::hardware_concurrency(); i++) {
      current->next = new HazardSlot{false};
      current = current->next;
    }
  }

  ~HazardPointers() {
    auto current = list_head;
    while (current) {
      auto old = std::exchange(current, current->next.load());
      delete old;
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
    static_assert(std::is_convertible_v<std::invoke_result_t<F, U>, garbage_type*>);
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
  void retire(garbage_type* p) noexcept {
    HazardSlot& my_slot = *local_slot.my_slot;
    my_slot.retired_list.push(p);

    if (++my_slot.num_retires_since_cleanup >= cleanup_threshold) [[unlikely]] {
      cleanup(my_slot);
    }
  }

  // Change the reclamation method.  Not thread safe, and not safe to call concurrently
  // while any hazard pointers are held, or while another thread might retire something.
  void set_reclamation_mode(ReclamationMethod new_mode) {
    if (new_mode == ReclamationMethod::background_thread_reclamation) {
      // Start a reclaimer thread
      reclaimer_thread = std::jthread(BackgroundReclaimer{*this});
    }
    else {
      auto stop_source = reclaimer_thread.get_stop_source();
      stop_source.request_stop();
      force_complete_background_cleanup();     // Flush all retired lists, so we don't leak anything
    }
    mode = new_mode;
  }

private:

  struct BackgroundReclaimer {

    explicit BackgroundReclaimer(HazardPointers& parent_) : parent(parent_) { }

    void operator()(std::stop_token stoken) {
      folly::asymmetric_thread_fence_heavy(std::memory_order_seq_cst);

      // Start the garbage pipeline by acquiring the set of announced hazard pointers.
      // The first set can not be interleaved since we can only retire garbage that we
      // know was retired before we read every announced hazard pointer.
      parent.for_each_slot([&](HazardSlot& slot) {
        slot.old_garbage = slot.available_garbage.exchange(nullptr);
      });
      parent.scan_hazard_pointers([&](garbage_type* p) {
        protected_set.insert(p);
      });

      while (!stoken.stop_requested()) {
        folly::asymmetric_thread_fence_heavy(std::memory_order_seq_cst);

        leftovers.cleanup([&](auto p) { return protected_set.count(p) > 0; });

        parent.for_each_slot([&](HazardSlot& slot) {
          RetiredList list(std::exchange(slot.old_garbage, slot.prev_garbage));
          list.cleanup_and_move(leftovers, [&](auto p) { return protected_set.count(p) > 0; });

          slot.prev_garbage = slot.available_garbage.exchange(nullptr);
          next_protected_set.insert(slot.protected_ptr.load());
        });

        protected_set.swap(next_protected_set);
        next_protected_set.clear();
      }
    }

    HazardPointers& parent;
    RetiredList leftovers;
    protected_set_type next_protected_set{2 * std::thread::hardware_concurrency()};
    protected_set_type protected_set{2 * std::thread::hardware_concurrency()};
  };

  template<typename F>
  void for_each_slot(F&& f) noexcept(std::is_nothrow_invocable_v<F&, HazardSlot&>) {
    auto current = list_head;
    while (current) {
      f(*current);
      current = current->next.load();
    }
  }

  // Apply the function f to all currently announced hazard pointers
  template<typename F>
  void scan_hazard_pointers(F&& f) noexcept(std::is_nothrow_invocable_v<F&, garbage_type*>) {
    for_each_slot([&, f = std::forward<F>(f)](HazardSlot& slot) {
      auto p = slot.protected_ptr.load();
      if (p) {
        f(p);
      }
    });
  }

  FOLLY_NOINLINE void cleanup(HazardSlot& slot) {
    slot.num_retires_since_cleanup = 0;

    if (mode == ReclamationMethod::amortized_reclamation) {
      // Perform reclamation now on the current thread
      folly::asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
      scan_hazard_pointers([&](auto p) { slot.protected_set.insert(p); });
      slot.retired_list.cleanup([&](auto p) { return slot.protected_set.count(p) > 0; });
      slot.protected_set.clear();    // Does not free memory, only clears contents
    }
    else {
      // Perform reclamation on the executor. Note that there is only one thread in the
      // thread pool because the reclamation method below is not thread safe.
      assert(mode == ReclamationMethod::background_thread_reclamation);

      // Hand off retired list
      garbage_type* current_available = slot.available_garbage.exchange(nullptr);
      slot.retired_list.tail->set_next(current_available);
      slot.available_garbage.store(std::exchange(slot.retired_list.head, nullptr));
    }
  }

  // Force all pending background retires to be processed.  This will clear out all
  // retired lists **without checking for hazard pointer protection**, so this
  // must not be called while holding any hazard pointers, or it could be unsafe.
  // No retires should happen concurrently either because they might trigger
  // additional reclamation which would race.
  void force_complete_background_cleanup() {
    assert(mode == ReclamationMethod::background_thread_reclamation);
    reclaimer_thread.join();
    auto current = list_head;
    while (current) {
      current->retired_list.cleanup([](auto&&) { return false; });
      current->leftovers.cleanup([](auto&&) { return false; });
      current = current->next.load();
    }
  }

  ReclamationMethod mode{ReclamationMethod::amortized_reclamation};
  HazardSlot* const list_head;

  // Background thread that reclaims unprotected garbage
  std::jthread reclaimer_thread{};

  static inline const thread_local HazardSlotOwner local_slot{get_hazard_list<garbage_type>()};
};


// Global singleton containing the list of hazard pointers.  Wrapping it
// inside an Immortal means it is never destroyed, even at termination.
//
// (i.e., a detached thread might grab a HazardSlot entry and not relinquish
// it until static destruction, at which point this global static would have
// already been destroyed. We avoid that using this pattern.)
//
// This does technically mean that we leak the HazardSlots, but that is
// a price we are willing to pay.
template<typename GarbageType>
HazardPointers<GarbageType>& get_hazard_list() {
  static_assert(std::is_trivially_destructible_v<detail::Immortal<HazardPointers<GarbageType>>>);

  static detail::Immortal<HazardPointers<GarbageType>> list;
  return list;
}

}  // namespace parlay
