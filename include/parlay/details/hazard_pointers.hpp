
#pragma once

#include <cstddef>

#include <deque>
#include <new>
#include <type_traits>
#include <unordered_set>


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

inline constexpr std::size_t CACHE_LINE_ALIGNMENT = std::hardware_destructive_interference_size;

#pragma GCC diagnostic pop

#else
inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;
#endif


// Base class for control blocks protected by hazard pointers
struct GarbageBase {
  virtual GarbageBase*& get_next() = 0;
  virtual void destroy() = 0;
};

class HazardList;

extern inline HazardList& get_hazard_list();

class HazardList {
  
  constexpr static std::size_t cleanup_threshold = 1000;
  
  using garbage_type = GarbageBase;
  
 private: 
  
  // The retired list takes advantage of available storage in the control block
  // to store the next pointers, so no additional instrusion or external next
  // pointers are required.
  //
  // Essentially, either:
  // - a control block refers to an externally allocated object via a pointer,
  //   and that object will have definitely  been deleted by the time the control
  //   block is retired, so we can re-use that pointer as the next pointer, or
  // - a control block contains an inline-allocated object, which will have been
  //   destroyed by the time the control block is retired, so we stick the next
  //   pointer inside its leftover storage.
  struct alignas(CACHE_LINE_ALIGNMENT) RetiredList {
    
    constexpr RetiredList() noexcept = default;
    
    ~RetiredList() {
      cleanup([](auto&&) { return false; });
    }
    
    void push(garbage_type* p) noexcept {
      p->get_next() = std::exchange(head, p);
      size++;
    }
    
    std::size_t get_size() const noexcept {
      return size;
    }
    
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
            prev->get_next() = current->get_next();
            current->destroy();
            current = prev->get_next();
          }
          else {
            prev = std::exchange(current, current->get_next());
          }
        }
      }
    }
    
   private:
    garbage_type* head = nullptr;
    std::size_t size = 0;
  };
  
  // Each thread owns a hazard entry slot which contains a single hazard pointer
  // (called protected_pointer) and the thread's local retired list.
  
  // The slots form a linked list so that threads can scan for the currently
  // protected pointers.
  //
  struct alignas(CACHE_LINE_ALIGNMENT) HazardSlot {
    HazardSlot(bool in_use_) : in_use(in_use_), retired{} { }
    
    std::atomic<garbage_type*> protected_ptr{nullptr};
    std::atomic<HazardSlot*> next{nullptr};
    std::atomic<bool> in_use;
    RetiredList retired;
  };
  
  HazardList() : head(new HazardSlot{false}) { }
  
  ~HazardList() {
    auto current = head;
    while (current) {
      auto old = std::exchange(current, current->next.load());
      delete old;
    }
  }
  
  friend HazardList& get_hazard_list();
  
  HazardSlot* get_slot() {
    auto current = head;
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
      }
      else {
        current = current->next.load();
      }
    }
  }
  
  void relinquish_slot(HazardSlot* slot) {
    slot->in_use.store(false);
  }
  
  struct ThreadManager {
    ThreadManager() : my_slot(get_hazard_list().get_slot()) { }
    
    ~ThreadManager() {
      get_hazard_list().relinquish_slot(my_slot);
    }
    
    HazardSlot* const my_slot;
  };

  void eject(garbage_type* p) {
    p->destroy();
  }
  
 public:
    
  // Apply the function f to all currently announced hazard pointers
  template<typename F>
  void scan_hazard_pointers(F&& f) noexcept(std::is_nothrow_invocable_v<F, garbage_type*>) {
    auto current = head;
    while (current) {
      auto p = current->protected_ptr.load();
      if (p) {
        f(p);
      }
      current = current->next.load();
    }
  }

  template<template<typename> typename Atomic, typename U, typename F>
  U protect(const Atomic<U>& src, F&& f) {
    static_assert(std::is_convertible_v<std::invoke_result_t<F, U>, garbage_type*>);
    
    U result;
    auto& slot = thread_manager.my_slot->protected_ptr;
    
    do {
      result = src.load();
      PARLAY_PREFETCH(result, 0, 0);
      slot.store(f(result));
    } while (src.load() != result);
    
    return result;
  }
  
  template<template<typename> typename Atomic, typename U>
  U protect(const Atomic<U>& src) {
    return protect(src, [](auto&& x) { return std::forward<decltype(x)>(x); });
  }
  
  void release() {
    thread_manager.my_slot->protected_ptr.store(nullptr, std::memory_order_release);
  }

  void retire(garbage_type* p) noexcept {
    RetiredList& my_list = thread_manager.my_slot->retired;
    my_list.push(p);
    
    if (my_list.get_size() >= cleanup_threshold) {
      std::unordered_set<garbage_type*> protected_blocks;
      scan_hazard_pointers([&](garbage_type* p) {
        protected_blocks.insert(p);
      });
      my_list.cleanup([&](garbage_type* p) {
        return protected_blocks.count(p) > 0;
      });
    }
  }

 private:
  HazardSlot* const head; 
  static inline const thread_local ThreadManager thread_manager;
};


// Global singleton containing the list of hazard pointers
HazardList& get_hazard_list() {
  static HazardList list;
  return list;
}

