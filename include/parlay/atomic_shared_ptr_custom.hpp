#pragma once

#include <atomic>

#include "details/atomic_details.hpp"
#include "details/hazard_pointers.hpp"

//#include "shared_ptr_opt.hpp"
#include "shared_ptr.hpp"

namespace parlay {

inline void enable_deamortized_reclamation() {
  //get_hazard_list<parlay::details::control_block_base>().enable_deamortized_reclamation();;
}

template<typename T>
class atomic_shared_ptr {
  
  using shared_ptr_type = shared_ptr<T>;
  using control_block_type = details::control_block_base;
  //using control_block_type = details::fast_control_block<T>;
  
 public:
  
  constexpr atomic_shared_ptr() noexcept = default;
  constexpr explicit(false) atomic_shared_ptr(std::nullptr_t) noexcept    // NOLINT(google-explicit-constructor)
    : control_block{nullptr} { }
  
  explicit(false) atomic_shared_ptr(shared_ptr_type desired) {            // NOLINT(google-explicit-constructor)
    auto [ptr_, control_block_] = desired.release_internals();
    control_block.store(control_block_, std::memory_order_relaxed);
  }
  
  atomic_shared_ptr(const atomic_shared_ptr&) = delete;
  atomic_shared_ptr& operator=(const atomic_shared_ptr&) = delete;
  
  ~atomic_shared_ptr() { store(nullptr); }
  
  bool is_lock_free() const noexcept {
    return control_block.is_lock_free();
  }
  
  constexpr static bool is_always_lock_free = std::atomic<control_block_type*>::is_always_lock_free;
  
  [[nodiscard]] shared_ptr_type load([[maybe_unused]] std::memory_order order = std::memory_order_seq_cst) const {
    control_block_type* current_control_block = nullptr;
    
    auto& hazptr = get_hazard_list<control_block_type>();
    
    while (true) {
      current_control_block = hazptr.protect(control_block);
      if (current_control_block == nullptr || current_control_block->increment_strong_count_if_nonzero()) break;
    }

    return make_shared_from_ctrl_block(current_control_block);
  }
  
  void store(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) {
    auto [ptr_, control_block_] = desired.release_internals();
    auto old_control_block = control_block.exchange(control_block_, order);
    if (old_control_block) {
      old_control_block->decrement_strong_count();
    }
  }
  
  shared_ptr_type exchange(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
    auto [ptr_, control_block_] = desired.release_internals();
    auto old_control_block = control_block.exchange(control_block_, order);
    return make_shared_from_ctrl_block(old_control_block);
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, shared_ptr_type&& desired,
      std::memory_order success, std::memory_order failure) {

      auto expected_ctrl_block = expected.control_block;
      auto desired_ctrl_block = desired.control_block;

      if (control_block.compare_exchange_weak(expected_ctrl_block, desired_ctrl_block, success, failure)) {
        if (expected_ctrl_block) {
          expected_ctrl_block->decrement_strong_count();
        }
        desired.release_internals();
        return true;
      }
      else {
        expected = load();   // It's possible that expected ABAs and stays the same on failure, hence
        return false;        // why this algorithm can not be used to implement compare_exchange_strong
      }
  }
  
  bool compare_exchange_strong(shared_ptr_type& expected, shared_ptr_type&& desired,
      std::memory_order success, std::memory_order failure) {

    auto expected_ctrl_block = expected.control_block;

    // If expected changes then we have completed the operation (unsuccessfully), we only
    // have to loop in case expected ABAs or the weak operation fails spuriously.
    do {
      if (compare_exchange_weak(expected, std::move(desired), success, failure)) {
        return true;
      }
    } while (expected_ctrl_block == expected.control_block);
    
    return false;
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, const shared_ptr_type& desired,
      std::memory_order success, std::memory_order failure) {
    
    // This version is not very efficient and should be avoided.  It's just here to provide the complete
    // API of atomic<shared_ptr>.  The issue with it is that if the compare_exchange fails, the reference
    // count of desired is incremented and decremented for no reason.  On the other hand, the rvalue
    // version doesn't modify the reference count of desired at all.
    
    return compare_exchange_weak(expected, shared_ptr_type{desired}, success, failure);
  }
  
  
  bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired,
      std::memory_order success, std::memory_order failure) {
  
    // This version is not very efficient and should be avoided.  It's just here to provide the complete
    // API of atomic<shared_ptr>.  The issue with it is that if the compare_exchange fails, the reference
    // count of desired is incremented and decremented for no reason.  On the other hand, the rvalue
    // version doesn't modify the reference count of desired at all.
  
    return compare_exchange_strong(expected, shared_ptr_type{desired}, success, failure);
  }
  

  bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_strong(expected, desired, order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, const shared_ptr_type& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_weak(expected, desired, order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_strong(shared_ptr_type& expected, shared_ptr_type&& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_strong(expected, std::move(desired), order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, shared_ptr_type&& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_weak(expected, std::move(desired), order, details::default_failure_memory_order(order));
  }
  
 private:
 
  static shared_ptr_type make_shared_from_ctrl_block(control_block_type* control_block_) {
    if (control_block_) {
      T* ptr = static_cast<T*>(control_block_->get_ptr());
      return shared_ptr_type{ptr, control_block_};
    }
    else {
      return shared_ptr_type{nullptr};
    }
  }
 
  mutable std::atomic<control_block_type*> control_block;
};

};

