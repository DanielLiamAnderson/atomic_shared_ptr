#pragma once

#include <atomic>
#include <type_traits>

// A wait-free atomic counter that supports increment and decrement,
// such that attempting to increment the counter from zero fails and
// does not perform the increment.
//
// Useful for incrementing reference counting, where the underlying
// managed memory is freed when the counter hits zero, so that other
// racing threads can not increment the counter back up from zero
//
// Note: The counter steals the top two bits of the integer for book-
// keeping purposes. Hence the maximum representable value in the
// counter is 2^(8*sizeof(T)-2) - 1
template<typename T>
struct WaitFreeCounter {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

  WaitFreeCounter() noexcept : x(1) {}
  explicit WaitFreeCounter(T desired) noexcept : x(desired) {}

  [[nodiscard]] bool is_lock_free() const { return true; }
  static constexpr bool is_always_lock_free = true;
  [[nodiscard]] constexpr T max_value() const { return zero_pending_flag - 1; }

  WaitFreeCounter& operator=(const WaitFreeCounter&) = delete;

  explicit operator T() const noexcept { return load(); }

  T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    auto val = x.load(order);
    if (val == 0 && x.compare_exchange_strong(val, zero_flag | zero_pending_flag)) [[unlikely]] return 0;
    return (val & zero_flag) ? 0 : val;
  }

  // Increment the counter by arg. Returns false on failure, i.e., if the counter
  // was previously zero. Otherwise returns true.
  T increment(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    auto val = x.fetch_add(arg, order);
    return (val & zero_flag) == 0;
  }

  // Decrement the counter by arg. Returns true if this operation was responsible
  // for decrementing the counter to zero. Otherwise, returns false.
  bool decrement(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
    if (x.fetch_sub(arg, order) == arg) {
      T expected = 0;
      if (x.compare_exchange_strong(expected, zero_flag)) [[likely]]
        return true;
      else if ((expected & zero_pending_flag) && (x.exchange(zero_flag) & zero_pending_flag))
        return true;
    }
    return false;
  }

private:
  static constexpr inline T zero_flag = T(1) << (sizeof(T)*8) - 1;
  static constexpr inline T zero_pending_flag = T(1) << (sizeof(T)*8) - 2;
  mutable std::atomic<T> x;
};
