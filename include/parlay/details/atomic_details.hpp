#pragma once

#include <atomic>

namespace parlay {
namespace details {
  
inline std::memory_order default_failure_memory_order(std::memory_order successMode) {
  switch (successMode) {
    case std::memory_order_acq_rel:
      return std::memory_order_acquire;
    case std::memory_order_release:
      return std::memory_order_relaxed;
    case std::memory_order_relaxed:
    case std::memory_order_consume:
    case std::memory_order_acquire:
    case std::memory_order_seq_cst:
      return successMode;
  }
  return successMode;
}  

}  // namespace details
}  // namespace parlay
