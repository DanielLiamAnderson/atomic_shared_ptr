# A Lock-Free Atomic Shared Pointer

This repository contains my lock-free **atomic_shared_ptr** implementation that I discussed at CppCon 2023 in
"*Lock-free Atomic Shared Pointers Without a Split Reference Count? It Can Be Done!*". Its still a proof of concept
and not quite ready for production, but it is reasonably featureful.

## Dependencies

The library currently depends on [Folly](https://github.com/facebook/folly) (for asymmetric fences and F14FastSet) and
[ParlayLib](https://github.com/cmuparlay/parlaylib) (for its allocator). The library itself is completely header only.


## Usage

```c++
#include <parlay/atomic_shared_ptr.hpp>

parlay::atomic_shared_ptr<int> asp{parlay::make_shared<int>(42)};

parlay::shared_ptr<int> sp = asp.load();  // {42} has a reference count of 2
sp = parlay::make_shared<int>(1);  // {42} has a reference count of 1
asp.store(sp);    // {42} is destroyed because the last owner is gone
```

## Learning

If you'd like to learn about the algorithm, you should watch my CppCon talk if you have not already.  If you'd like to
see the code, there is a "basic" implementation in [basic_atomic_shared_ptr.hpp](./include/parlay/basic_atomic_shared_ptr.hpp).
It is unoptimized and very feature incomplete, but it is designed to be as readible as possible to demonstrate the algorithm
in a simple and understandable way. It uses [Folly's Hazard Pointers](https://github.com/facebook/folly/blob/main/folly/synchronization/Hazptr.h)
for the protection of the control block, which adds some additional overhead.

The more optimized and feature complete implementations can be found in
[atomic_shared_ptr.hpp](./include/parlay/atomic_shared_ptr.hpp) and [shared_ptr.hpp](./include/parlay/shared_ptr.hpp).
They use a custom-implemented Hazard Pointer library, [hazard_pointers.hpp](./include/parlay/details/hazard_pointers.hpp).


## Benchmarks

There are some basic latency benchmarks in the [benchmark](./benchmark) directory. The throughput benchmarks from my
CppCon talk can be found in the **new_sps** branch of [this repository](https://github.com/cmuparlay/concurrent_deferred_rc/tree/new_sps)
(it was much easier to integrate *this* library into my existing benchmarks from a previous project than to do it
the other way around, sorry!)


## Deamortized Reclamation (Experimental Feature!)

One drawback of Hazard-Pointer based cleanup schemes is that while they produce high throughput, they are not ideal for
latency-critical applications since once in every while, a thread will have to perform a garbage collection operation
which could take tens of microseconds, while an ordinary store operations would only take tens of nanoseconds.  This
spike in latency is undesirable in certain domains.

To address this, the library comes with an **experimental** feature, deamortized reclamation.  This means that instead
of accumulating garbage and then performing cleanup once every 1000 retires or so, it tries to incrementally scan the
Hazard Pointers one per retire and delete one or two ready-to-reclaim objects each time.  This effectively spreads out (formally,
deamortizes) the cleanup operation so that it does not introduce large latency spikes. As a tradeoff, load latency is
increase by about 25% in uncontended benchmarks (from 16ns to 20ns), but the benefit is reducing the tail latency of
a store operation from 14 microseconds to just over 100 nanoseconds.

More thorough benchmarks are needed to determine the impliciations of this technique, and there is plenty of room left
to optimize it.


## Work in Progress

**Implementation**
- Support for `atomic_weak_ptr`
- Support for aliased `shared_ptr`

**Cleanup**
- Port the throughput benchmarks into this repository
- Drop the dependence on Folly when/if we get asymmetric fences in the standard library
- Drop the dependence on ParlayLib if we can fix the allocator problem (essentially, jemalloc **hates** deferred
  reclamation and it performs terribly, so I need to use my own allocator for now instead).
- Put some CI in this repository
