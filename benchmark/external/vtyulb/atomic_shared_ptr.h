#pragma once

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <thread>
#include <stack>

namespace LFStructs {

const int MAGIC_LEN = 16;
const size_t MAGIC_MASK = 0x0000'0000'0000'FFFF;
const int CACHE_LINE_SIZE = 128;

template<typename T>
struct alignas(CACHE_LINE_SIZE) ControlBlock {
    explicit ControlBlock() = delete;
    explicit ControlBlock(T *data)
        : data(data)
        , refCount(1)
    {
        assert(reinterpret_cast<size_t>(data) <= 0x0000'FFFF'FFFF'FFFF);
    }

    T *data;
    std::atomic<size_t> refCount;
};


template<typename T>
class SharedPtr {
public:
    SharedPtr(): controlBlock(nullptr) {}
    explicit SharedPtr(T *data)
        : controlBlock(new ControlBlock<T>(data))
    {

    }
    explicit SharedPtr(ControlBlock<T> *controlBlock): controlBlock(controlBlock) {}
    SharedPtr(const SharedPtr &other) {
        controlBlock = other.controlBlock;
        if (controlBlock != nullptr) {
            int before = controlBlock->refCount.fetch_add(1);
            assert(before);
        }
    };
    SharedPtr(SharedPtr &&other) noexcept {
        controlBlock = other.controlBlock;
        other.controlBlock = nullptr;
    };
    SharedPtr& operator=(const SharedPtr &other) {
        auto old = controlBlock;
        controlBlock = other.controlBlock;
        if (controlBlock != nullptr) {
            int before = controlBlock->refCount.fetch_add(1);
            assert(before);
        }
        unref(old);
        return *this;
    };
    SharedPtr& operator=(SharedPtr &&other) {
        if (controlBlock != other.controlBlock) {
            auto old = controlBlock;
            controlBlock = other.controlBlock;
            other.controlBlock = nullptr;
            unref(old);
        }
        return *this;
    }

    SharedPtr& operator=(std::nullptr_t) {
        auto old = std::exchange(controlBlock, nullptr);
        unref(old);
        return *this;
    }

    ~SharedPtr() {
        thread_local std::vector<ControlBlock<T>*> destructionQueue;
        thread_local bool destructionInProgress = false;

        destructionQueue.push_back(controlBlock);
        if (!destructionInProgress) {
            destructionInProgress = true;
            while (destructionQueue.size()) {
                ControlBlock<T> *blockToUnref = destructionQueue.back();
                destructionQueue.pop_back();
                unref(blockToUnref);
            }
            destructionInProgress = false;
        }
    }

    SharedPtr copy() { return SharedPtr(*this); }
    T* get() const { return controlBlock ? controlBlock->data : nullptr; }
    T* operator->() const { return controlBlock->data; }

    /* implicit */ explicit(false) operator bool() const noexcept { return get() != nullptr; }      // NOLINT

private:
    void unref(ControlBlock<T> *blockToUnref) {
        if (blockToUnref) {
            int before = blockToUnref->refCount.fetch_sub(1);
            assert(before);
            if (before == 1) {
                delete blockToUnref->data;
                delete blockToUnref;
            }
        }
    }

    template<typename A> friend class AtomicSharedPtr;
    ControlBlock<T> *controlBlock;
};


template<typename T>
class alignas(CACHE_LINE_SIZE) FastSharedPtr {
public:
    FastSharedPtr(const FastSharedPtr<T> &other) = delete;
    FastSharedPtr(FastSharedPtr<T> &&other)
        : knownValue(other.knownValue)
        , foreignPackedPtr(other.foreignPackedPtr)
        , data(other.data)
    {
        other.foreignPackedPtr = nullptr;
    };
    FastSharedPtr& operator=(FastSharedPtr<T> &&other) {
        destroy();
        knownValue = other.knownValue;
        foreignPackedPtr = other.foreignPackedPtr;
        data = other.data;
        other.foreignPackedPtr = nullptr;
        return *this;
    }
    ~FastSharedPtr() {
        destroy();
    };

    ControlBlock<T>* getControlBlock() { return reinterpret_cast<ControlBlock<T>*>(knownValue >> MAGIC_LEN); }
    T* get() { return data; }
    T* operator->(){ return data; }
private:
    void destroy() {
        if (foreignPackedPtr != nullptr) {
            size_t expected = knownValue;
            while (!foreignPackedPtr->compare_exchange_weak(expected, expected - 1)) {
                if (((expected >> MAGIC_LEN) != (knownValue >> MAGIC_LEN)) || !(expected & MAGIC_MASK)) {
                    ControlBlock<T> *block = reinterpret_cast<ControlBlock<T>*>(knownValue >> MAGIC_LEN);
                    size_t before = block->refCount.fetch_sub(1);
                    if (before == 1) {
                        delete data;
                        delete block;
                    }
                    break;
                }
            }
        }
    }
    FastSharedPtr(std::atomic<size_t> *packedPtr)
        : knownValue(packedPtr->fetch_add(1) + 1)
        , foreignPackedPtr(packedPtr)
        , data(getControlBlock()->data)
    {
        auto block = getControlBlock();
        int diff = knownValue & MAGIC_MASK;
        while (diff > 1000 && block == getControlBlock()) {
            block->refCount.fetch_add(diff);
            if (packedPtr->compare_exchange_strong(knownValue, knownValue - diff)) {
                foreignPackedPtr = nullptr;
                break;
            }
            block->refCount.fetch_sub(diff);
            diff = knownValue & MAGIC_MASK;
        }
    };

    size_t knownValue;
    std::atomic<size_t> *foreignPackedPtr;
    T *data;

    template<typename A> friend class AtomicSharedPtr;
};


template<typename T>
class alignas(CACHE_LINE_SIZE) AtomicSharedPtr {
public:
    AtomicSharedPtr(T *data = nullptr);
    ~AtomicSharedPtr();

    AtomicSharedPtr(const AtomicSharedPtr &other) = delete;
    AtomicSharedPtr(AtomicSharedPtr &&other) = delete;
    AtomicSharedPtr& operator=(const AtomicSharedPtr &other) = delete;
    AtomicSharedPtr& operator=(AtomicSharedPtr &&other) = delete;

    SharedPtr<T> load();
    FastSharedPtr<T> getFast();

    bool compareExchange(T *expected, SharedPtr<T> &&newOne);                       // this actually is strong version

    bool compare_exchange_strong(SharedPtr<T>& expected, SharedPtr<T> desired);      // this actually is strong version
    bool compare_exchange_weak(SharedPtr<T>& expected, SharedPtr<T> desired);

    void store(T *data);
    void store(SharedPtr<T>&& data);

private:
    void destroyOldControlBlock(size_t oldPackedPtr);

    /* first 48 bit - pointer to control block
     * last 16 bit - local refcount if anyone is accessing control block
     * through current AtomicSharedPtr instance right now */
    std::atomic<size_t> packedPtr;
    static_assert(sizeof(T*) == sizeof(size_t));
};

template<typename T>
AtomicSharedPtr<T>::AtomicSharedPtr(T *data) {
    auto block = new ControlBlock(data);
    packedPtr.store(reinterpret_cast<size_t>(block) << MAGIC_LEN);
}

template<typename T>
SharedPtr<T> AtomicSharedPtr<T>::load() {
    // taking copy and notifying about read in progress
    size_t packedPtrCopy = packedPtr.fetch_add(1);
    auto block = reinterpret_cast<ControlBlock<T>*>(packedPtrCopy >> MAGIC_LEN);
    int before = block->refCount.fetch_add(1);
    assert(before);
    // copy is completed

    // notifying about completed copy
    size_t expected = packedPtrCopy + 1;
    while (true) {
        assert((expected & MAGIC_MASK) > 0);
        size_t expCopy = expected;
        if (packedPtr.compare_exchange_weak(expected, expected - 1)) {
            break;
        }

        // if control block pointer just changed, then
        // handling object's refcount is not our responsibility
        if (((expected >> MAGIC_LEN) != (packedPtrCopy >> MAGIC_LEN)) ||
                ((expected & MAGIC_MASK) == 0)) // >20 hours wasted here
        {
            int before = block->refCount.fetch_sub(1);
            assert(before);
            break;
        }

        if ((expected & MAGIC_MASK) == 0) {
            abort();
            break;
        }
    }
    // notification finished

    return SharedPtr<T>(block);
}

template<typename T>
FastSharedPtr<T> AtomicSharedPtr<T>::getFast() {
    return FastSharedPtr<T>(&packedPtr);
}

template<typename T>
AtomicSharedPtr<T>::~AtomicSharedPtr() {
    thread_local std::vector<size_t> destructionQueue;
    thread_local bool destructionInProgress = false;

    size_t packedPtrCopy = packedPtr.load();
    auto block = reinterpret_cast<ControlBlock<T>*>(packedPtrCopy >> MAGIC_LEN);
    size_t diff = packedPtrCopy & MAGIC_MASK;
    if (diff != 0) {
        block->refCount.fetch_add(diff);
    }

    destructionQueue.push_back(packedPtrCopy);
    if (!destructionInProgress) {
        destructionInProgress = true;
        while (destructionQueue.size()) {
            size_t controlBlockToDestroy = destructionQueue.back();
            destructionQueue.pop_back();
            destroyOldControlBlock(controlBlockToDestroy);
        }
        destructionInProgress = false;
    }
}

template<typename T>
void AtomicSharedPtr<T>::store(T *data) {
    store(SharedPtr<T>(data));
}

template<typename T>
void AtomicSharedPtr<T>::store(SharedPtr<T> &&data) {
    while (true) {
        auto holder = this->getFast();
        if (compareExchange(holder.get(), std::move(data))) {
            break;
        }
    }
}

template<typename T>
bool AtomicSharedPtr<T>::compareExchange(T *expected, SharedPtr<T> &&newOne) {
    if (expected == newOne.get()) {
        return true;
    }
    auto holder = this->getFast();
    if (holder.get() == expected) {
        size_t holdedPtr = reinterpret_cast<size_t>(holder.getControlBlock());
        size_t desiredPackedPtr = reinterpret_cast<size_t>(newOne.controlBlock) << MAGIC_LEN;
        size_t expectedPackedPtr = holdedPtr << MAGIC_LEN;
        while (holdedPtr == (expectedPackedPtr >> MAGIC_LEN)) {
            if (expectedPackedPtr & MAGIC_MASK) {
                int diff = expectedPackedPtr & MAGIC_MASK;
                holder.getControlBlock()->refCount.fetch_add(diff);
                if (!packedPtr.compare_exchange_weak(expectedPackedPtr, expectedPackedPtr & ~MAGIC_MASK)) {
                    holder.getControlBlock()->refCount.fetch_sub(diff);
                }
                continue;
            }
            assert((expectedPackedPtr >> MAGIC_LEN) != (desiredPackedPtr >> MAGIC_LEN));
            if (packedPtr.compare_exchange_weak(expectedPackedPtr, desiredPackedPtr)) {
                newOne.controlBlock = nullptr;
                assert((expectedPackedPtr >> MAGIC_LEN) == holdedPtr);
                destroyOldControlBlock(expectedPackedPtr);
                return true;
            }
        }
    }

    return false;
}

template<typename T>
bool AtomicSharedPtr<T>::compare_exchange_strong(SharedPtr<T>& expected, SharedPtr<T> desired) {
  return compareExchange(expected.get(), std::move(desired));
}

template<typename T>
bool AtomicSharedPtr<T>::compare_exchange_weak(SharedPtr<T>& expected, SharedPtr<T> desired) {
  return compareExchange(expected.get(), std::move(desired));
}

template<typename T>
void AtomicSharedPtr<T>::destroyOldControlBlock(size_t oldPackedPtr) {
    auto block = reinterpret_cast<ControlBlock<T>*>(oldPackedPtr >> MAGIC_LEN);
    auto refCountBefore = block->refCount.fetch_sub(1);
    assert(refCountBefore);
    if (refCountBefore == 1) {
        delete block->data;
        delete block;
    }
}

} // namespace LFStructs
