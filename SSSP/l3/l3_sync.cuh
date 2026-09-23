#pragma once

#include <cuda/atomic>

// Scoped helpers for L3 protocol words. Legacy CUDA atomics are relaxed and
// device scoped unless suffixed; use these wrappers only where the word is a
// publication/ownership signal.
template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ T l3_atomic_load_relaxed(T *word)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    return ref.load(cuda::memory_order_relaxed);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ T l3_atomic_load_acquire(T *word)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    return ref.load(cuda::memory_order_acquire);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ void l3_atomic_store_relaxed(T *word, T value)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    ref.store(value, cuda::memory_order_relaxed);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ void l3_atomic_store_release(T *word, T value)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    ref.store(value, cuda::memory_order_release);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ T l3_atomic_exchange_acq_rel(T *word, T value)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    return ref.exchange(value, cuda::memory_order_acq_rel);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ T l3_atomic_fetch_add_acq_rel(T *word, T value)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    return ref.fetch_add(value, cuda::memory_order_acq_rel);
}

template <cuda::thread_scope Scope, typename T>
__device__ __forceinline__ bool l3_atomic_compare_exchange_acq_rel(
    T *word, T expected, T desired)
{
    cuda::atomic_ref<T, Scope> ref(*word);
    return ref.compare_exchange_strong(
        expected, desired, cuda::memory_order_acq_rel,
        cuda::memory_order_acquire);
}

__device__ __forceinline__ unsigned l3_device_mark_publish(
    unsigned *word, unsigned bits)
{
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> ref(*word);
    return ref.fetch_or(bits, cuda::memory_order_release);
}

__device__ __forceinline__ bool l3_device_mark_claim(
    unsigned *word, unsigned snapshot)
{
    return l3_atomic_compare_exchange_acq_rel<cuda::thread_scope_device>(
        word, snapshot, 0u);
}

// dirty_bitmap is owned by one GPU, but the peer GPU may publish improvements
// directly into it. Both local and peer producers/consumers therefore use a
// common system-scope protocol on this authoritative bitmap.
__device__ __forceinline__ unsigned l3_system_mark_publish(
    unsigned *word, unsigned bits)
{
    cuda::atomic_ref<unsigned, cuda::thread_scope_system> ref(*word);
    return ref.fetch_or(bits, cuda::memory_order_release);
}

__device__ __forceinline__ bool l3_system_mark_claim(
    unsigned *word, unsigned snapshot)
{
    return l3_atomic_compare_exchange_acq_rel<cuda::thread_scope_system>(
        word, snapshot, 0u);
}

// A peer distance lives in the other GPU's allocation. Use system-scope CAS
// so this update is atomic with both peer updates and the owner GPU's updates.
template <typename T>
__device__ __forceinline__ T l3_atomic_min_system(T *word, T value)
{
    cuda::atomic_ref<T, cuda::thread_scope_system> ref(*word);
    T old = ref.load(cuda::memory_order_relaxed);
    while (value < old)
    {
        T expected = old;
        if (ref.compare_exchange_weak(expected, value,
                                      cuda::memory_order_relaxed,
                                      cuda::memory_order_relaxed))
            return old;
        old = expected;
    }
    return old;
}
