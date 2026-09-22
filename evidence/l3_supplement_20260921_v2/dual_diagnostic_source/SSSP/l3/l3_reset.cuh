#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Query reset only, while all previous solver work is quiescent. Typed stores
// preserve INT_MAX / FLT_MAX / DBL_MAX exactly (a byte memset does not).
template <typename T>
__global__ void l3_fill_values_kernel(T *first, T *second, size_t count, T value)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        first[i] = value;
        if (second) second[i] = value;
    }
}

// Enqueued on the calling device's default stream. The caller must complete
// reset before publishing work to either GPU; sssp_re_init already synchronizes.
template <typename T>
inline cudaError_t l3_fill_values(T *first, size_t count, T value, T *second = nullptr)
{
    if (!count) return cudaSuccess;
    if (!first) return cudaErrorInvalidValue;
    const size_t required = (count + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(required < 4096 ? required : 4096);
    l3_fill_values_kernel<<<blocks, 256>>>(first, second, count, value);
    return cudaGetLastError();
}
