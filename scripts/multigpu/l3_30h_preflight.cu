#include <cuda_runtime.h>

#include <cstdio>

static bool ok(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA_FAIL op=%s rc=%d message=%s\n",
                 what, int(rc), cudaGetErrorString(rc));
    return false;
}

int main() {
    int count = 0;
    if (!ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return 2;
    std::printf("CUDA_DEVICE_COUNT count=%d\n", count);
    if (count != 2) {
        std::fprintf(stderr, "EXPECTED_EXACTLY_TWO_VISIBLE_GPUS actual=%d\n", count);
        return 3;
    }
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop{};
        if (!ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return 4;
        if (!ok(cudaSetDevice(device), "cudaSetDevice")) return 5;
        size_t free_bytes = 0, total_bytes = 0;
        if (!ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) return 6;
        std::printf(
            "CUDA_DEVICE id=%d name=%s capability=%d.%d sm=%d total_bytes=%zu free_bytes=%zu\n",
            device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            total_bytes, free_bytes);
    }
    for (int source = 0; source < count; ++source) {
        for (int target = 0; target < count; ++target) {
            if (source == target) continue;
            int access = 0, rank = -1, atomics = 0;
            if (!ok(cudaDeviceCanAccessPeer(&access, source, target),
                    "cudaDeviceCanAccessPeer")) return 7;
            if (!ok(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank,
                                              source, target),
                    "cudaDevP2PAttrPerformanceRank")) return 8;
            if (!ok(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                                              source, target),
                    "cudaDevP2PAttrNativeAtomicSupported")) return 9;
            std::printf("CUDA_P2P source=%d target=%d access=%d rank=%d native_atomics=%d\n",
                        source, target, access, rank, atomics);
            if (!access) return 10;
        }
    }
    std::puts("CUDA_PREFLIGHT PASS");
    return 0;
}
