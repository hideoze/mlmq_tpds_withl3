#pragma once
#ifdef __CUDACC__
#define L3_RECOVERY_HD __host__ __device__
#else
#define L3_RECOVERY_HD
#endif
struct l3_recovery_range { int begin, end; };
L3_RECOVERY_HD inline l3_recovery_range l3_recovery_slice(int size, int rank, int warps) {
    const int per = size / warps, rem = size % warps;
    const int begin = 1 + rank * per + (rank < rem ? rank : rem);
    return {begin, begin + per + (rank < rem ? 1 : 0)};
}
struct l3_recovery_cursor {
    int seen = 0;
    L3_RECOVERY_HD bool fresh(int request) const { return request > seen; }
    L3_RECOVERY_HD void completed(int request) { seen = request; }
};
#undef L3_RECOVERY_HD
