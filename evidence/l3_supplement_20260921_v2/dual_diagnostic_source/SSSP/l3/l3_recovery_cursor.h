#pragma once
#ifdef __CUDACC__
#define L3_CURSOR_HD __host__ __device__
#else
#define L3_CURSOR_HD
#endif
struct l3_scan_slice { int begin, end; };
struct l3_bounded_scan_cursor {
    int next = 1;
    L3_CURSOR_HD l3_scan_slice take(int size, int budget) {
        if (size <= 0) { next = 1; return {1, 1}; }
        const int remaining = size - (next - 1);
        const int length = remaining < budget ? remaining : budget;
        l3_scan_slice slice{next, next + length};
        next = slice.end == size + 1 ? 1 : slice.end;
        return slice;
    }
};
#undef L3_CURSOR_HD
