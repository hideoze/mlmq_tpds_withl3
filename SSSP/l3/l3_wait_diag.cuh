#pragma once
#ifndef L3_WAIT_DIAG
#define L3_WAIT_DIAG false
#endif
#if (L3_WAIT_DIAG == true)
// Single lane writer, same-warp clock domain. No diagnostic value drives work.
struct l3_wait_sample {
    unsigned long long cycles[6], span, last_work;
};
__device__ l3_wait_sample g_l3_wait[258];
__device__ unsigned long long g_l3_wait_reason[18];
struct l3_wait_clock {
    l3_wait_sample data = {};
    unsigned long long start = 0, last = 0;
    int phase = 0;
    bool enabled;
    __device__ explicit l3_wait_clock(bool on): enabled(on) {
        if(enabled) start = last = clock64();
    }
    __device__ void tick(int next) {
        if(enabled) {
            const auto now = clock64();
            data.cycles[phase] += now-last;
            last=now; phase=next;
        }
    }
    __device__ void finish_read(bool ready) {
        if(enabled && ready) phase=2;
        tick(0);
    }
    __device__ void finish_work() {
        tick(0);
        if(enabled) data.last_work=last-start;
    }
    __device__ void save(int slot) {
        tick(0);
        if(enabled) { data.span=last-start; g_l3_wait[slot]=data; }
    }
};
#endif
