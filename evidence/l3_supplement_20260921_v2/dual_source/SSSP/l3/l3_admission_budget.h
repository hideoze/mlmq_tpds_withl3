#pragma once
#ifdef __CUDACC__
#define L3_ADMISSION_HD __host__ __device__
#else
#define L3_ADMISSION_HD
#endif
struct l3_admission_cursor {
    unsigned long long start=0;
    bool armed=false;
    L3_ADMISSION_HD bool allow(unsigned long long now,int minimum,int horizon,bool force) {
        if(force || minimum<=horizon) { armed=false;return true; }
        if(!armed) { armed=true;start=now; }
        if(now-start>=25000ull) { armed=false;return true; }
        return false;
    }
};
#undef L3_ADMISSION_HD
