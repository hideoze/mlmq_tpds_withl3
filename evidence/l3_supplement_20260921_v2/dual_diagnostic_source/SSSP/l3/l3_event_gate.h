#pragma once
#ifdef __CUDACC__
#define L3_EVENT_HD __host__ __device__
#else
#define L3_EVENT_HD
#endif

// Lane-private scheduling hints, never candidate ownership or termination truth.
struct l3_event_gate {
    unsigned long long observed = 0, last_full = 0;
    bool pending = true;  // Include candidates produced before TX startup.

    L3_EVENT_HD bool full_due(unsigned long long now,
                               unsigned long long interval) const {
        return now - last_full >= interval;
    }
    L3_EVENT_HD bool work_hint(unsigned long long signal) const {
        return pending || signal != observed;
    }
    L3_EVENT_HD void scanned(unsigned long long now,
                              unsigned long long signal_before_scan,
                              unsigned count, bool full) {
        // Never acknowledge events arriving while the collector is running.
        observed = signal_before_scan;
        // A nonempty batch may have left candidates behind at a lane/cap limit.
        // One empty follow-up is intentional; hidden hints use the full timer.
        pending = count != 0;
        if (full) last_full = now;
    }
};
#undef L3_EVENT_HD
