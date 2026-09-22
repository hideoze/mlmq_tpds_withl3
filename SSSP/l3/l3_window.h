#pragma once

#ifdef __CUDACC__
#define L3_WINDOW_HD __host__ __device__
#else
#define L3_WINDOW_HD
#endif

// Scheduling only: no candidate ownership or termination state lives here.
// Signal deltas count improvements, not unique vertices. They are a heuristic.
struct l3_window_state {
    unsigned long long start = 0, signal = 0;
    unsigned long long budget = 25000;
    bool armed = false;
    unsigned long long last_probe = 0;
    bool probed = false;

    enum action { WAIT, SCAN, PROBE };

    // No extraction was attempted: reset polling cadence, not density/budget.
    L3_WINDOW_HD void deferred(unsigned long long now) {
        start = now;
        armed = true;
        probed = false;
    }

    // Decide from private state before sampling any queue or peer state.
    // A fixed window never needs an idle probe. Dynamic probes are bounded
    // by the same minimum cooldown as early draining; deadlines always win.
    L3_WINDOW_HD action decision(unsigned long long now, bool dynamic,
                                 unsigned long long maximum) {
        if (!armed) { start = now; armed = true; }
        const auto elapsed = now - start;
        if (elapsed >= (dynamic ? budget : maximum)) return SCAN;
        if (dynamic && elapsed >= 25000 &&
            (!probed || now - last_probe >= 25000)) {
            last_probe = now;
            probed = true;
            return PROBE;
        }
        return WAIT;
    }

    L3_WINDOW_HD void effective_feedback(unsigned count, unsigned effective,
                                         unsigned long long maximum) {
        if (!count) return;
        // Low winner rate means the peer is seeing stale/redundant candidates;
        // coalesce the next batch. A productive batch restores responsiveness.
        if (effective * 4u < count)
            budget = budget < maximum / 2 ? budget * 2 : maximum;
        else if (effective * 2u >= count)
            budget = budget / 2;
        if (budget < 25000) budget = 25000;
        if (budget > maximum) budget = maximum;
    }

    L3_WINDOW_HD bool allow(unsigned long long now, bool dynamic,
                            bool drain, unsigned long long maximum) {
        if (!armed) { start = now; armed = true; }
        const auto limit = dynamic ? budget : maximum;
        // Even drain must respect a short cooldown; idle hints alone must not
        // trigger unbounded empty scans. Caller also requires a new event.
        return (dynamic && drain && now - start >= 25000) || now - start >= limit;
    }

    L3_WINDOW_HD void scanned(unsigned long long now, unsigned long long next_signal,
                              unsigned count, bool dynamic,
                              unsigned long long maximum) {
        const auto events = next_signal - signal;
        if (dynamic) {
            // Empty means no useful extraction, not latency-sensitive work.
            // Back off instead of driving idle scans to the minimum budget.
            if (!count || events / count >= 2)
                budget = budget > maximum / 2 ? maximum : budget * 2;
            else
                budget = budget / 2;
            if (budget < 25000) budget = 25000;
            if (budget > maximum) budget = maximum;
        }
        signal = next_signal;
        start = now;
        armed = true;
        probed = false;
    }
};
#undef L3_WINDOW_HD
