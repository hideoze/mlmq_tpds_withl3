#pragma once
#if (L3_PROGRESS_DIAG == true)
// Each field has one writer: normally manager warp0 lane0; recovery_delays in
// helper-delay tests is written by the designated final helper's lane0.
// clock64 differences stay within one warp; no cross-device comparison is valid.
struct l3_progress_metrics {
#if (L3_IDLE_TOKEN_PROBE == true)
    unsigned long long idle_probe_skips;
#endif
    unsigned long long start, end;
    unsigned long long first_receive, last_receive;
    unsigned long long first_winner, last_commit;
    unsigned long long receive_cycles, batches, items, winners;
    unsigned long long backstop_cycles, backstop_calls;
    unsigned long long backstop_positive, term_requests, term_cancels;
    // Cancellation categories are exclusive: invalid predicate takes priority.
    unsigned long long term_invalid, term_ack_pending, term_peer_cancel;
    unsigned long long term_ack_waits; // nonblocking QUIESCING loop visits
    unsigned long long term_inbox_cancel;
    // Separate periodic caller from collaborative pre-termination recovery.
    unsigned long long periodic_cycles, periodic_calls, periodic_vertices;
    unsigned long long periodic_active, periodic_uncertified, periodic_certified;
    unsigned long long periodic_delegated;
    unsigned long long dropped_notification;
    unsigned long long recovery_epochs, recovery_helpers, recovery_done, recovery_delays;
    unsigned long long rx_seen_snapshot, rx_below_base, rx_far_ahead;
    unsigned long long rx_max_lag_buckets;
};
__device__ l3_progress_metrics g_l3_progress;
#endif
