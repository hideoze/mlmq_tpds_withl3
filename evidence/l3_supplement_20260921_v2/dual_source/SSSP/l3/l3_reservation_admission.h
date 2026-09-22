#pragma once
// Advisory permission for a NEW L2 reservation, never cancellation of an
// existing ticket or task. Primary readers remain unconditional.
// Stale counters may admit extra readers or delay auxiliaries; neither counter
// supplies the actual ticket (that still comes from the queue's atomicAdd).
#ifdef __CUDACC__
#define MLMQ_RESERVATION_HD __host__ __device__
#else
#define MLMQ_RESERVATION_HD
#endif
MLMQ_RESERVATION_HD inline bool l3_allow_new_reservation(
    bool primary, bool has_ticket, int next_ticket_seen, int published_seen) {
    return primary || has_ticket || next_ticket_seen < published_seen;
}
#undef MLMQ_RESERVATION_HD
