#pragma once

// 第一版 DW-L3 的动态部分只调整“何时扫描/发送”，不改变 remote_cand/mark
// 的权威语义，也不建立第二套完整 Delta queue。
__device__ __forceinline__ void l3_dynamic_reset_after_scan(
    l3_dynamic_state &state, int batch_count)
{
    state.last_batch_count = batch_count;
    state.rounds_since_scan = 0;
    state.candidate_age = 0;
    state.mode = (batch_count > 0) ? L3_DYNAMIC_EAGER : L3_DYNAMIC_COALESCE;
}

// 返回 true 表示本轮应立即进入候选扫描；返回 false 表示继续有限等待。
// signal_changed 只表示 remote_mark 曾发生新事件，不代表必须立刻发送。
// peer idle、本卡没有本地工作、候选积累较大或年龄达到上限时强制发送，
// 以保证不会因为 COALESCE 造成 starvation。
__device__ __forceinline__ bool l3_dynamic_should_scan(
    l3_dynamic_state &state,
    int batch_capacity,
    bool peer_idle,
    bool local_idle,
    bool signal_changed,
    unsigned long long signal_delta)
{
    if (signal_changed)
    {
        // g_bulk_mark_signal is a monotonic count of mark 0->1 events. It is
        // not the correctness source, but it is a useful low-cost burst
        // indicator before the authoritative bitmap scan.
        unsigned long long bounded_delta = signal_delta;
        if (bounded_delta > (unsigned long long)L3_DYNAMIC_MAX_DELAY_ROUNDS)
            bounded_delta = L3_DYNAMIC_MAX_DELAY_ROUNDS;
        if (bounded_delta == 0) bounded_delta = 1;
        int next_age = state.candidate_age + (int)bounded_delta;
        state.candidate_age = (next_age > L3_DYNAMIC_MAX_DELAY_ROUNDS)
                            ? L3_DYNAMIC_MAX_DELAY_ROUNDS : next_age;
    }
    else if (state.candidate_age > 0)
        state.candidate_age++;

    state.rounds_since_scan++;

    const int dense_threshold = (batch_capacity >= 4) ? (batch_capacity / 4) : 1;
    const bool dense_batch = state.last_batch_count >= dense_threshold;
    const bool force_age = state.candidate_age >= L3_DYNAMIC_MAX_DELAY_ROUNDS;
    const bool burst = signal_delta >= (unsigned long long)L3_DYNAMIC_SIGNAL_THRESHOLD;

    if (peer_idle || local_idle || dense_batch || force_age || burst)
    {
        state.mode = L3_DYNAMIC_EAGER;
        if (force_age && !peer_idle && !local_idle && !dense_batch)
            state.forced_flushes++;
        state.rounds_since_scan = 0;
        return true;
    }

    state.mode = L3_DYNAMIC_COALESCE;
    // 没有新事件时仍周期性检查 authoritative remote_mark，不能只依赖 hint。
    if (state.rounds_since_scan >= L3_DYNAMIC_COALESCE_ROUNDS)
    {
        state.rounds_since_scan = 0;
        return true;
    }
    return false;
}
