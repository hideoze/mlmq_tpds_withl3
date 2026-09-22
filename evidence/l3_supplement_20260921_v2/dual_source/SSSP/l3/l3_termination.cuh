#pragma once

// 默认 BULK 路径的终止状态。请求 token 与状态分离：token 防止取消后
// 立即重试时把旧 ACK 误认为新请求，state=READY 则表示本卡已经冻结所有
// work/injection producer 和 TX collector，并完成本地最终复核。
// TX ACK 只能在持有批次发布后、下一次提取 mark 前给出；收到 ACK 后
// 必须重新检查权威 mark 与最新 published epoch 的远端 ACK。
enum l3_term_state_code
{
    L3_TERM_ACTIVE = 0,
    L3_TERM_QUIESCING = 1,
    L3_TERM_READY = 2
};

__device__ __forceinline__ bool l3_termination_quiescent(
    bool candidate_empty,
    bool inbox_empty,
    bool outbox_acked,
    bool apply_idle,
    bool peer_idle)
{
    return candidate_empty && inbox_empty && outbox_acked
        && apply_idle && peer_idle;
}
