#pragma once
#include "l3_admission_budget.h"
#if (L3_ADMISSION_BUDGET == true)
#if (BULK_ROUND == false || L3_DIRECT_RX == false || L3_TERM_HANDSHAKE == false || L3_WORKER_RECOVERY == true || L3_RX_PRIORITY_BOOTSTRAP == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || BULK_EPOCH == true || SEED_BARRIER == true || GHOST_DEPTH > 0)
#error "admission budget requires unmodified default BULK ownership"
#endif
// Per-device symbol: advisory busy, distance base, local horizon, deferred batches.
// These words never participate in correctness or termination predicates.
__device__ int g_l3_admission[4]={0,0,DIST_MAX,0};
template <typename Q>
__device__ void l3_admission_publish(Q &,int *,int *,int,unsigned long long &) {}
template <typename E>
__device__ void l3_admission_publish(l2_delta_queue<E> &q,int *own,int *peer,int busy,unsigned long long &last) {
    const auto now=clock64();
    if(now-last<25000ull)return;
    last=now;
    long long base=static_cast<long long>(atomicAdd(q.first_pos,0))*int(q.delta);
    atomicExch(own+1,int(min(base,static_cast<long long>(DIST_MAX))));
    __threadfence_system();
    atomicExch(own,busy!=0);
    int horizon=DIST_MAX;
    if(peer && atomicAdd(peer,0)!=0) {
        long long ceiling=static_cast<long long>(atomicAdd(peer+1,0))+static_cast<long long>(q.bucketNum)*int(q.delta);
        horizon=int(min(ceiling,static_cast<long long>(DIST_MAX)));
    }
    atomicExch(g_l3_admission+2,horizon);
}
#endif
