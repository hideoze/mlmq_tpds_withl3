#pragma once
#if (L3_RX_LAG_DIAG == true)
#if (L3_PROGRESS_DIAG == false || L3_DIRECT_RX == false)
#error "RX lag diagnosis requires progress and direct RX"
#endif
template <typename Q>
__device__ void l3_rx_base_snapshot(Q &, long long &base, int &delta, int &buckets) {
    base=0; delta=1; buckets=1;
}
template <typename E>
__device__ void l3_rx_base_snapshot(l2_delta_queue<E> &q, long long &base, int &delta, int &buckets) {
    delta=int(q.delta); buckets=q.bucketNum;
    base=static_cast<long long>(atomicAdd(q.first_pos,0))*delta;
}
#endif
