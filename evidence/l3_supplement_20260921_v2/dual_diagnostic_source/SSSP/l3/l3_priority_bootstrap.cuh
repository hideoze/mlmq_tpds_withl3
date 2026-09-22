#pragma once
#if (L3_RX_PRIORITY_BOOTSTRAP == true)
#if (BULK_ROUND == false || L3_DIRECT_RX == false || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || BULK_EPOCH == true || SEED_BARRIER == true)
#error "priority bootstrap requires default BULK direct receive"
#endif
// No effect on unsupported queue instantiations (DIRECT_RX host gate rejects
// those configurations). Queue storage/reservations are never reset here.
template <typename Q>
__device__ int l3_bootstrap_empty_priority(Q &, int, int) { return -1; }

template <typename E>
__device__ int l3_bootstrap_empty_priority(l2_delta_queue<E> &q, int minimum, int lane)
{
    bool pristine=true;
    for(int i=lane;i<q.bucketNum;i+=WARP_SIZE)
        if(atomicAdd(q.write_reserve+i,0)!=0)pristine=false;
    if(__ballot_sync(FULL_MASK,pristine)!=FULL_MASK)return -1;
    // Whole ring turns preserve physical bucket identity for readers that
    // have already reserved empty ranges. Late lower priorities still clamp
    // into the current bucket by the existing DQ rule; nothing is dropped.
    int position=(minimum/int(q.delta))/q.bucketNum*q.bucketNum;
    int result=-1;
    if(!lane && minimum>=0 && minimum<DIST_MAX && position>0)
        if(atomicCAS(q.first_pos,0,position)==0)result=position;
    __threadfence();
    return __shfl_sync(FULL_MASK,result,0);
}
#endif
