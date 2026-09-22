#pragma once

// CP1c：低扰动、有界的 L3 状态事件记录。
//
// 事件环只记录协议边界，不记录每个候选/每轮扫描，因此即使诊断构建运行
// 很久也不会产生无界 device printf 或 host 缓冲。head 单调递增，槽位按
// head % L3_EVENT_RING_CAP 覆盖；overflow 记录覆盖发生次数。
enum l3_event_code
{
    L3_EVENT_SCAN_COALESCED = 1,
    L3_EVENT_PACK_BEGIN = 2,
    L3_EVENT_TX_PUBLISHED = 3,
    L3_EVENT_TX_RETRY = 4,
    L3_EVENT_RX_CLAIM = 5,
    L3_EVENT_RX_APPLY = 6,
    L3_EVENT_TERM_REQUEST = 7,
    L3_EVENT_TERM_CANCEL = 8,
    L3_EVENT_TERM_READY = 9,
    L3_EVENT_TERM_EXIT = 10
};

struct l3_event_entry
{
    unsigned long long tick;
    int code;
    int value0;
    int value1;
    int value2;
    int value3;
};

#if (L3_EVENT_RING == true)
extern __device__ l3_event_entry g_l3_event_ring[L3_EVENT_RING_CAP];
extern __device__ unsigned int g_l3_event_head;
extern __device__ unsigned int g_l3_event_overflow;

__device__ __forceinline__ void l3_event_push(
    int code, int value0, int value1, int value2, int value3)
{
    unsigned int seq = atomicAdd(&g_l3_event_head, 1u);
    if (seq >= L3_EVENT_RING_CAP)
        atomicAdd(&g_l3_event_overflow, 1u);
    l3_event_entry &entry = g_l3_event_ring[seq % L3_EVENT_RING_CAP];
    entry.tick = clock64();
    entry.code = code;
    entry.value0 = value0;
    entry.value1 = value1;
    entry.value2 = value2;
    entry.value3 = value3;
}
#else
__device__ __forceinline__ void l3_event_push(
    int, int, int, int, int)
{
}
#endif
