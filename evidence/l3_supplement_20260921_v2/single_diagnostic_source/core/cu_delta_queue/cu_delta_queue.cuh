#include "common.h"
#include <cub/cub.cuh>
#define MEM_BLOCK_SIZE 512

#define DQ_READ_MIN 1

// L3_READ_GATE: dual-GPU read gate for the delta queue.  A warp with no
// pending claim skips the blind batchSize claim when all claimed slots
// already cover the published visibility window of that bucket.  Only the
// wasted global atomic and the counter pass are skipped; claim ownership,
// read accounting and termination semantics are unchanged.  Default off.
#ifndef L3_READ_GATE
#define L3_READ_GATE false
#endif

// Optional diagnostics only; never used to decide a bucket or queue state.
#ifndef DQ_CLAMP_DIAG
// Preserve the measured baseline: r28 disabling this counter did not improve
// W/CAL. Explicit false remains available for controlled diagnostic ablation.
#define DQ_CLAMP_DIAG true
#endif
#if (DQ_CLAMP_DIAG == true)
__device__ extern unsigned long long g_dq_clamp;
#endif

// Queue-supply diagnosis is compile-time isolated.  It records the state seen
// by an ordinary q2 reader before the reservation and distinguishes an empty
// queue from work that is still waiting for manager publication.  The default
// path has no counter or extra queue-size scans.
#ifndef DQ_QUEUE_DIAG
#define DQ_QUEUE_DIAG false
#endif
#if (DQ_QUEUE_DIAG == true)
struct dq_queue_diag_metrics
{
    unsigned long long read_calls;
    unsigned long long read_success;
    unsigned long long read_empty;
    unsigned long long records;
    unsigned long long empty_no_work;
    unsigned long long empty_unpublished;
    unsigned long long empty_published;
    unsigned long long manager_calls;
    unsigned long long manager_pending;
};
__device__ extern dq_queue_diag_metrics g_dq_queue_diag;
#endif

// read/write in a multiple-write-multiple-read way
#ifndef DQ_SPARSE_DIAG
#define DQ_SPARSE_DIAG false
#endif
#ifndef DQ_SPARSE_PHASE
#define DQ_SPARSE_PHASE false
#endif
#if (DQ_SPARSE_PHASE == true && DQ_SPARSE_DIAG == false)
#error "phase observation requires sparse observation"
#endif
#if (DQ_SPARSE_DIAG == true)
static constexpr int DQ_SPARSE_SLOTS = 256;
static constexpr unsigned long long DQ_SPARSE_PERIOD = 1024;
struct dq_sparse_metrics {
    unsigned long long calls, samples, empty, success, records;
    unsigned long long unfinished, local_lag, own_ready;
    unsigned long long unpublished_wait, speculative_wait;
    unsigned long long unreserved_inside, unreserved_outside;
};
__device__ extern dq_sparse_metrics g_dq_sparse[DQ_SPARSE_SLOTS];
#if (DQ_SPARSE_PHASE == true)
struct dq_phase_metrics {
    unsigned long long start, first, last, end, work_calls;
    unsigned long long first_empty, first_unfinished, last_empty, last_unfinished;
};
__device__ extern dq_phase_metrics g_dq_phase[DQ_SPARSE_SLOTS];
#endif
#endif

template <typename eletype>
struct l2_delta_queue
{
// public:

    // Not considered: When read_pos / write_reserve / write_done overflow.
    eletype *data;

    // the number of finished writing operations in each memory block
    int *block_write_done;

    // size: bucketNum
    int *read_pos;
    int *read_ptr;
    int *bucket_read_done;
    int *write_reserve;
    int *last_read_warp;

    int *debug_write_done;

    // total number of read elements done
    int *read_done;

    // indicate base
    int *first_pos;

    int *read_size;
    int *run_begin;

    // total number of batches in each bucket
    int total_size;
    int total_block_size;
    int shmem_offset;

    VALUE_TYPE delta;

    // Maximum number of buckets
    int bucketNum;
    int batchSize;
    // Number of concurrent buckets
    int bucket_max;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        bucketNum = setup.s_BNUM;
        batchSize = setup.s_l2_batch_size;
        bucket_max = setup.s_BUCKET_MAX;

        int all_total_size = max_size / sizeof(eletype);
        int all_total_block_size = all_total_size / MEM_BLOCK_SIZE;
        cudaMalloc(&data, sizeof(eletype) * all_total_size);
        cudaMemset(data, 0, sizeof(eletype) * all_total_size);
        cudaMalloc(&block_write_done, sizeof(int) * all_total_block_size);
        cudaMemset(block_write_done, 0, sizeof(int) * all_total_block_size);

        // Round DOWN within the allocation. Ceil division can advertise a
        // final bucket/block whose tail was never allocated (e.g. INT_MAX bytes).
        total_size = all_total_size / bucketNum / MEM_BLOCK_SIZE * MEM_BLOCK_SIZE;
        total_block_size = total_size / MEM_BLOCK_SIZE;

        cudaMalloc(&read_pos, sizeof(int) * bucketNum);
        cudaMemset(read_pos, 0, sizeof(int) * bucketNum);
        cudaMalloc(&read_ptr, sizeof(int) * bucketNum);
        cudaMemset(read_ptr, 0, sizeof(int) * bucketNum);
        cudaMalloc(&bucket_read_done, sizeof(int) * bucketNum);
        cudaMemset(bucket_read_done, 0, sizeof(int) * bucketNum);
        cudaMalloc(&write_reserve, sizeof(int) * bucketNum);
        cudaMemset(write_reserve, 0, sizeof(int) * bucketNum);
        cudaMalloc(&last_read_warp, sizeof(int) * bucketNum);
        cudaMemset(last_read_warp, 0, sizeof(int) * bucketNum);

        cudaMalloc(&read_done, sizeof(int));
        cudaMemset(read_done, 0, sizeof(int));
        cudaMalloc(&first_pos, sizeof(int));
        cudaMemset(first_pos, 0, sizeof(int));
        cudaMalloc(&run_begin, sizeof(int));
        cudaMemset(run_begin, 0, sizeof(int));
        cudaMalloc(&debug_write_done, sizeof(int));
        cudaMemset(debug_write_done, 0, sizeof(int));

        int read_size_host = batchSize;
        cudaMalloc(&read_size, sizeof(int));
        cudaMemcpy(read_size, &read_size_host, sizeof(int), cudaMemcpyHostToDevice);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);
        delta = setup.s_l2_delta;

        printf("L2_CAPACITY budget=%d record_bytes=%zu buckets=%d per_bucket=%d allocated_records=%d counter_bits=%zu\n",max_size,sizeof(eletype),bucketNum,total_size,all_total_size,8*sizeof(int));
        return INIT_SUCCESS;
    }

    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        bucketNum = setup.s_BNUM;
        batchSize = setup.s_l2_batch_size;
        bucket_max = setup.s_BUCKET_MAX;

        int all_total_size = max_size / sizeof(eletype);
        int all_total_block_size = all_total_size / MEM_BLOCK_SIZE;
        cudaMemset(data, 0, sizeof(eletype) * all_total_size);
        cudaMemset(block_write_done, 0, sizeof(int) * all_total_block_size);

        total_size = all_total_size / bucketNum / MEM_BLOCK_SIZE * MEM_BLOCK_SIZE;
        total_block_size = total_size / MEM_BLOCK_SIZE;

        cudaMemset(read_pos, 0, sizeof(int) * bucketNum);
        cudaMemset(read_ptr, 0, sizeof(int) * bucketNum);
        cudaMemset(bucket_read_done, 0, sizeof(int) * bucketNum);
        cudaMemset(write_reserve, 0, sizeof(int) * bucketNum);
        cudaMemset(last_read_warp, 0, sizeof(int) * bucketNum);

        cudaMemset(read_done, 0, sizeof(int));
        cudaMemset(first_pos, 0, sizeof(int));
        cudaMemset(run_begin, 0, sizeof(int));
        cudaMemset(debug_write_done, 0, sizeof(int));

        int read_size_host = batchSize;
        cudaMemcpy(read_size, &read_size_host, sizeof(int), cudaMemcpyHostToDevice);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);
        delta = setup.s_l2_delta;

        return INIT_SUCCESS;
    }

    __device__ init_status device_init(int bid, int wid, int lane_id)
    {
        extern __shared__ int s[];
        int *local_read_ptr = s + shmem_offset + 2 * bucketNum * wid;

        int *local_read_pos = s + shmem_offset + 2 * bucketNum * WARP_NUM_PER_BLOCK;

        for (int i = 0; i < bucketNum; i++)
            local_read_ptr[i] = -1;

        if (!wid)
        {
            for (int i = 0; i < bucketNum; i++)
                local_read_pos[i] = 0;
        }
        
        return INIT_SUCCESS;
    }

    __host__ __device__ int manage_warp_num()
    {
        return bucketNum;
    }

    __device__ __forceinline__ int get_bucket_addr(int bucket_addr, int dst_bucket_id)
    {
        return bucket_addr + dst_bucket_id * total_size;
    }

    __device__ __forceinline__ int get_block_addr(int block_addr, int dst_bucket_id)
    {
        return block_addr + dst_bucket_id * total_block_size;
    }

#if (DQ_SPARSE_DIAG == true)
    // Called only by the selected reader lane, after an ordinary empty read.
    // Independent loads are NOT a coherent queue snapshot. No scheduling writes.
    __device__ __forceinline__ void sparse_empty_snapshot(
        dq_sparse_metrics &d, int *local_ptr, int *local_pos, int observed_first)
    {
        bool lag = false, ready = false, unpublished = false, speculative = false;
        bool inside = false, outside = false;
        long long writes = 0;
        for (int b = 0; b < bucketNum; ++b) {
            const int w = ((volatile int *)write_reserve)[b];
            const int pub = ((volatile int *)read_pos)[b];
            const int claimed = ((volatile int *)read_ptr)[b];
            const int mine = ((volatile int *)local_ptr)[b];
            const int visible = ((volatile int *)local_pos)[b];
            writes += w;
            if (mine >= 0) {
                lag |= pub > mine && visible <= mine;
                ready |= visible > mine;
                unpublished |= mine >= pub && mine < w;
                speculative |= mine >= w;
            }
            if (pub > claimed) {
                const int offset = (b - observed_first % bucketNum + bucketNum) % bucketNum;
                inside |= offset < bucket_max;
                outside |= offset >= bucket_max;
            }
        }
        d.unfinished += writes > *((volatile int *)read_done);
        d.local_lag += lag; d.own_ready += ready;
        d.unpublished_wait += unpublished; d.speculative_wait += speculative;
        d.unreserved_inside += inside; d.unreserved_outside += outside;
    }
#endif

    __device__ read_status read(eletype* node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        extern __shared__ int s[];
        int *local_read_ptr = s + shmem_offset + 2 * bucketNum * wid;
        int *local_dst_read_ptr = local_read_ptr + bucketNum;
        //int *local_read_pos = s + shmem_offset + WARP_NUM_PER_BLOCK * 2;
        int current_read_size = batchSize;

        int *local_read_pos = s + shmem_offset + 2 * bucketNum * WARP_NUM_PER_BLOCK;

#if (DQ_QUEUE_DIAG == true)
        int diag_queue_size = 0;
        int diag_available = 0;
        if (!lane_id)
        {
            diag_queue_size = get_queue_size();
            diag_available = get_available_queue_size();
        }
        diag_queue_size = __shfl_sync(FULL_MASK, diag_queue_size, 0);
        diag_available = __shfl_sync(FULL_MASK, diag_available, 0);
#endif

        read_num = 0;

        int first_pos_old = *first_pos;
        int i = 0;
        while (i < bucket_max && read_num < DQ_READ_MIN)
        {
            int vec_id = (first_pos_old + i) % bucketNum;

            // Read gate (L3_READ_GATE, dual-GPU builds): when this warp holds
            // no pending claim on the bucket and every already-claimed slot
            // covers the published visibility window (read_ptr >= visible
            // boundary), a fresh blind claim of batchSize slots would start
            // beyond any visible item and only burn a global atomic plus a
            // wasted pass over the counters.  Skip straight to the next
            // bucket; the manager warps keep refreshing the visibility
            // boundary in shared memory, so the gate reopens by itself the
            // moment new work is published.  Semantics (claim ownership,
            // bucket_read_done accounting, last_read_warp, first_pos
            // advance) are untouched: we only decline to make a claim that
            // the existing checks would immediately discard.
#if (L3_READ_GATE == true)
            if (local_read_ptr[vec_id] == -1
                && read_ptr[vec_id] >= local_read_pos[vec_id])
            {
                i++;
                continue;
            }
#endif

            if (!lane_id && local_read_ptr[vec_id] == -1)
            {
                local_read_ptr[vec_id] = atomicAdd(&read_ptr[vec_id], current_read_size);
                assert(local_read_ptr[vec_id] >= 0 && (long long)local_read_ptr[vec_id]+current_read_size < INT_MAX);
                local_dst_read_ptr[vec_id] = local_read_ptr[vec_id] + current_read_size;
            }

            __syncwarp();

            int old_read_pos = local_read_pos[vec_id];//local_read_pos[vec_id];

            int bucket_read_num = 0;
            if (local_read_ptr[vec_id] < old_read_pos)
            {
                bucket_read_num =  mlq_min(old_read_pos, local_dst_read_ptr[vec_id]) - local_read_ptr[vec_id];
                if (bucket_read_num > current_read_size) bucket_read_num = current_read_size;
            }

            if (bucket_read_num)
            {
                for (int cpy_iter = lane_id; cpy_iter < bucket_read_num; cpy_iter += WARP_SIZE)
                {
                    node_in[read_num + cpy_iter] = data[get_bucket_addr((local_read_ptr[vec_id] + cpy_iter) % total_size, vec_id)];
                }
                read_num += bucket_read_num;

                __syncwarp();

                if (!lane_id)
                {
                    local_read_ptr[vec_id] += bucket_read_num;
                    if (local_read_ptr[vec_id] == write_reserve[vec_id])
                        last_read_warp[vec_id] = bid * WARP_NUM_PER_BLOCK + wid;

                    atomicAdd(&bucket_read_done[vec_id], bucket_read_num);

                    if (local_read_ptr[vec_id] == local_dst_read_ptr[vec_id])
                    {
                        // assume l2_batch_size <= MEM_BLOCK_SIZE
                        if (local_dst_read_ptr[vec_id] % MEM_BLOCK_SIZE == 0)
                            block_write_done[get_block_addr(((local_dst_read_ptr[vec_id] - 1) / MEM_BLOCK_SIZE) % total_block_size, vec_id)] = 0;
                        local_read_ptr[vec_id] = -1;
                    }
                }
            }
            else
            {
                i++;
            }
            
        }

        if (last_read_warp[first_pos_old % bucketNum] == bid * WARP_NUM_PER_BLOCK + wid && (i > 0 || read_num == 0))
        {
            int first_size = get_bucket_size(first_pos_old % bucketNum);
            int queue_size = get_total_size();
            if (!lane_id && first_size == 0 && queue_size > 0)
            {
                int res = atomicCAS(first_pos, first_pos_old, first_pos_old + 1);
            }
        }

        __syncwarp();
#if (DQ_SPARSE_DIAG == true)
        // One reader per block (rotating warp identity across blocks), one
        // independent writer per slot. Fixed sampling; no atomics or clock64.
        if (!lane_id && wid == bid % WARP_NUM_PER_BLOCK) {
            assert(bid < DQ_SPARSE_SLOTS);
            dq_sparse_metrics &d = g_dq_sparse[bid];
            const unsigned long long call = ++d.calls;
            if ((call % DQ_SPARSE_PERIOD) == 0) {
                ++d.samples;
                if (read_num > 0) { ++d.success; d.records += read_num; }
                else {
                    ++d.empty;
                    sparse_empty_snapshot(d, local_read_ptr, local_read_pos, first_pos_old);
                }
            }
        }
#endif
#if (DQ_QUEUE_DIAG == true)
        if (!lane_id)
        {
            atomicAdd(&g_dq_queue_diag.read_calls, 1ull);
            if (read_num > 0)
            {
                atomicAdd(&g_dq_queue_diag.read_success, 1ull);
                atomicAdd(&g_dq_queue_diag.records,
                         static_cast<unsigned long long>(read_num));
            }
            else
            {
                atomicAdd(&g_dq_queue_diag.read_empty, 1ull);
                if (diag_queue_size <= 0)
                    atomicAdd(&g_dq_queue_diag.empty_no_work, 1ull);
                else if (diag_available <= 0)
                    atomicAdd(&g_dq_queue_diag.empty_unpublished, 1ull);
                else
                    atomicAdd(&g_dq_queue_diag.empty_published, 1ull);
            }
        }
#endif
        if (read_num == 0)
            return READ_EMPTY;
        else
            return READ_SUCCESS;
    }

    __device__ write_status write(eletype* node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        volatile int first_pos_old = *first_pos;
        VALUE_TYPE base = first_pos_old * delta;
        for (int i = 0; i < write_num; i+=WARP_SIZE)
        {
            int out_pos = i + lane_id;

            int dst_bucket_id = -1;
            int write_pos = 0;
            
            if (out_pos < write_num)
            {
                dst_bucket_id = (int)((node_out[out_pos].get_data() - base) / delta);
                if (dst_bucket_id < 0) {
                    dst_bucket_id = 0;
#if (DQ_CLAMP_DIAG == true)
                    atomicAdd(&g_dq_clamp, 1ull);
#endif
                }
                if (dst_bucket_id >= bucketNum) dst_bucket_id = bucketNum - 1;
                dst_bucket_id = (first_pos_old + dst_bucket_id) % bucketNum;
            }

            unsigned active_mask = __ballot_sync(FULL_MASK, dst_bucket_id != -1);
            unsigned write_mask = __match_any_sync(active_mask, dst_bucket_id);
            int leader_lane = find_ms_bit(write_mask);
            write_pos = count_bit(set_bits(write_mask, 0, lane_id, 32));
            int write_bucket_num = count_bit(write_mask);

            int current_reserve;
            if (out_pos < write_num && lane_id == leader_lane)
            {
                current_reserve = atomicAdd(&write_reserve[dst_bucket_id], write_bucket_num);
                assert(current_reserve >= 0 && (long long)current_reserve + write_bucket_num <= total_size);
                    //printf("dst_bucket_id %d after current_reserve %d write_reserve[dst_bucket_id] %d write_bucket_num %d data %d %d\n", dst_bucket_id,
                    //current_reserve, write_reserve[dst_bucket_id], write_bucket_num, node_out[out_pos].id, node_out[out_pos].dist);
            }
            __syncwarp();
            current_reserve = __shfl_sync(write_mask, current_reserve, leader_lane);

            int global_idx = get_bucket_addr((current_reserve + write_pos) % total_size, dst_bucket_id);
            int block_idx = global_idx / MEM_BLOCK_SIZE;
            unsigned block_mask = __match_any_sync(write_mask, block_idx);
            int block_leader_lane = find_ms_bit(block_mask);
            int block_write_num = count_bit(block_mask);

            if (out_pos < write_num)
                data[global_idx] = node_out[out_pos];

            // Every producing lane orders its payload stores before the leader
            // publishes completion to manager warps on other SMs. A fence after
            // block_write_done is too late to establish that publication order.
            __threadfence();
            __syncwarp();
            if (out_pos < write_num && lane_id == block_leader_lane)
            {
                atomicAdd(&block_write_done[block_idx], block_write_num);
                atomicAdd(debug_write_done, block_write_num);
            }
            __syncwarp();
        }

        return WRITE_SUCCESS;
    }

    // update read_pos in bucket[manager_id]
    __device__ void manager_run(int manager_id, int lane_id)
    {
        int old_write_reserve = write_reserve[manager_id];
#if (DQ_QUEUE_DIAG == true)
        if (!lane_id)
        {
            atomicAdd(&g_dq_queue_diag.manager_calls, 1ull);
            if (old_write_reserve > read_pos[manager_id])
                atomicAdd(&g_dq_queue_diag.manager_pending, 1ull);
        }
#endif
#if (BULK_DIAG == true)
        __device__ extern unsigned long long g_bulk_dq_prints;
        if (lane_id == 0 && old_write_reserve > read_pos[manager_id])
        {
            unsigned long long p = atomicAdd(&g_bulk_dq_prints, 1ull);
            if (p < 32)
                printf("BULK_DQ id=%d first=%d rpos=%d rptr=%d bdone=%d wres=%d lread=%d read_done=%d q=%d\\n",
                       manager_id, *first_pos, read_pos[manager_id], read_ptr[manager_id],
                       bucket_read_done[manager_id], write_reserve[manager_id],
                       last_read_warp[manager_id], *read_done, get_queue_size());
        }
#endif
        if (old_write_reserve > read_pos[manager_id])
        {
            int start_block_idx = read_pos[manager_id] / MEM_BLOCK_SIZE;
            int end_block_idx = old_write_reserve / MEM_BLOCK_SIZE;

            int nofull_block_lane = 0;
            int nofull_block_write_done = 0;

            int i = start_block_idx;
            for (; i < end_block_idx + 1; i += WARP_SIZE)
            {
                int block_idx = i + lane_id;
                int current_write_done = block_write_done[get_block_addr(block_idx % total_block_size, manager_id)];
                bool block_valid = i < end_block_idx;
                bool block_nofull = true;

                if (block_valid)
                    block_nofull = current_write_done != MEM_BLOCK_SIZE;

                unsigned nofull_mask = __ballot_sync(FULL_MASK, block_nofull);
                __syncwarp();
                if (nofull_mask != 0)
                {
                    nofull_block_lane = find_nth_bit(nofull_mask, 0, 1);
                    nofull_block_write_done = __shfl_sync(FULL_MASK, current_write_done, nofull_block_lane);
                    break;
                }
            }
            int nofull_block_idx = i + nofull_block_lane;

            int bucket_addr = nofull_block_idx * MEM_BLOCK_SIZE;
            // If in the last block must 
            if (nofull_block_idx == end_block_idx)
            {
                if (nofull_block_write_done + bucket_addr == old_write_reserve)
                {
                    if (read_pos[manager_id] < old_write_reserve)
                        read_pos[manager_id] = old_write_reserve;
                }
                else
                {
                    if (read_pos[manager_id] < bucket_addr)
                        read_pos[manager_id] = bucket_addr;
                }
            }
            else
            {
                if (read_pos[manager_id] < bucket_addr)
                    read_pos[manager_id] = bucket_addr;
            }
            __syncwarp();
            __threadfence();
        }
    }

    // use read_ptr
    __device__ __forceinline__ int get_bucket_size(int dst_bucket_id)
    {
        return write_reserve[dst_bucket_id] - bucket_read_done[dst_bucket_id];
    }

    __device__ int get_total_size()
    {
        int total_read_size = 0;
        for (int i = 0; i < bucketNum; i++)
        {
            total_read_size += write_reserve[i] - bucket_read_done[i];
        }
        return total_read_size;
    }

    // use read_done
    __device__ __forceinline__ int get_queue_size()
    {
        int total_read_size = 0;
        for (int i = 0; i < bucketNum; i++)
        {
            total_read_size += write_reserve[i];
        }
        // volatile read_done: 防编译器 LICM 缓存，终止检测需读到 work 的 update_done
        return total_read_size - *((volatile int *)read_done);
    }

    // Published-but-unreserved upper bound for all buckets.  The bucket
    // selector may still make this an upper bound rather than an exact
    // immediately readable count.
    __device__ __forceinline__ int get_available_queue_size()
    {
        int total_available = 0;
        for (int i = 0; i < bucketNum; i++)
        {
            const int available = read_pos[i] - read_ptr[i];
            if (available > 0)
                total_available += available;
        }
        return total_available;
    }

    __device__ void update_done(int on_the_fly_num)
    {
        atomicAdd(read_done, on_the_fly_num);
    }

    __device__ void update_local_info(int lane_id)
    {
        extern __shared__ int s[];

        int *local_read_pos = s + shmem_offset + 2 * bucketNum * WARP_NUM_PER_BLOCK;

        for (int i = lane_id; i < bucketNum; i += WARP_SIZE)
        {
            local_read_pos[i] = read_pos[i];
        }

        __syncwarp();

    }

    int get_shm_size()
    {
        return 2 * bucketNum * WARP_NUM_PER_BLOCK * sizeof(int) + bucketNum * sizeof(int);
    }

};

#undef DQ_READ_MIN
