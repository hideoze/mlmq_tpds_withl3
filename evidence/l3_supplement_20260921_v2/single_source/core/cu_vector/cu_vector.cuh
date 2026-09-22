#pragma once
#include "common.h"
#include <cub/cub.cuh>
#define MEM_BLOCK_SIZE 256

// read/write in a multiple-write-multiple-read way
template <typename eletype>
struct l2_vector_queue
{
    // Not considered: When read_pos / write_reserve / write_done overflow.
    eletype *data;

    // the number of finished writing operations in each memory block
    int *block_write_done;

    int *read_pos;
    int *read_ptr;
    int *read_done;
    int *write_reserve;

    // The number of elements read in each time
    int *read_size;

    // total number of batches
    int total_size;
    int total_block_size;
    int shmem_offset;
    int batchSize;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        batchSize = setup.s_l2_batch_size;

        total_size = max_size / sizeof(eletype) / MEM_BLOCK_SIZE * MEM_BLOCK_SIZE;
        cudaMalloc(&data, sizeof(eletype) * total_size);

        total_block_size = total_size / MEM_BLOCK_SIZE;
        cudaMalloc(&block_write_done, sizeof(int) * total_block_size);
        cudaMemset(block_write_done, 0, sizeof(int) * total_block_size);

        cudaMalloc(&read_pos, sizeof(int));
        cudaMemset(read_pos, 0, sizeof(int));
        cudaMalloc(&read_ptr, sizeof(int));
        cudaMemset(read_ptr, 0, sizeof(int));
        cudaMalloc(&read_done, sizeof(int));
        cudaMemset(read_done, 0, sizeof(int));
        cudaMalloc(&write_reserve, sizeof(int));
        cudaMemset(write_reserve, 0, sizeof(int));

        int read_size_host = batchSize;
        cudaMalloc(&read_size, sizeof(int));
        cudaMemcpy(read_size, &read_size_host, sizeof(int), cudaMemcpyHostToDevice);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);

        return INIT_SUCCESS;
    }

    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        batchSize = setup.s_l2_batch_size;

        total_size = max_size / sizeof(eletype) / MEM_BLOCK_SIZE * MEM_BLOCK_SIZE;

        total_block_size = total_size / MEM_BLOCK_SIZE;
        cudaMemset(block_write_done, 0, sizeof(int) * total_block_size);

        cudaMemset(read_pos, 0, sizeof(int));
        cudaMemset(read_ptr, 0, sizeof(int));
        cudaMemset(read_done, 0, sizeof(int));
        cudaMemset(write_reserve, 0, sizeof(int));

        int read_size_host = batchSize;
        cudaMemcpy(read_size, &read_size_host, sizeof(int), cudaMemcpyHostToDevice);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);

        return INIT_SUCCESS;
    }

    __device__ init_status device_init(int bid, int wid, int lane_id)
    {
        extern __shared__ int s[];
        int *local_read_ptr = s + shmem_offset + 2 * wid;
        //int *local_dst_read_ptr = local_read_ptr + 1;
        *local_read_ptr = -1;
        
        int *local_read_pos = s + shmem_offset + WARP_NUM_PER_BLOCK * 2;
        *local_read_pos = 0;

        return INIT_SUCCESS;
    }

    __host__ __device__ int manage_warp_num()
    {
        return 1;
    }

    __device__ read_status read(eletype* node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        extern __shared__ int s[];
        int *local_read_ptr = s + shmem_offset + 2 * wid;
        int *local_dst_read_ptr = local_read_ptr + 1;
        int *local_read_pos = s + shmem_offset + WARP_NUM_PER_BLOCK * 2;
        int current_read_size = batchSize;

        if (!lane_id && *local_read_ptr == -1)
        {
            *local_read_ptr = atomicAdd(read_ptr, current_read_size);
            *local_dst_read_ptr = *local_read_ptr + current_read_size;
        }
        __syncwarp();
        
        int old_read_pos = *local_read_pos;

        read_num = 0;
        if (*local_read_ptr < old_read_pos)
        {
            read_num += mlq_min(old_read_pos, *local_dst_read_ptr) - *local_read_ptr;
            if (read_num > current_read_size) read_num = current_read_size;
        }

        if (read_num == 0)
        {
            return READ_EMPTY;
        }

        __syncwarp();

        for (int cpy_iter = lane_id; cpy_iter < read_num; cpy_iter += WARP_SIZE)
        {
            node_in[cpy_iter] = data[(*local_read_ptr + cpy_iter) % total_size];
        }

        __syncwarp();
        if (!lane_id)
        {
            *local_read_ptr += read_num;
            if (*local_read_ptr == *local_dst_read_ptr)
            {
                // assume l2_batch_size <= MEM_BLOCK_SIZE
                if (*local_dst_read_ptr % MEM_BLOCK_SIZE == 0)
                    block_write_done[((*local_dst_read_ptr - 1) / MEM_BLOCK_SIZE) % total_block_size] = 0;
                // for (int block_idx = *local_read_ptr / MEM_BLOCK_SIZE; block_idx < *local_dst_read_ptr / MEM_BLOCK_SIZE; block_idx++)
                //     block_write_done[block_idx % total_block_size] = 0;
                *local_read_ptr = -1;
            }
        }

        __syncwarp();
        return READ_SUCCESS;
    }

    __device__ write_status write(eletype* node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        int current_reserve;
        if (!lane_id)
        {
            current_reserve = atomicAdd(write_reserve, write_num);
        }
        __syncwarp();

        current_reserve = __shfl_sync(0xffffffff, current_reserve, 0);

        for (int cpy_iter = 0; cpy_iter < write_num; cpy_iter+=WARP_SIZE)
        {
            int write_idx = cpy_iter + lane_id;
            bool write_valid = write_idx < write_num;
            int global_idx = (current_reserve + write_idx) % total_size;
            int block_idx = global_idx / MEM_BLOCK_SIZE;
            
            unsigned write_mask = __ballot_sync(FULL_MASK, write_valid);
            unsigned block_mask = __match_any_sync(write_mask, block_idx);
            int leader_lane = find_ms_bit(block_mask);
            int block_write_num = count_bit(block_mask);

            if (write_valid)
                data[global_idx] = node_out[write_idx];
            __threadfence();

            if (lane_id == leader_lane)
            {
                atomicAdd(&block_write_done[block_idx % total_block_size], block_write_num);
            }
            __syncwarp();

        }

        return WRITE_SUCCESS;
    }

    __device__ void manager_run(int wid, int lane_id)
    {
        int old_write_reserve = *write_reserve;
        if (old_write_reserve > *read_pos)
        {
            int start_block_idx = *read_pos / MEM_BLOCK_SIZE;
            int end_block_idx = old_write_reserve / MEM_BLOCK_SIZE;

            int nofull_block_lane = 0;
            int nofull_block_write_done = 0;

            int i = start_block_idx;
            for (; i < end_block_idx + 1; i += WARP_SIZE)
            {
                int block_idx = i + lane_id;
                int current_write_done = block_write_done[block_idx % total_block_size];
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

            int global_addr = nofull_block_idx * MEM_BLOCK_SIZE;
            // If in the last block must 
            if (nofull_block_idx == end_block_idx)
            {
                if (nofull_block_write_done + global_addr == old_write_reserve)
                {
                    if (*read_pos < old_write_reserve)
                        *read_pos = old_write_reserve;
                }
                else
                {
                    if (*read_pos < global_addr)
                        *read_pos = global_addr;
                }
            }
            else
            {
                if (*read_pos < global_addr)
                    *read_pos = global_addr;
            }
            __syncwarp();
            __threadfence();
        }
    }

    __device__ int get_queue_size()
    {
        return *write_reserve - *read_done;
    }

    // Published-but-unreserved work.  This deliberately differs from
    // get_queue_size(), which includes reservations already held by workers.
    __device__ int get_available_queue_size()
    {
        const int available = *read_pos - *read_ptr;
        return available > 0 ? available : 0;
    }

    __device__ void update_done(int on_the_fly_num)
    {
        atomicAdd(read_done, on_the_fly_num);
    }

    __device__ void update_local_info(int lane_id)
    {
        extern __shared__ int s[];
        int *local_read_pos = s + shmem_offset + 2 * WARP_NUM_PER_BLOCK;
        *local_read_pos = *read_pos;
        __threadfence_block();
    }

    int get_shm_size()
    {
        return 2 * WARP_NUM_PER_BLOCK * sizeof(int) + sizeof(int);
    }

};
