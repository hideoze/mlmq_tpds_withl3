#pragma once
#include "common.h"

// read/write in a batch-wise way
template <typename eletype>
class l2_batch_vector_queue
{
public:

    // Not considered: When read_pos / write_reserve / write_done overflow.
    eletype *data;
    // array for data size of each batch, also used for synchronization
    // 0: not finished, 0 < data_size <= batchSize: finished writing
    int *data_size;
    int *read_pos;
    int *read_done;

    int *write_reserve;
    int *write_done;

    // read_reserve: a integer in shared memory, reserve the last read position
    // located at shmem_offset in the dynamic shared memory
    // 0: not reserved, -1: reserved but not finished, > 0: finished
    int shmem_offset;

    // total number of batches
    int total_batches;
    int batchSize;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        batchSize = setup.s_l2_batch_size;

        total_batches = max_size / sizeof(eletype) / batchSize;
        cudaMalloc(&data, sizeof(eletype) * total_batches * batchSize);
        cudaMalloc(&data_size, sizeof(int) * total_batches);
        cudaMemset(data_size, 0, sizeof(int) * total_batches);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);

        cudaMalloc(&read_pos, sizeof(int));
        cudaMemset(read_pos, 0, sizeof(int));
        cudaMalloc(&read_done, sizeof(int));
        cudaMemset(read_done, 0, sizeof(int));
        cudaMalloc(&write_reserve, sizeof(int));
        cudaMemset(write_reserve, 0, sizeof(int));
        cudaMalloc(&write_done, sizeof(int));
        cudaMemset(write_done, 0, sizeof(int));

        return INIT_SUCCESS;
    }

    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        batchSize = setup.s_l2_batch_size;

        total_batches = max_size / sizeof(eletype) / batchSize;
        cudaMemset(data_size, 0, sizeof(int) * total_batches);

        shmem_offset = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK / sizeof(int);

        cudaMemset(read_pos, 0, sizeof(int));
        cudaMemset(read_done, 0, sizeof(int));
        cudaMemset(write_reserve, 0, sizeof(int));
        cudaMemset(write_done, 0, sizeof(int));

        return INIT_SUCCESS;
    }

    __device__ init_status device_init(int bid, int wid, int lane_id)
    {
        extern __shared__ int s[];
        int *read_reserve = s + shmem_offset + wid;
        *read_reserve = -1;

        return INIT_SUCCESS;
    }

    __host__ __device__ int manage_warp_num()
    {
        return 0;
    }

    __device__ read_status read(eletype* node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        extern __shared__ int s[];
        int *read_reserve = s + shmem_offset + wid;

        if (!lane_id && *read_reserve == -1)
        {
            *read_reserve = atomicAdd(read_pos, 1);
        }
        __syncwarp();

        if (data_size[(*read_reserve) % total_batches] == 0)
        {
            __syncwarp();
            return READ_EMPTY;
        }

        read_num = data_size[(*read_reserve) % total_batches];

        for (int cpy_iter = lane_id; cpy_iter < read_num; cpy_iter+= WARP_SIZE)
        {
            node_in[cpy_iter] = data[((*read_reserve) % total_batches) * batchSize + cpy_iter];
        }
        // coop_mem_cpy<eletype>(node_in, data + ((*read_reserve) % total_batches) * batchSize, read_num, lane_id);

        __threadfence();

        __syncwarp();

        if (!lane_id)
        {
            data_size[(*read_reserve) % total_batches] = 0;
            *read_reserve = -1;
        }

        __syncwarp();

        return READ_SUCCESS;
    }

    __device__ write_status write(eletype* node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        int write_batches = (write_num + batchSize - 1) / batchSize;
        int current_reserve;
        if (!lane_id)
        {
            current_reserve = atomicAdd(write_reserve, write_batches);
            atomicAdd(write_done, write_num);
        }
        __syncwarp();

        current_reserve = __shfl_sync(0xffffffff, current_reserve, 0);

        while (data_size[(current_reserve + write_batches - 1) % total_batches] != 0) {}

        __syncwarp();

        //coop_mem_cpy<eletype>(data + (current_reserve % total_batches) * batchSize, node_out, write_num, lane_id);
        for (int cpy_iter = lane_id; cpy_iter < write_num; cpy_iter += WARP_SIZE)
        {
            data[(current_reserve % total_batches) * batchSize + cpy_iter] = node_out[cpy_iter];
        }
        __threadfence();
        __syncwarp();

        if (!lane_id)
        {
            for (int i = current_reserve; i < current_reserve + write_batches - 1; i++)
            {
                data_size[i % total_batches] = batchSize;
            }
            data_size[(current_reserve + write_batches - 1) % total_batches] = write_num - batchSize * (write_batches - 1);
        }

        __syncwarp();
    }

    __device__ void manager_run(int wid, int lane_id)
    {
    }

    __device__ int get_queue_size()
    {
        return *write_done - *read_done;
    }

    // No separate published cursor is exposed by this queue.  Keep the
    // legacy credit metric as a compatibility fallback.
    __device__ int get_available_queue_size()
    {
        const int available = *write_done - *read_done;
        return available > 0 ? available : 0;
    }

    __device__ void update_done(int on_the_fly_num)
    {
        atomicAdd(read_done, on_the_fly_num);
    }

    __device__ void update_local_info(int lane_id)
    {
    }

    int get_shm_size()
    {
        return WARP_NUM_PER_BLOCK * sizeof(int);
    }

};
