#pragma once
#include "cu_vector.cuh"

#define NUM_MV 2

template <typename eletype>
struct l2_multi_vector_queue
{
    l2_vector_queue<eletype> mvector[NUM_MV];
    int write_id;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        int vector_size = max_size / NUM_MV;
        for (int vid = 0; vid < NUM_MV; vid++)
        {
            mvector[vid].host_init(vector_size, mdata, init_limits, setup);
        }
    }

    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
    {
        int vector_size = max_size / NUM_MV;
        for (int vid = 0; vid < NUM_MV; vid++)
        {
            mvector[vid].host_reinit(vector_size, mdata, init_limits, setup);
        }
    }

    __device__ init_status device_init(int bid, int wid, int lane_id)
    {
        int qid = bid % NUM_MV;

        mvector[qid].device_init(bid, wid, lane_id);

        write_id = 0;
        __syncwarp();

        return INIT_SUCCESS;
    }

    __host__ __device__ int manage_warp_num()
    {
        return NUM_MV;
    }

    __device__ read_status read(eletype* node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        int qid = bid % NUM_MV;
        return mvector[qid].read(node_in, read_num, bid, wid, lane_id, debug_time);
    }

    __device__ write_status write(eletype* node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
    {
        mvector[write_id].write(node_out, write_num, bid, wid, lane_id, debug_time);
        write_id = (write_id + 1) % NUM_MV;

        __syncwarp();

        return WRITE_SUCCESS;
    }

    __device__ void manager_run(int wid, int lane_id)
    {
        mvector[wid].manager_run(wid, lane_id);
    }

    __device__ int get_queue_size()
    {
        int qsize = 0;
        for (int qid = 0; qid < NUM_MV; qid++)
        {
            qsize += mvector[qid].get_queue_size();
        }

        return qsize;
    }

    __device__ int get_available_queue_size()
    {
        int qsize = 0;
        for (int qid = 0; qid < NUM_MV; qid++)
            qsize += mvector[qid].get_available_queue_size();
        return qsize;
    }

    __device__ void update_done(int on_the_fly_num)
    {
        int qid = blockIdx.x % NUM_MV;
        mvector[qid].update_done(on_the_fly_num);
        //atomicAdd(read_done, on_the_fly_num);
    }

    __device__ void update_local_info(int lane_id)
    {
        int qid = blockIdx.x % NUM_MV;
        mvector[qid].update_local_info(lane_id);
    }

    int get_shm_size()
    {
        return 2 * WARP_NUM_PER_BLOCK * sizeof(int) + sizeof(int);
    }
};
