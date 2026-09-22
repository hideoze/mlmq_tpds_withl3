#pragma once

#include "common.h"
#include "GPU_setup.h"
#include "../cu_vector/cu_vector.cuh"
#include "../cu_vector/cu_multi_vector.cuh"
#include "../cu_vector/cu_batch_vector.cuh"
#include "../cu_delta_queue/cu_delta_queue.cuh"
#include "../cu_heap/bgpq_heap.cuh"

// To implement a global l2_queue, define the following functions in the queue clas
// init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup);
// __device__ read_status read(eletype *node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time);
// __device__ write_status write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time);
// __device__ int get_queue_size();
// __device__ void manager_run(int wid, int lane_id);
// __device__ void update_done(int on_the_fly_num);

template <typename eletype>
class l2_bgpq_queue
{
public:
    BGPQ_Heap<eletype> *bgpq;
    int *smemOffset;
    int *read_done;
    int *write_reserve;
    int batchSize;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup);
    
    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup);

    __device__ init_status device_init(int bid, int wid, int lane_id);

    __device__ read_status read(eletype *node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time);

    __device__ write_status write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time);

    __device__ void show_bgpq();

    __device__ int get_queue_size();

    __device__ int get_available_queue_size();

    __device__ void manager_run(int wid, int lane_id);

    __device__ void update_done(int on_the_fly_num);

    __device__ void update_local_info(int lane_id);

    __host__ __device__ int manage_warp_num();
};

template <typename eletype>
class l2_multi_queue
{
public:
    BGPQ_Heap<eletype> *bgpq;
    int *smemOffset;
    int *read_done;
    int *write_reserve;
    int batchSize;
    // buffer size of one single priority queue
    int buffer_size;
    int write_id;
    int NUM_PQ;

    init_status host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup);

    init_status host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup);

    __device__ init_status device_init(int bid, int wid, int lane_id);

    __device__ read_status read(eletype *node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time);

    __device__ write_status write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();

    __device__ int get_available_queue_size();

    __device__ void manager_run(int wid, int lane_id);

    __device__ void update_done(int on_the_fly_num);

    __device__ void update_local_info(int lane_id);

    __host__ __device__ int manage_warp_num();
};

#include "../src/l2_queue.cu"
