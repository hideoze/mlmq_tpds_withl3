#pragma once

#include "GPU_setup.h"
#include "graph_info.h"
#include "adaptive.h"
#include "warp_primitive.h"
#include "common.h"
#include "l1_queue.cuh"
#include "l2_queue.cuh"

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
struct ml_queue
{
    // l1 queue is shared by one warp
    // access with q1[local_wid] in shared memory

    // l2 queue is the global queue
    l2_queue_type q2;
    
    // store meta data of the queues of each warp
    mlmq_mdata mdata;

    // a lock indicating the queue initiation is done and SSSP run can begin
    int *run_begin;

    // minimum priority
    eletype init_limits;

    // minimum number of read elements
    int min_read_gra;

    // allocate memory of q0, q1 and q2 on device
    init_status init_host(int max_size, eletype init_limits_in, mlmq_setup setup);

    init_status reinit_host(int max_size, eletype init_limits_in, mlmq_setup setup);


    // allocate shared memory
    __device__ init_status init_device(int bid, int wid, int lane_id, mlmq_setup setup);

    // read primitive
    // on_the_fly_num: number of elements read from L2 queue
    __device__ read_status read(eletype *node_in, int &read_num, int &on_the_fly_num, int bid, int wid, int lane_id, unsigned *debug_time);

    // write primitive
    __device__ write_status write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time);

    __device__ write_status write_through(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time);

    // Caller retains its original L2 in-flight credit until every spill is
    // published. Uses the existing warp-private L1 scratch, not a new queue.
    __device__ int spill_local_to_l2(int bid, int wid, int lane_id, unsigned *debug_time);

    // get shared memory size for buffer
    int get_shm_size();

    // get size of L2 queue
    __device__ int get_global_queue_size();

    // get published but not yet reserved L2 work when the queue exposes it
    __device__ int get_available_queue_size();

    // get size of L1 queue
    __device__ int get_local_queue_size(int wid);

    // update read_done information
    __device__ void update_done(int on_the_fly_num);

    // warp function for L2 queue management
    __device__ void l2_manager(int wid, int lane_id);

    // update local information for each work thread block
    __device__ void update_local_info(int lane_id);

    // number of manage warp of each work thread block
    __host__ __device__ int manage_warp_num();
};

#include "../src/ml_queue.cu"
