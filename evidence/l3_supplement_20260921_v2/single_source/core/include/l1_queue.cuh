#pragma once

#include "common.h"
#include "GPU_setup.h"
#include "warp_primitive.h"

// template <typename eletype>
// class l1_queue
// {
// public:
//     virtual __device__ init_status init(void *shmem, int warp_id, int lane_id, mlmq_mdata &mdata, eletype init_limits_in) = 0;

//     virtual __device__ read_status read(eletype *node_in, int &read_num, int lane_id) = 0;

//     virtual __device__ write_status write(eletype *node_out, int &write_num, int lane_id) = 0;
// };

// To implement a warp-level local l1_queue, define the following functions in the queue class
// __device__ init_status init(int wid, int lane_id);
// __device__ read_status read(eletype *node_in, int &read_num, int lane_id);
// __device__ write_status write(eletype *node_out, int &write_num, int lane_id);
// __device__ int get_queue_size();

template <typename eletype>
class l1_none_queue
{
public:
    eletype buffer[MAX_L1NQ_BATCH_SIZE];
    eletype init_limits;
    int batchSize;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};

template <typename eletype>
class l1_vector_queue
{
public:
    eletype data[MAX_L1V_BATCH_SIZE];
    eletype buffer[MAX_L1V_BATCH_SIZE];
    eletype init_limits;
    int batchSize;
    int data_size;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};

// only write back 
template <typename eletype>
class l1_filter_queue
{
public:
    eletype data[MAX_L1FQ_BATCH_SIZE];
    eletype buffer[MAX_L1FQ_BATCH_SIZE];
    eletype init_limits;
    VALUE_TYPE base;
    int batchSize;
    int data_size;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};

// relaxed shortest length first
template <typename eletype>
class l1_SLF_queue{
public:
    eletype data[MAX_L1SLF_BATCH_SIZE];
    eletype buffer[MAX_L1SLF_BATCH_SIZE];
    eletype init_limits;
    int batchSize;
    int data_size;
    int l,r;
    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);
    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);
    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);
    __device__ int get_queue_size();
};

// Read Min: At least read READ_MIN elements. If near bucket is empty, read far bucket.
// Only read far bucket when near bucket is empty.
#define READ_MIN 1

template <typename eletype>
class l1_near_far_queue
{
public:
    int near_num;
    int far_num;

    // near: dist <  base + delta
    // far:  dist >= base + delta 
    VALUE_TYPE base;
    VALUE_TYPE delta;

    // maximum element
    eletype init_limits;

    // data : [0 ~ near_bucket_size] near data
    //        [near_bucket_size ~ near_bucket_size + far_bucket_size] far data
    eletype data[(2 * MAX_L1NFQ_BATCH_SIZE)];
    // buffer
    eletype buffer[(2 * MAX_L1NFQ_BATCH_SIZE)];

    // vector_size
    int near_bucket_size;
    int far_bucket_size;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    // read at most node_size elements
    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    // write size must < bucket_size / BF
    // buffer: [0 ~ near_bucket_size / BF] near data write back 
    //         [near_bucket_size / BF ~ (near_bucket_size + far_bucket_size) / BF] far data write back
    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);
    __device__ write_status write2(eletype ele_out, bool update_predicate, eletype *buffer, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};

#include "../src/l1_queue.cu"

#include "../new_filter/new_filter_queue.cuh"
#include "../l1_hop_queue/l1_hop_queue.cuh"

#undef READ_MIN
#undef N_TH
#undef N_WB
#undef F_TH
#undef F_WB