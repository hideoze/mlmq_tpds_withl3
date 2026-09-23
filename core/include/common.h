#pragma once

#include "GPU_setup.h"

// Type of Multi-Level-Multi-Queue
// #define MLMQ_TYPE L1V_L2DQ
// #define MLMQ_TYPE L1SLF_L2DQ
#define MLMQ_TYPE L1SLF_L2DQ
// #define MLMQ_TYPE L1NF_L2V

#define TYPE_INT

#ifdef TYPE_FLOAT
    #define VALUE_TYPE float
    #define DIST_MAX FLT_MAX
#else
    #define VALUE_TYPE int
    #define DIST_MAX INT_MAX
#endif

//------------------------------------------------------//
// Here are parameters users can change
// The size of SSSP node buffer: {16, 32, 64}
#define node_size 32
// The batch size of l2 queue: {8, 16, 32}
// The batch size of l1 queue is dependent on node_size
#ifndef l2_batch_size
#define l2_batch_size 8
#endif
// delta value, related to edge weights
#define mlmq_delta 2e5
// L2 delta queue
// Number of delta queue buckets
#ifndef BNUM
#define BNUM 16
#endif
// Number of buckets can concurrently read data from
#ifndef BUCKET_MAX
#define BUCKET_MAX 4
#endif
#if (l2_batch_size != 8 && l2_batch_size != 16 && l2_batch_size != 32) || \
    BNUM <= 0 || BUCKET_MAX <= 0 || BUCKET_MAX > BNUM
#error "Invalid L2 delta-queue bucket geometry"
#endif
//------------------------------------------------------//

// Do not change following parameters

// For L2PQ and L2MPQ
// #define MIN_READ_GRA (1)
// For other queues
#define MIN_READ_GRA (l2_batch_size)

// L2 delta queue
#define l2_delta (mlmq_delta)

// l1 none queue
#define MAX_L1NQ_BATCH_SIZE 64

// l1 vector queue
#define MAX_L1V_BATCH_SIZE 64
#define l1_vector_size (32)

// l1 filter queue
// l1_filter_size must >= 2 * node_size
#define MAX_L1FQ_BATCH_SIZE 64
#define l1_filter_size (2 * node_size)
#define l1_filter_delta (l2_delta)

#define MAX_L1SLF_BATCH_SIZE 64
#define l1_SLF_size (node_size)

// l1 near far queue
#define MAX_L1NFQ_BATCH_SIZE 64
#define l1_near_far_size (2 * node_size)
#define l1_near_far_delta (l2_delta)
// When near bucket data num > N_TH, write back size of N_WB - N_TH
#define N_TH (near_bucket_size / 2)
#define N_WB (near_bucket_size)
// When near bucket data num > F_TH, write back size of F_WB
#define F_TH (far_bucket_size / 2)
#define F_WB (far_bucket_size)

// l2 multi queue
#define L2_MQ_NUM 4

#define FULL_MASK 0xffffffff

#define USE_DIST_IN_STRUCT true

#define SETUP_SHOW true

#define ACCESS_THROUGH false

enum mlmq_type
{
    L1V_L2DQ,
    L1N_L2DQ,
    L1NF_L2DQ,
    L1V_L2V,
    L1N_L2V,
    L1NF_L2V,
    L1FQ_L2DQ,
    L1V_L2PQ,
    L1NF_L2PQ,
    L1FQ_L2PQ,
    L1V_L2MPQ,
    L1NF_L2MPQ,
    L1FQ_L2MPQ,
    L1HQ_L2V,
    L1HQ_L2DQ,
    L1FQ_L2V,
    L1N_L2MV,
    L1N_L2BV,
    L1V_L2BV,
    L1SLF_L2DQ,
    L1SLF_L2V
};

enum init_status
{
    INIT_SUCCESS,
    INIT_FAILED
};

enum read_status
{
    READ_SUCCESS,
    READ_EMPTY,
    READ_PARTIAL
};

enum write_status
{
    WRITE_SUCCESS,
    WRITE_EMPTY
};

struct node_data
{
    int id;
    unsigned dist;
};

struct mlmq_mdata
{
    // record the offset in shared memory buffer (bytes)
    int l1_queue_offset = 0;
    int l1_bufsize_perwarp = 0;
    int l2_queue_offset = 0;
    int l2_bufsize_perwarp = 0;
    int l2_bufsize_extra = 0;
};

#define mlq_max(a, b) ((a) > (b)?(a): (b))
#define mlq_min(a, b) ((a) < (b)?(a): (b))

#define CUDA_CHECK(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
