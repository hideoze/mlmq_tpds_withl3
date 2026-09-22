#pragma once

#include "csr_graph.h"
#include "../core/include/ml_queue.cuh"

// 1GB device memory
#define GPU_MEMORY INT_MAX

// calculate total work count
#define ANALYSIS false

#ifndef WORK_COUNT
#define WORK_COUNT true
#endif
#define WORK_CLOCK ANALYSIS
#define PROFILE_COUNT ANALYSIS

#define EDGE_WISE false

#define LARGEV 16

class node_struct
{
public:
    int id = 0;
#if (USE_DIST_IN_STRUCT == true)
    VALUE_TYPE dist;
#endif

    __device__ bool operator<(const node_struct& b);
    __device__ bool operator>(const node_struct& b);
    __device__ bool operator<=(const node_struct& b);
    __device__ bool operator>=(const node_struct& b);

    __host__ __device__ node_struct& operator=(const int id_in);

    __device__ bool filter();

    __device__ VALUE_TYPE get_data();

    __host__ __device__ node_struct(int id_in, VALUE_TYPE dist_in);

    __host__ __device__ node_struct();
};

#define NODE_TYPE node_struct
