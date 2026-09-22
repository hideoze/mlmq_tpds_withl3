
#pragma once

#include "graph_info.h"

struct mlmq_setup
{
    mlmq_type type;

    // common
    // l1_batch_size equals to node_size
    int s_l2_batch_size;
    int s_min_read_gra;

    // L2DQ
    int s_l2_delta;
    int s_BNUM;
    int s_BUCKET_MAX;

    // L1V
    int s_l1_vector_size;

    // L1LSF
    int s_l1_SLF_size;

    // L1NF
    int s_l1_near_far_size;
    int s_l1_near_far_delta;

    // L1FQ
    int s_l1_filter_delta;
    int s_l1_filter_size;

    // init with configurations in common.h
    void init_setup()
    {
        type = MLMQ_TYPE;

        s_l2_batch_size = l2_batch_size;
        s_min_read_gra = MIN_READ_GRA;

        // L2DQ
        s_l2_delta = l2_delta;
        s_BNUM = BNUM;
        s_BUCKET_MAX = BUCKET_MAX;

        // L1V
        s_l1_vector_size = l1_vector_size;

        // L1SLF
        s_l1_SLF_size = l1_SLF_size;
        
        // L1NF
        s_l1_near_far_size = l1_filter_size;
        s_l1_near_far_delta = l1_filter_delta;

        // L1FQ
        s_l1_filter_size = l1_near_far_size;
        s_l1_filter_delta = l1_near_far_delta;
    }

    // Currently the adaptive setup policy is limited to the dataset used in the paper.
    // We are appending more datasets and refining our setup policy.
    void init_setup_adaptive(graph_info info)
    {
#if (SETUP_SHOW == true)
        printf("----------Adaptive setup-------------\n");
#endif
        // power-law graphs
        // Use L1V_L2DQ with limited parallelism for better work efficiency
        if (info.dev_nnz > 10)
        {
            type = L1V_L2DQ;
            s_BNUM = 8;
            s_BUCKET_MAX = 4;
            s_l2_delta = info.avg_weight - info.dev_weight;
            s_l2_batch_size = node_size / 4;
            s_min_read_gra = s_l2_batch_size;
#if (SETUP_SHOW == true)
            printf("MLMQ type: L1V_L2DQ\n");
            printf("l2 batchSize %d\n", s_l2_batch_size);
            printf("L2DQ bucket num %d concurrent bucket num %d delta %.2f\n", s_BNUM, s_BUCKET_MAX, (float)s_l2_delta);
#endif
        }
        else if (info.avg_nnz > 3)
        {
            type = L1V_L2V;
            s_l2_batch_size = node_size / 4;
            s_min_read_gra = s_l2_batch_size;
#if (SETUP_SHOW == true)
            printf("MLMQ type: L1V_L2V\n");
            printf("l2 batchSize %d\n", s_l2_batch_size);
#endif
        }
        else if (info.diameter == -1)
        {
            type = L1NF_L2V;
            s_l2_delta = info.avg_weight * 5;
            s_l1_near_far_size = node_size * 2;
            s_l1_near_far_delta = s_l2_delta;
            s_l2_batch_size = 8;
            s_min_read_gra = 8;
#if (SETUP_SHOW == true)
            printf("MLMQ type: L1NF_L2V\n");
            printf("l1_filter_delta %.2f l1_filter_size %d\n", (float)s_l1_near_far_delta, s_l1_near_far_size);
            printf("l2 batchSize %d\n", s_l2_batch_size);
#endif
        }
        else
        {
            printf("Not implemented!\n");
        }
#if (SETUP_SHOW == true)
        printf("-------------------------------------\n");
#endif
    }

    mlmq_setup() {}
};