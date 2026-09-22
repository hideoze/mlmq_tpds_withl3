
#pragma once

#include <math.h>
#include "common.h"

#define INFO_VALUE_TYPE VALUE_TYPE

struct graph_info
{
    int m;
    int nnz;
    float avg_nnz;
    float max_nnz;
    float dev_nnz;
    float avg_weight;
    float dev_weight;
    float max_weight;

    // -1: No estimation
    float diameter;

    graph_info()
    {
    }

    void init_graph_info(int m_in, int nnz_in, int *RowPtr, int *ColIdx, INFO_VALUE_TYPE *values)
    {
        m = m_in;
        nnz = nnz_in;

        avg_nnz = max_nnz = 0;
        for (int i = 0; i < m; i++)
        {
            avg_nnz += RowPtr[i + 1] - RowPtr[i];
            if (RowPtr[i + 1] - RowPtr[i] > max_nnz)
                max_nnz = RowPtr[i + 1] - RowPtr[i];
        }
        avg_nnz /= m;

        dev_nnz = 0;
        for (int i = 0; i < m; i++)
            dev_nnz += pow(RowPtr[i + 1] - RowPtr[i] - avg_nnz, 2);
        dev_nnz = sqrt(dev_nnz / m);

        avg_weight = 0.0;
        max_weight = 0.0;
        for (int i = 0; i < nnz; i++)
        {
            avg_weight += values[i];
            if (values[i] > max_weight)
                max_weight = values[i];
            // if (values[i] != 1)
            // {
            //     printf("error? %d %d\n", i, values[i]);
            // }
            // if (i % 1000 == 0)
            //     printf("i %d v %.3f %.3f\n", i, float(values[i]), avg_weight);
        }
        // printf("??? %.3f\n", avg_weight);
        avg_weight /= nnz;

        dev_weight = 0;
        for (int i = 0; i < nnz; i++)
        {
            dev_weight += pow(values[i] - avg_weight, 2);
        }
        // printf("%.3f %d %.3f\n", dev_weight, nnz, avg_weight);
        dev_weight = sqrt(dev_weight / nnz);

        diameter = -1;
    }

    void show()
    {
        printf("----------Graph information----------\n");
        printf("number of vertices %d number of edges %d\n", m, nnz);
        printf("average nnz %.2f maximum nnz %.2f deviation nnz %.2f\n", avg_nnz, max_nnz, dev_nnz);
        printf("average weight %.2f maximum weight %.2f deviation weight %.2f \n", avg_weight, max_weight, dev_weight);
        printf("-------------------------------------\n");
    }
};