#pragma once

#include "csr_graph.h"
#include <cstring>

#define MAX_GPU 8

#ifndef EDGE_BALANCED_PARTITION
// 连续顶点区间默认按顶点数切分；按边数切分只作为显式实验开关。
// 可用 -DEDGE_BALANCED_PARTITION=true 构建边数均衡对照。
#define EDGE_BALANCED_PARTITION false
#endif

#ifndef PARTITION_CUT_PERCENT
// 仅用于分区敏感性实验：nGPU=2 时把 cut 固定在 nnodes 的百分比位置。
// 默认 -1，不改变现有按顶点数/按边数的分区策略。
// 运行时可用 MLMQ_CUT_PERCENT 环境变量覆盖编译值（L3 setup 参数，便于
// 逐图扫描与选择而无需重编译；非法/缺省回落到编译值）。
#define PARTITION_CUT_PERCENT -1
#endif
#include <cstdlib>
static inline int l3_runtime_cut_percent()
{
    const char *env = getenv("MLMQ_CUT_PERCENT");
    if (env && *env)
    {
        char *end = nullptr;
        long v = strtol(env, &end, 10);
        if (end && *end == '\0' && v >= 1 && v <= 99)
            return (int)v;
    }
    return PARTITION_CUT_PERCENT;
}

// 返回第 part_id 个分区的右边界（0-based、半开区间）。边数前缀单调，
// 用二分查找找到目标边数附近的顶点边界，再选相邻两个顶点中更接近者。
// 仍然是连续 id 区间，因此 node_data/peer_v_begin/CSR 局部索引协议无需改变。
static inline int partition_vertex_boundary(const CSRGraphTy &g, int n_gpu, int part_id)
{
    if (part_id <= 0)
        return 0;
    if (n_gpu <= 1)
        return g.nnodes;
    if (part_id >= n_gpu)
        return g.nnodes;
    if (g.nnodes <= n_gpu)
        return (int)((long long)g.nnodes * part_id / n_gpu);

    long long target = (long long)g.nedges * part_id / n_gpu;
    int lo = 0;
    int hi = g.nnodes;
    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if ((long long)g.row_start[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }

    int upper = lo;
    int boundary = upper;
    if (upper > 0 && upper < g.nnodes)
    {
        int lower = upper - 1;
        long long dl = target - (long long)g.row_start[lower];
        long long du = (long long)g.row_start[upper] - target;
        boundary = (dl <= du) ? lower : upper;
    }

    // 每张卡至少保留一个顶点，避免极小图或零边图产生空分区。
    int min_boundary = part_id;
    int max_boundary = g.nnodes - (n_gpu - part_id);
    if (boundary < min_boundary) boundary = min_boundary;
    if (boundary > max_boundary) boundary = max_boundary;
    return boundary;
}

// 返回当前实际分区策略的边界。所有依赖 owner/cut 的 host 逻辑必须使用此函数，
// 不能在 EDGE_BALANCED_PARTITION=false 时误用按边数边界。
static inline int partition_boundary(const CSRGraphTy &g, int n_gpu, int part_id)
{
    const int cut_percent = l3_runtime_cut_percent();
    if (cut_percent >= 0)
    {
        if (n_gpu == 2 && part_id == 1)
        {
            int boundary = (int)((long long)g.nnodes * cut_percent / 100);
            if (boundary < 1) boundary = 1;
            if (boundary >= g.nnodes) boundary = g.nnodes - 1;
            return boundary;
        }
    }
#if (EDGE_BALANCED_PARTITION == true)
    return partition_vertex_boundary(g, n_gpu, part_id);
#else
    if (part_id <= 0)
        return 0;
    if (n_gpu <= 1)
        return g.nnodes;
    if (part_id >= n_gpu)
        return g.nnodes;
    return (int)((long long)g.nnodes * part_id / n_gpu);
#endif
}

// ===== 顶点分区（edge-cut 连续 id 区间，design_v2 §4/A） =====
// 每 GPU 持 [v_begin, v_end) 顶点（0-based 全局 id）。
// 本地 CSR 子图行索引 = 全局 id - v_begin；col_idx 存全局 0-based 目标 id。
// nGPU=1 时 v_begin=0, v_end=m，子图 = 全图（完全退化为单卡逻辑）。
struct graph_partition
{
    int n_gpu;
    int gpu_id;
    int v_begin;   // 本卡第一个全局顶点 (0-based)
    int v_end;     // 本卡最后一个全局顶点 + 1
    int v_local;   // = v_end - v_begin
    int m;         // 全局顶点数
    int nnz;       // 全局边数
    int nedges_local;

    int *row_start_h;
    int *col_idx_h;
    VALUE_TYPE *edge_data_h;

    int *row_start_d;
    int *col_idx_d;
    VALUE_TYPE *edge_data_d;

    graph_partition() : n_gpu(1), gpu_id(0), v_begin(0), v_end(0), v_local(0),
        m(0), nnz(0), nedges_local(0),
        row_start_h(NULL), col_idx_h(NULL), edge_data_h(NULL),
        row_start_d(NULL), col_idx_d(NULL), edge_data_d(NULL) {}

    // 从全图 host CSR 构造本卡子图
    void construct(CSRGraphTy &g, int n_gpu_in, int gpu_id_in)
    {
        n_gpu = n_gpu_in;
        gpu_id = gpu_id_in;
        m = g.nnodes;
        nnz = g.nedges;
        v_begin = partition_boundary(g, n_gpu, gpu_id);
        v_end   = partition_boundary(g, n_gpu, gpu_id + 1);
        v_local = v_end - v_begin;

        row_start_h = (int *)malloc((v_local + 1) * sizeof(int));
        for (int i = 0; i <= v_local; i++)
            row_start_h[i] = g.row_start[v_begin + i] - g.row_start[v_begin];
        nedges_local = row_start_h[v_local];
        col_idx_h = (int *)malloc(nedges_local * sizeof(int));
        edge_data_h = (VALUE_TYPE *)malloc(nedges_local * sizeof(VALUE_TYPE));
        memcpy(col_idx_h, g.edge_dst + g.row_start[v_begin], nedges_local * sizeof(int));
        memcpy(edge_data_h, g.edge_data + g.row_start[v_begin], nedges_local * sizeof(VALUE_TYPE));
    }

    // 拷子图到当前 device
    void copy_to_gpu()
    {
        cudaMalloc((void **)&row_start_d, (v_local + 1) * sizeof(int));
        cudaMalloc((void **)&col_idx_d, nedges_local * sizeof(int));
        cudaMalloc((void **)&edge_data_d, nedges_local * sizeof(VALUE_TYPE));
        cudaMemcpy(row_start_d, row_start_h, (v_local + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(col_idx_d, col_idx_h, nedges_local * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_data_d, edge_data_h, nedges_local * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    }

    void free_host()
    {
        free(row_start_h); free(col_idx_h); free(edge_data_h);
        row_start_h = NULL; col_idx_h = NULL; edge_data_h = NULL;
    }

    void free_device()
    {
        cudaFree(row_start_d); cudaFree(col_idx_d); cudaFree(edge_data_d);
        row_start_d = NULL; col_idx_d = NULL; edge_data_d = NULL;
    }
};
