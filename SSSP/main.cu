#include "common.h"
#include "supplement_oracle.h"
#include "csr_graph.h"
#include "graph_info.h"
//#include "ml_queue.cuh"
#include "paras.h"
#include "graph_partition.h"
#include "sssp.cuh"
#include <queue>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <vector>
#include "benchmark.h"
#if (L3_CHAIN_SHORTCUTS == true)
#include "l3/l3_chain_shortcuts.h"
#endif

#if (L3_CHAIN_PARTITION == true)
#include "l3/l3_chain_partition.h"
extern void l3_chain_partition_install(int,const l3_chain_partition_index &);
#endif

#if (L3_REGION_RELAX == true)
#include "l3/l3_region_index.h"
extern void l3_region_install(int,const l3_region_index &);
#endif

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

extern int sssp_init(int gpu_id, int m_in, int nnz_in, int *RowPtr_in, int *ColIdx_in,
                     VALUE_TYPE *edge_data_in, int v_begin_in, int v_end_in);
extern int sssp_setup_peers(int n_gpu);
extern int sssp_setup_independent(int physical_gpu_count);
extern int sssp_run_adaptive(int gpu_id, int src, graph_info info);
extern int sssp_cp_data(int gpu_id, VALUE_TYPE *node_data_h);
extern int sssp_release_l3(int gpu_id);
#if (L3_TIMING_DIAG == true)
extern void sssp_report_timing(int n_gpu, int sample, bool warmup);
#endif
#ifdef TYPE_INT
extern void sssp_audit_final(int, int, const int *, const int *, const VALUE_TYPE *,
                             const VALUE_TYPE *, const VALUE_TYPE *);
#endif
extern int g_delta_override;
extern int g_queue_override;
extern int g_multi_source_quiet;
#if (L3_RX_EXPRESS == true)
extern int g_rx_express_enabled;
#endif
#if (L3_RX_L2_PULL == true)
extern int g_rx_l2_pull_enabled;
#endif
#if (SEED_EXP == true)
extern int sssp_set_seeds(int gpu_id, int n_seeds, int *seed_ids, VALUE_TYPE *seed_dists);
#endif
#if (GHOST_DEPTH > 0)
extern int sssp_set_ghost(int gpu_id, int ghost_num, int *ghost_row_start_h, int *ghost_col_h,
                          VALUE_TYPE *ghost_edge_data_h, int *g_id_to_idx_h);
#endif

VALUE_TYPE *node_data_base;

// struct on host, only for 
class node_struct_base
{
public:
    int id = 0;
	VALUE_TYPE dist;

    bool operator<(const node_struct_base& b) const;
    bool operator>(const node_struct_base& b) const;
    bool operator<=(const node_struct_base& b) const;
    bool operator>=(const node_struct_base& b) const;

    node_struct_base& operator=(const int id_in) { this->id = id_in; return *this; };

    VALUE_TYPE get_data() { return this->dist; };

    node_struct_base (int id_in, VALUE_TYPE dist_in) : id(id_in), dist(dist_in) {};

    node_struct_base () {};
};

bool node_struct_base::operator<(const node_struct_base& b) const { return this->dist < b.dist; }
bool node_struct_base::operator>(const node_struct_base& b) const { return this->dist > b.dist; }
bool node_struct_base::operator<=(const node_struct_base& b) const { return this->dist <= b.dist; }
bool node_struct_base::operator>=(const node_struct_base& b) const { return this->dist >= b.dist; }

void sssp_sequential(int m, int nnz, int *RowPtr, int *ColIdx, VALUE_TYPE *edge_data, VALUE_TYPE *node_data, int *prep, int *hop, int src, int &total_work, bool print_hop = false)
{
	std::priority_queue<node_struct_base, std::vector<node_struct_base>, std::greater<node_struct_base>> q;
	q.push(node_struct_base(src, 0));

	prep[src] = src;
	hop[src] = 0;
	int hop_max = 0;

	int count = 0;

	total_work = 0;

	int *done=new int[m];
	for (int i = 0; i < m; i++) done[i] = 0;

	while (!q.empty())
	{
		int tid = q.top().id;
		q.pop();
		if (done[tid]) continue;
		done[tid] = 1;
		//printf("new id, %d\n", tid);

		total_work += RowPtr[tid + 1] - RowPtr[tid];

		for (int j = RowPtr[tid]; j < RowPtr[tid + 1]; j++)
		{

			int nid = ColIdx[j];

			if (node_data[nid] > node_data[tid] + edge_data[j])
			{
				node_data[nid] = node_data[tid] + edge_data[j];
				prep[nid] = tid;
				hop[nid] = hop[tid] + 1;
				if (hop[nid] > hop_max)
					hop_max = hop[nid];
				q.push(node_struct_base(nid, node_data[tid] + edge_data[j]));
			}
		}
		count++;
	}
	delete[] done;
	if (print_hop)
	{
		int hop_count[hop_max + 1];
		memset(hop_count, 0, sizeof(int) * (hop_max + 1));
		hop_max = 0;
		for (int i = 0; i < m; i++)
		{
			hop_count[hop[i]]++;
			if (hop[i] > hop_max)
				hop_max = hop[i];
		}
		float avg_weight = 0;
		int weight_num = 0;
		for (int i = 0; i < m; i++)
		{
			if (i == src || hop[i] > 0)
			{
				avg_weight += edge_data[ColIdx[i]];
				weight_num++;
			}
		}
		printf("average weight %.2f\n", avg_weight / weight_num);
		printf("total hop: %d hop count: ", hop_max);
		for (int i = 0; i < hop_max + 1; i++)
			printf(" %d", hop_count[i]);
		printf("\n");
	}
}

void find_path_to(int *prep, int src, int dst)
{
	printf("path to %d:\n", dst);
	while (src != dst)
	{
		printf("%d(%d) ", dst);
		dst = prep[dst];
	}
	printf("%d ", dst);
	printf("\n");
}

int node_data_check(int m, VALUE_TYPE *node_data, VALUE_TYPE *node_data_base)
{
    if(supplement_oracle_check(node_data,m)) return 1;
	for (int i = 0; i < m; i++)
	{
		if (node_data[i] != node_data_base[i])
		{
			std::cout << "Error at node " << i << std::endl;
			std::cout << "dist " << node_data[i] << " base " << node_data_base[i] << std::endl;
			return 1;
		}
	}
	return 0;
}

// 多源数据并行路径的逐源校验。每个 worker 只写自己负责的 source_id 槽位，
// 因而可以在 GPU worker 并行运行时收集结果，最后由 host 统一打印。
static int multi_source_check(int m, const VALUE_TYPE *gpu_data,
                              const VALUE_TYPE *reference, int &first_bad)
{
    first_bad = -1;
    int errors = 0;
    for (int i = 0; i < m; i++)
    {
        if (gpu_data[i] != reference[i])
        {
            if (first_bad < 0) first_bad = i;
            errors++;
        }
    }
    return errors;
}

// 同一张完整图的多源批量：单卡串行，多卡按 source 动态分配任务。
// 每个物理 GPU 的 gctx 仍按 n_gpu=1 初始化，避免进入现有跨卡协议。
static int run_multi_source_batch(CSRGraphTy &g, graph_info info, int n_gpu,
                                  const std::vector<int> &source_ids,
                                  const std::vector<std::vector<VALUE_TYPE>> &references)
{
    const int source_count = (int)source_ids.size();
    graph_partition full_graph[MAX_GPU];

    for (int i = 0; i < n_gpu; i++)
    {
        // n_gpu=1 使每个副本都构造成完整 CSR，i 只作为物理设备编号使用。
        full_graph[i].construct(g, 1, 0);
        printf("MULTI-SOURCE GPU%d full graph: v=%d e=%d\n",
               i, full_graph[i].v_local, full_graph[i].nedges_local);
    }

    for (int i = 0; i < n_gpu; i++)
    {
        cudaSetDevice(i);
        full_graph[i].copy_to_gpu();
        sssp_init(i, g.nnodes, g.nedges,
                  full_graph[i].row_start_d, full_graph[i].col_idx_d,
                  full_graph[i].edge_data_d, 0, g.nnodes);
    }

    if (sssp_setup_independent(n_gpu) != 0)
    {
        printf("MULTI-SOURCE failed to configure independent GPU contexts\n");
        return 1;
    }

    printf("MULTI-SOURCE begin: sources=%d physical_gpus=%d logical_gpus=1\n",
           source_count, n_gpu);
	// kernel_adaptive 的逐 query 诊断输出会让多个 worker 争用 stdout，
	// 只在 batch 计时期间关闭；错误路径和最终逐源汇总仍保留。
	g_multi_source_quiet = 1;

    std::vector<int> source_errors(source_count, 0);
    std::vector<int> source_first_bad(source_count, -1);
    std::atomic<int> next_source(0);
    std::vector<std::thread> workers;
    workers.reserve(n_gpu);

    struct timeval tv_gpu_begin, tv_gpu_end;
    gettimeofday(&tv_gpu_begin, NULL);

    for (int gpu_id = 0; gpu_id < n_gpu; gpu_id++)
    {
        workers.emplace_back([&, gpu_id]()
        {
            while (true)
            {
                int source_index = next_source.fetch_add(1, std::memory_order_relaxed);
                if (source_index >= source_count) break;

                sssp_run_adaptive(gpu_id, source_ids[source_index], info);

                std::vector<VALUE_TYPE> gpu_data(g.nnodes);
                sssp_cp_data(gpu_id, gpu_data.data());
                source_errors[source_index] = multi_source_check(
                    g.nnodes, gpu_data.data(), references[source_index].data(),
                    source_first_bad[source_index]);
            }
        });
    }

    for (std::thread &worker : workers)
        worker.join();
	g_multi_source_quiet = 0;

    gettimeofday(&tv_gpu_end, NULL);
    float batch_time = duration(tv_gpu_begin, tv_gpu_end) / 1000.0;

    int total_errors = 0;
    for (int k = 0; k < source_count; k++)
    {
        total_errors += source_errors[k];
        if (source_errors[k] == 0)
            printf("MULTI-SOURCE source[%d]=%d correct\n", k, source_ids[k]);
        else
            printf("MULTI-SOURCE source[%d]=%d errors=%d first_bad=%d\n",
                   k, source_ids[k], source_errors[k], source_first_bad[k]);
    }

    printf("MULTI-SOURCE sources=%d nGPU=%d batch wall time=%.2f ms\n",
           source_count, n_gpu, batch_time);
    printf("MULTI-SOURCE average query time=%.2f ms\n",
           batch_time / source_count);
    printf("MULTI-SOURCE errors=%d\n", total_errors);
    if (total_errors == 0)
        printf("MULTI-SOURCE all sources correct!\n");

    for (int gpu_id = 0; gpu_id < n_gpu; gpu_id++)
        sssp_release_l3(gpu_id);
    return total_errors == 0 ? 0 : 1;
}

int main(int argc, char *argv[]) {
    const double bench_process_start = mlmq_bench_ms();

	// argument parsing
	char *input_name;
	int n_gpu;
	int delta_override;
	int num_sources;
	if (parse_args(argc, argv, input_name, n_gpu, delta_override, num_sources) != 0) return -1;

	printf("Graph name: %s\n", input_name);
	printf("nGPU: %d\n", n_gpu);
	printf("sources: %d\n", num_sources);
	if (delta_override > 0)
		printf("delta override: %d\n", delta_override);
	g_delta_override = delta_override;

	int dev_count = 0;
	cudaGetDeviceCount(&dev_count);
	if (n_gpu > dev_count) {
		printf("n_gpu %d > device count %d\n", n_gpu, dev_count);
		return -1;
	}

	float avg_degree, avg_weight;

	CSRGraphTy g;
	const double bench_input_start = mlmq_bench_ms();
	g.read(input_name, avg_degree, avg_weight);
	const double bench_input_ms = mlmq_bench_ms() - bench_input_start;

	// ===== 双卡通信开销分析（2026-08-20 用户指示：取消单卡降级）=====
	// 所有图都真正跑双卡（n=2 不降级），用"双卡 vs 单卡"的 overhead 差值反映
	// 通信开销来源，据此反思优化。仅保留队列自适应（选 L1V_L2DQ，不降级）。
	if (n_gpu > 1)
	{
		long long cross = 0;
		// 队列统计必须使用当前实际的连续分区边界。
		int cut = partition_boundary(g, n_gpu, 1);
		for (int u = 0; u < g.nnodes; u++)
			for (int e = g.row_start[u]; e < g.row_start[u + 1]; e++)
				if ((u < cut) != (g.edge_dst[e] < cut)) cross++;
		if (cross > 0 && g.nnodes >= 300000)
		{
			// 队列自适应: 跨卡边占比 < 0.07%（计算主导，实测 rgg_22/hugetrace）→ L1V_L2DQ
			//   （双卡绝对更快: rgg_22 35.9→28.8ms、hugetrace 36→34.7ms）；del_n23（0.09%）
			//   保持 L1SLF_L2DQ（L1V 单卡快但比值更差）
			double cross_ratio = (double)cross / g.nedges;
			if (cross_ratio < QUEUE_CROSS_RATIO_THRESHOLD)
			{
				g_queue_override = L1V_L2DQ;
				printf("ADAPTIVE-Q: cross_ratio=%.4f%% < %.4f%% -> L1V_L2DQ\n",
				       cross_ratio * 100.0, QUEUE_CROSS_RATIO_THRESHOLD * 100.0);
			}
		}
	}

	// graph information for adaptive setup on host
	graph_info info;
	info.init_graph_info(g.nnodes, g.nedges, g.row_start, g.edge_dst, g.edge_data);
	info.show();

    VALUE_TYPE init_max = std::numeric_limits<VALUE_TYPE>::max();

	// No negative edge weight is allowed
	for (int i = 0; i < g.nedges; i++)
		if (g.edge_data[i] < 0)
		{
			printf("Negative edge weight!\n");
			return -1;
		}

#if (L3_MULTI_PRODUCER == true)
    // Query gate reset relies on the existing BENCH pre-launch two-device
    // barrier. Reject unsupported execution paths rather than racing reset.
    const char *producer_bench=std::getenv("MLMQ_BENCH");
    if(n_gpu!=2 || num_sources>1 || !producer_bench || std::string(producer_bench)!="1") {
        fprintf(stderr,"multi-producer prototype requires n=2, one source, MLMQ_BENCH=1\n");
        return 2;
    }
#endif
#if (L3_CHAIN_PARTITION == true)
    if(num_sources>1) {fprintf(stderr,"chain partition prototype uses repeated single-source queries\n");return 2;}
#endif
#if (L3_REGION_RELAX == true)
    if(num_sources>1) {fprintf(stderr,"region prototype uses repeated single-source queries\n");return 2;}
#endif
	if (num_sources > 1)
	{
#if (L3_CHAIN_SHORTCUTS == true)
        fprintf(stderr,"stage222 supports one source per process; no independent-source fallback\n");
        return 2;
#endif
		std::vector<int> source_ids(num_sources);
		for (int k = 0; k < num_sources; k++)
		{
			if (g.nnodes <= 0) source_ids[k] = 0;
			else if (num_sources <= g.nnodes)
				source_ids[k] = (int)((long long)k * g.nnodes / num_sources);
			else
				source_ids[k] = k % g.nnodes;
		}

		// CPU 参考解全部在 GPU 计时前完成，确保 batch wall time 只反映
		// 每源 re_init + GPU kernel + 结果 D2H 拷贝。
		std::vector<std::vector<VALUE_TYPE>> references;
		references.reserve(num_sources);
		std::vector<int> prep(g.nnodes);
		std::vector<int> hop(g.nnodes);
		struct timeval tv_ref_begin, tv_ref_end;
		gettimeofday(&tv_ref_begin, NULL);
		for (int k = 0; k < num_sources; k++)
		{
			references.emplace_back(g.nnodes, init_max);
			references.back()[source_ids[k]] = 0;
			int source_work = 0;
			sssp_sequential(g.nnodes, g.nedges, g.row_start, g.edge_dst,
			                g.edge_data, references.back().data(), prep.data(),
			                hop.data(), source_ids[k], source_work);
		}
		gettimeofday(&tv_ref_end, NULL);
		printf("MULTI-SOURCE CPU references ready: sources=%d time=%.2f ms\n",
		       num_sources, duration(tv_ref_begin, tv_ref_end) / 1000.0);

		return run_multi_source_batch(g, info, n_gpu, source_ids, references);
	}

	int src = bench_env_int("BENCH_SOURCE", 0, 0, g.nnodes - 1);
	const char *fixed_queue = std::getenv("BENCH_QUEUE");
	if (fixed_queue) {
		if (std::string(fixed_queue) == "L1V_L2DQ") g_queue_override = L1V_L2DQ;
		else if (std::string(fixed_queue) == "L1SLF_L2DQ") g_queue_override = L1SLF_L2DQ;
		else { fprintf(stderr, "Unsupported BENCH_QUEUE\n"); return 2; }
	}

#if (L3_REGION_RELAX == true)
    if(n_gpu>2) {fprintf(stderr,"region prototype supports <=2 GPUs\n");return 2;}
    l3_region_index region_indices[MAX_GPU];
    if(n_gpu==2) for(int gpu=0;gpu<n_gpu;++gpu) {
        const double region_start=mlmq_bench_ms();
        try {region_indices[gpu].build(g.nnodes,g.row_start,g.edge_dst,g.edge_data,
                  partition_boundary(g,n_gpu,gpu),partition_boundary(g,n_gpu,gpu+1));}
        catch(const std::exception&e){fprintf(stderr,"REGION_INDEX_ERROR %s\n",e.what());return 2;}
        const auto &index=region_indices[gpu];
        printf("L3_REGION_SETUP gpu=%d width=32 vertices=%d regions=%d padded=%zu internal_edges=%d build_ms=%.6f\n",
               gpu,index.vertices,index.regions,index.members.size(),index.internal_edges,mlmq_bench_ms()-region_start);
    }
#endif
#if (L3_CHAIN_PARTITION == true)
    if(n_gpu>2 || g.nnodes>=L3_CHAIN_PARTITION_TAG) {fprintf(stderr,"chain partition range\n");return 2;}
    l3_chain_partition_index chain_partitions[MAX_GPU];
    if(n_gpu==2) for(int gpu=0;gpu<n_gpu;++gpu) {
        const double begin_time=mlmq_bench_ms();
        try {
            chain_partitions[gpu].build(g.nnodes,g.row_start,g.edge_dst,g.edge_data,
                partition_boundary(g,n_gpu,gpu),partition_boundary(g,n_gpu,gpu+1));
        } catch(const std::exception &e) {fprintf(stderr,"CHAIN_PARTITION_ERROR %s\n",e.what());return 2;}
        printf("L3_CHAIN_PARTITION_SETUP gpu=%d segments=%d interiors=%d overflow=%d build_ms=%.6f\n",
            gpu,chain_partitions[gpu].segments,chain_partitions[gpu].interiors,
            chain_partitions[gpu].overflow,mlmq_bench_ms()-begin_time);
    }
#endif
#if (L3_CHAIN_SHORTCUTS == true)
    // L3 owner-local chain shortcuts are a runtime feature of the L3 layer:
    // they are only built for the multi-GPU (L3) path.  The independent
    // no-L3 single-GPU reference keeps the unmodified CSR, so the shortcut
    // benefit is attributable to L3 and not to a global graph transform.
    bool l3_chain_active = (n_gpu > 1);
    CSRGraphTy chain_graph;
    const double chain_begin=mlmq_bench_ms();
    l3_chain_shortcut_view chain_view;
    if(n_gpu>2) {fprintf(stderr,"stage222 requires one or two GPUs\n");return 2;}
    if (l3_chain_active)
    {
        const int chain_cut=partition_boundary(g,n_gpu,1);
        try {
            chain_view.build(g.nnodes,g.nedges,g.row_start,g.edge_dst,g.edge_data,chain_cut,src);
        } catch(const std::exception &error) {
            fprintf(stderr,"L3_CHAIN_ERROR %s\n",error.what());return 2;
        }
        chain_graph.nnodes=g.nnodes;chain_graph.nedges=int(chain_view.destinations.size());
        chain_graph.row_start=chain_view.rows.data();chain_graph.edge_dst=chain_view.destinations.data();
        chain_graph.edge_data=chain_view.weights.data();
        printf("L3_CHAIN_SETUP loose=%d source=%d cut=%d vertices=%d original_edges=%d eligible=%d shortcuts=%d represented_edges=%lld longest_path=%d overflow_paths=%d build_ms=%.6f\n",
               (int)L3_CHAIN_SHORTCUTS_LOOSE,
               src,chain_cut,g.nnodes,g.nedges,chain_view.eligible_vertices,chain_view.shortcuts,
               chain_view.represented_edges,chain_view.longest_path,chain_view.overflow_paths,
               mlmq_bench_ms()-chain_begin);
    }
#endif
	// sequential run on the unmodified input graph
	node_data_base = (VALUE_TYPE*)malloc(g.nnodes * sizeof(VALUE_TYPE));
	for (int i = 0; i < g.nnodes; i++)
	{
		node_data_base[i] = init_max;
	}
	node_data_base[src] = 0;

	int *prep = (int*)malloc(g.nnodes * sizeof(int));
	int *hop = (int*)malloc(g.nnodes * sizeof(int));

	int total_work = 0;
	struct timeval tv_begin, tv_end;
	gettimeofday(&tv_begin, NULL);
	sssp_sequential(g.nnodes, g.nedges, g.row_start, g.edge_dst, g.edge_data, node_data_base, prep, hop, src, total_work);
	gettimeofday(&tv_end, NULL);
	float cpu_time = duration(tv_begin, tv_end) / 1000.0;

	// find_path_to(prep, src, 9);

	printf("CPU done!\n");
	printf("CPU total work count %d\n", total_work);
	printf("CPU sequential time %.2f ms\n", cpu_time);

#if (SEED_EXP == true)
	// ===== 前置实验（决定性对照）: gpu1 分区单独跑 + 边界最终值预置种子 =====
	// 单卡（GPU0）跑 n_gpu=2 的第二个分区 [m/2, m)，种子 = 该分区边界顶点
	// （有入边来自前半个分区的顶点）的 CPU 参考解最终距离。
	// 量化 gpu1 内部纯 delta-stepping 固有 work（对照 phase化 PoC 的 4755万）。
	{
		int gi = 1;   // 模拟 gpu1 分区
		graph_partition sp;
		sp.construct(g, 2, gi);   // v_begin = m/2, v_end = m
		int v_begin = sp.v_begin, v_end = sp.v_end;
		std::vector<int> seed_ids;
		std::vector<VALUE_TYPE> seed_dists;
		{
			std::vector<char> is_bnd(v_end - v_begin, 0);
			// 全图出边扫描：u→w，w∈[v_begin,v_end) 且 u<v_begin（入边来自前区）→ w 是边界
			for (int u = 0; u < g.nnodes; u++)
			{
				for (int j = g.row_start[u]; j < g.row_start[u + 1]; j++)
				{
					int w = g.edge_dst[j];
					if (w >= v_begin && w < v_end && u < v_begin && node_data_base[w] < init_max)
						is_bnd[w - v_begin] = 1;
				}
			}
			for (int v = v_begin; v < v_end; v++)
				if (is_bnd[v - v_begin])
				{
					seed_ids.push_back(v + 1);                 // 全局 1-based
					seed_dists.push_back(node_data_base[v]);   // 最终距离
				}
		}
		printf("SEED_EXP: gpu1 partition [%d,%d) boundary seeds = %d\n",
		       v_begin, v_end, (int)seed_ids.size());

		cudaSetDevice(0);
		sp.copy_to_gpu();
		sssp_init(0, g.nnodes, g.nedges, sp.row_start_d, sp.col_idx_d,
		          sp.edge_data_d, sp.v_begin, sp.v_end);
		sssp_set_seeds(0, (int)seed_ids.size(), seed_ids.data(), seed_dists.data());
		sssp_setup_peers(1);

		printf("GPU begin!\n");
		struct timeval tv_gpu_begin, tv_gpu_end;
		gettimeofday(&tv_gpu_begin, NULL);
		sssp_run_adaptive(0, src, info);
		gettimeofday(&tv_gpu_end, NULL);
		printf("SEED_EXP wall time %.2f ms\n", duration(tv_gpu_begin, tv_gpu_end) / 1000.0);

		VALUE_TYPE *nd = (VALUE_TYPE*)malloc(g.nnodes * sizeof(VALUE_TYPE));
		sssp_cp_data(0, nd);
		int bad = 0, unreachable_ref = 0, unreachable_gpu = 0;
		for (int v = v_begin; v < v_end; v++)
		{
			if (node_data_base[v] >= init_max) { if (nd[v] < init_max) bad++; unreachable_ref++; continue; }
			if (nd[v] != node_data_base[v]) bad++;
			if (nd[v] >= init_max) unreachable_gpu++;
		}
		printf("SEED_EXP correct: gpu1 range errors=%d (ref_unreachable=%d gpu_unreachable=%d)\n",
		       bad, unreachable_ref, unreachable_gpu);
		if (!bad) printf("SEED_EXP gpu1 range all correct!\n");
		sssp_release_l3(0);
		return 0;
	}
#endif

	// 顶点分区 + 本地 CSR 子图构造（用 host 图 g）
	graph_partition part[MAX_GPU];
	for (int i = 0; i < n_gpu; i++)
	{
#if (L3_CHAIN_SHORTCUTS == true)
        part[i].construct(l3_chain_active ? chain_graph : g, n_gpu, i);
#else
		part[i].construct(g, n_gpu, i);
#endif
		printf("GPU%d partition: [%d, %d) v_local=%d nedges=%d\n",
		       i, part[i].v_begin, part[i].v_end, part[i].v_local, part[i].nedges_local);
	}

	// 逐卡初始化（拷贝子图 + node_data 分区到各卡）
	for (int i = 0; i < n_gpu; i++)
	{
		cudaSetDevice(i);
		part[i].copy_to_gpu();
        sssp_init(i, g.nnodes, g.nedges, part[i].row_start_d, part[i].col_idx_d,
                  part[i].edge_data_d, part[i].v_begin, part[i].v_end);
#if (L3_REGION_RELAX == true)
        if(n_gpu==2)l3_region_install(i,region_indices[i]);
#endif
#if (L3_CHAIN_PARTITION == true)
        if(n_gpu==2) l3_chain_partition_install(i,chain_partitions[i]);
#endif
		cudaError_t init_err = cudaGetLastError();
		if (init_err != cudaSuccess)
		{
			printf("GPU%d initialization CUDA error: %s\n",
			       i, cudaGetErrorString(init_err));
			return -1;
		}
	}

	// 步骤 C: 双向 P2P peer access（跨卡原子直访对方的 node_data/dirty_bitmap）
	if (n_gpu > 1)
	{
		for (int i = 0; i < n_gpu; i++)
		{
			cudaSetDevice(i);
			for (int j = 0; j < n_gpu; j++)
			{
				if (i == j) continue;
				int can = 0;
				cudaError_t peer_err = cudaDeviceCanAccessPeer(&can, i, j);
				if (peer_err != cudaSuccess || !can)
				{
					printf("GPU%d cannot peer access GPU%d: can=%d err=%s\n",
					       i, j, can, cudaGetErrorString(peer_err));
					return -1;
				}
				cudaError_t enable_err = cudaDeviceEnablePeerAccess(j, 0);
				if (enable_err != cudaSuccess
				    && enable_err != cudaErrorPeerAccessAlreadyEnabled)
				{
					printf("GPU%d enable peer GPU%d failed: %s\n",
					       i, j, cudaGetErrorString(enable_err));
					return -1;
				}
			}
		}
	}

	// 步骤 C: 设置 peer 指针（跨卡直访），须在全部卡 init 后
	sssp_setup_peers(n_gpu);

#if (GHOST_DEPTH > 0)
	// ===== 方案 C: 源卡 ghost 子图构造（D 跳闭包，host 全图）=====
	// ghost 集 = 对端边界顶点（有 in-edge 从本卡）→ 沿对端内出边闭包 GHOST_DEPTH-1 次。
	// ghost 出边 = 返回本卡的边 + 指向其他 ghost 的边（非 ghost 对端边无法处理，丢弃）。
	if (n_gpu == 2)
	{
		int peer_beg = part[1].v_begin;
		int peer_end = part[1].v_end;
		int peer_v_local = peer_end - peer_beg;
		std::vector<char> is_ghost(peer_v_local, 0);
		std::vector<int> frontier;
		for (int u = 0; u < peer_beg; u++)
			for (int e = g.row_start[u]; e < g.row_start[u + 1]; e++)
			{
				int w = g.edge_dst[e];
				if (w >= peer_beg && !is_ghost[w - peer_beg])
				{
					is_ghost[w - peer_beg] = 1;
					frontier.push_back(w - peer_beg);
				}
			}
		for (int d = 1; d < GHOST_DEPTH; d++)
		{
			std::vector<int> nxt;
			for (int gi : frontier)
			{
				int x = peer_beg + gi;
				for (int e = g.row_start[x]; e < g.row_start[x + 1]; e++)
				{
					int y = g.edge_dst[e];
					if (y >= peer_beg && !is_ghost[y - peer_beg])
					{
						is_ghost[y - peer_beg] = 1;
						nxt.push_back(y - peer_beg);
					}
				}
			}
			frontier.swap(nxt);
		}
		std::vector<int> ghost_list;
		for (int gi = 0; gi < peer_v_local; gi++)
			if (is_ghost[gi]) ghost_list.push_back(gi);
		int ghost_num = (int)ghost_list.size();
		std::vector<int> g_id_to_idx(peer_v_local, -1);
		for (int i = 0; i < ghost_num; i++) g_id_to_idx[ghost_list[i]] = i;
		std::vector<int> g_rs(ghost_num + 1, 0);
		std::vector<int> g_col;
		std::vector<VALUE_TYPE> g_ed;
		for (int i = 0; i < ghost_num; i++)
		{
			int x = peer_beg + ghost_list[i];
			for (int e = g.row_start[x]; e < g.row_start[x + 1]; e++)
			{
				int y = g.edge_dst[e];
				if (y < peer_beg || is_ghost[y - peer_beg])
				{
					g_col.push_back(y + 1);            // 全局 1-based
					g_ed.push_back(g.edge_data[e]);
				}
			}
			g_rs[i + 1] = (int)g_col.size();
		}
		printf("GHOST: ghost_num=%d (%.1f%% of peer, peer=%d) ghost_edges=%d\n",
		       ghost_num, 100.0 * ghost_num / peer_v_local, peer_v_local, (int)g_col.size());
		sssp_set_ghost(0, ghost_num, g_rs.data(), g_col.data(), g_ed.data(), g_id_to_idx.data());
	}
#endif

	printf("GPU begin!\n");
	// Explicit opt-in; the multi-source throughput path never enters this block.
	const char *benchmark_env = std::getenv("MLMQ_BENCH");
	g_benchmark.enabled = benchmark_env && std::string(benchmark_env) == "1";
	g_benchmark.participants = n_gpu;
	if (g_benchmark.enabled) g_multi_source_quiet = 1;
#if (L3_TILE_LOAN == true)
	// H/P use the same compile-time binary. The runtime value is fixed before
	// the per-GPU query threads are created and cannot change mid-query.
	g_l3_tile_loan_enabled = bench_env_int("MLMQ_TILE_LOAN_ENABLED", 1, 0, 1);
	if (g_benchmark.enabled)
		printf("L3_TILE_LOAN_RUNTIME enabled=%d\n", g_l3_tile_loan_enabled);
#endif
#if (L3_RX_EXPRESS == true)
	// Fixed for the whole process/query batch before GPU worker threads start.
	// Disabled mode retains the express binary's ring allocation and checks,
	// but never publishes or consumes a ring record.
	g_rx_express_enabled = bench_env_int("MLMQ_RX_EXPRESS_ENABLED", 1, 0, 1);
#endif
#if (L3_RX_L2_PULL == true)
	// Fixed for the whole process/query batch before GPU worker threads start.
	// H mode observes the same RX commit hints but forbids the extra q2 read.
	g_rx_l2_pull_enabled = bench_env_int("MLMQ_RX_L2_PULL_ENABLED", 1, 0, 1);
#endif
	if (g_benchmark.enabled) {
		const double setup_wall = mlmq_bench_ms() - bench_process_start;
		printf("BENCH_SETUP algorithm=MLMQ input_ms=%.6f reference_ms=%.6f "
		       "initialization_other_ms=%.6f setup_wall_ms=%.6f\n",
		       bench_input_ms, (double)cpu_time, setup_wall - bench_input_ms - cpu_time, setup_wall);
	}
	const int repeats = g_benchmark.enabled ? bench_env_int("BENCH_REPEATS", 1, 1, 10000) : 1;
	const int warmups = g_benchmark.enabled ? bench_env_int("BENCH_WARMUPS", 0, 0, 10000) : 0;
	int failures = 0;
	VALUE_TYPE *node_data = (VALUE_TYPE *)malloc(g.nnodes * sizeof(VALUE_TYPE));
	for (int sample = 0; sample < repeats + warmups; ++sample) {
	g_benchmark.ready = g_benchmark.finished = 0;
	g_benchmark.start = g_benchmark.end = 0;
	const double query_start = mlmq_bench_ms();

	// 每卡独立跑（并发启动，避免串行等待跨卡死锁）
	struct timeval tv_gpu_begin, tv_gpu_end;
	gettimeofday(&tv_gpu_begin, NULL);

	std::thread th[MAX_GPU];
	for (int i = 0; i < n_gpu; i++)
		th[i] = std::thread(sssp_run_adaptive, i, src, info);
	for (int i = 0; i < n_gpu; i++)
		th[i].join();

	gettimeofday(&tv_gpu_end, NULL);
	float gpu_time = duration(tv_gpu_begin, tv_gpu_end) / 1000.0;

	// 收集每卡归属区间的距离，拼回全量
	const double collection_start = mlmq_bench_ms();
	for (int i = 0; i < n_gpu; i++)
		sssp_cp_data(i, node_data);
	const double collection_ms = mlmq_bench_ms() - collection_start;
	const double query_wall_ms = mlmq_bench_ms() - query_start;
	printf("GPU multi-GPU wall time %.2f ms\n", gpu_time);

	int show_n = 30;
	if (g.nnodes < show_n) show_n = g.nnodes;
	printf("gpu\n");
	for (int i = 0; i < show_n; i++)
#ifdef TYPE_INT
		printf("%d ", node_data[i]);
#else
		printf("%.3f ", node_data[i]);
#endif
	printf("\ncpu\n");
	for (int i = 0; i < show_n; i++)
#ifdef TYPE_INT
		printf("%d ", node_data_base[i]);
#else
		printf("%.3f ", node_data_base[i]);
#endif
	printf("\n");

	int checkres = node_data_check(g.nnodes, node_data, node_data_base);
#ifdef TYPE_INT
	const char *audit_env = std::getenv("MLMQ_FINAL_AUDIT");
	if (audit_env && (std::string(audit_env)=="all" ||
	                 (checkres && std::string(audit_env)=="failure")))
		sssp_audit_final(n_gpu,g.nnodes,g.row_start,g.edge_dst,g.edge_data,node_data,node_data_base);
#endif
	failures += checkres != 0;
	if (!checkres) printf("mlmq sssp correct!\n");
	if (g_benchmark.enabled) {
		g_benchmark.require(g_benchmark.finished == n_gpu, "missing solver participation");
		printf("BENCH algorithm=MLMQ gpu_count=%d source=%d repeat=%d warmup=%d "
		       "queue=%s solve_ms=%.6f collection_ms=%.6f query_wall_ms=%.6f "
		       "prepare_ms=%.6f post_solve_ms=%.6f correct=%d\n",
		       n_gpu, src, sample, sample < warmups, fixed_queue ? fixed_queue : "auto",
		       g_benchmark.end - g_benchmark.start, collection_ms, query_wall_ms,
		       g_benchmark.start - query_start, collection_start - g_benchmark.end, !checkres);
#if (L3_TIMING_DIAG == true)
		sssp_report_timing(n_gpu, sample, sample < warmups);
#endif
	}
	}
	free(node_data);

	for (int i = 0; i < n_gpu; i++)
		sssp_release_l3(i);
	return failures ? 1 : 0;
}
