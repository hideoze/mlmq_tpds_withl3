#include "common.h"
#include "supplement_oracle.h"
#include "csr_graph.h"
#include "graph_info.h"
//#include "ml_queue.cuh"
#include "paras.h"
#include <queue>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <unistd.h>

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

extern int sssp_init(int m_in, int nnz_in, int *RowPtr_in, int *ColIdx_in, VALUE_TYPE *edge_data_in);
extern int sssp_run_adaptive(int src, graph_info info);
extern int sssp_cp_data(VALUE_TYPE **node_data_h);

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

int main(int argc, char *argv[]) {

	cudaSetDevice(0);

	// argument parsing
	char *input_name;
	if (parse_args(argc, argv, input_name) != 0) return -1;

	printf("Graph name: %s\n", input_name);

	float avg_degree, avg_weight;

	CSRGraphTy g, gg;
	g.read(input_name, avg_degree, avg_weight);

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

	int src = getenv("BENCH_SOURCE") ? atoi(getenv("BENCH_SOURCE")) : 0;
	if (src < 0 || src >= g.nnodes) return 2;

	// sequential run
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

	// parallel run	
	g.copy_to_gpu(gg);

	printf("GPU begin!\n");
	
	sssp_init(gg.nnodes, gg.nedges, gg.row_start, gg.edge_dst, gg.edge_data);

	sssp_run_adaptive(src, info);

	printf("GPU done!\n");

	VALUE_TYPE *node_data;
	sssp_cp_data(&node_data);

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
	if (!checkres) printf("mlmq sssp correct!\n");

	return checkres ? 2 : 0;
}
