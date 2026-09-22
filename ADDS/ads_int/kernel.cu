/*  -*- mode: c++ -*-  */
#include <cuda.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>
#include <cassert>
#include <cub/cub.cuh>
#include "common.h"
#include "csr_graph.h"
#include "support.h"
#include "wl.h"
#include <sys/time.h>
#include <unistd.h>
#include <queue>
#include <chrono>
#include <vector>
#include "../../SSSP/benchmark.h"

#ifndef RUN_LOOP
#define RUN_LOOP 8
#endif
static_assert(RUN_LOOP > 0, "RUN_LOOP must be positive");

static double bench_ms() {
	return std::chrono::duration<double, std::milli>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void bench_cuda(cudaError_t status) {
	if (status != cudaSuccess) {
		fprintf(stderr, "ADDS CUDA error: %s\n", cudaGetErrorString(status));
		exit(2);
	}
}

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

#define MAX_EDGE_DATA INT_MAX

#define TB_SIZE 512
int CUDA_DEVICE = 0;
int start_node = 0;
char *INPUT, *OUTPUT;

#define VALUE_TYPE edge_data_type

__global__ void kernel(CSRGraph graph, int src) {
	unsigned tid = thread_id_x() + block_id_x() * block_dim_x();
	unsigned nthreads = block_dim_x() * grid_dim_x();

	index_type node_end;
	node_end = (graph).nnodes;
	for (index_type node = 0 + tid; node < node_end; node += nthreads) {
		graph.node_data[node] = (node == src) ? 0 : INF;
	}
}
#define HOR_EDGE 16
__global__ void
__launch_bounds__(896, 1)
sssp_kernel(CSRGraph graph, worklist wl, unsigned* work_count, unsigned *cmp_count, unsigned *debug_time) {

	wl.init_regular();

	unsigned tid = thread_id_x() + block_id_x() * block_dim_x();
	const int warpid = thread_id_x() / 32;
	const int laneid = thread_id_x() % 32;
	unsigned total_work = 0;
	__shared__ unsigned char leader_lane_tb_storage[TB_SIZE * 32];
	unsigned char * leader_lane_storage = &(leader_lane_tb_storage[warpid * 32 * 32]);

	unsigned tb_coop_threshold = max(32 * TB_COOP_MUL, (graph.nedges / graph.nnodes) * TB_COOP_MUL);

	uint wl_offset = get_lane_id();

#define USE_BUFFER false
#define BUFFER_TYPE_VECTOR
#define LOCAL_SIZE 64
#define SHOW_GRA false
#define COUNT_CLOCK false

#if (USE_BUFFER == true)
	__shared__ unsigned vertex_buf[LOCAL_SIZE * TB_SIZE / 32];
	int buf_offset = warpid * LOCAL_SIZE;
	int buf_size = 0;
#ifdef BUFFER_TYPE_FILTER
	__shared__ edge_data_type buf_base[TB_SIZE / 32];
#endif
#endif

#if (COUNT_CLOCK == true)
	unsigned clock_process = 0;
	unsigned clock_queue = 0;
	unsigned start_time, end_time;
	unsigned start_time2;
#endif

	unsigned src_bag_id = 0;
	//get work
	unsigned long long m_assignment;
	unsigned num = 0;

	int total_edge_num = 0;
	int total_edge_count = 0;

	while (1) {

		int local_cmp = 0;

#if (COUNT_CLOCK == true)
		start_time2 = clock();
		clock_queue = 0;
#endif
		

		int m_first_edge;
		int m_size = 0;
		int m_vertex;
#if (USE_BUFFER == true)
		if (buf_size > 0)
		{
			int buf_alter = (buf_size < 32)? buf_size : 32;
			//work in buffer
			if (wl_offset < buf_size)
			{
				m_vertex = vertex_buf[buf_offset + buf_size - buf_alter + wl_offset];
				m_first_edge = graph.row_start[m_vertex];
				m_size = graph.row_start[m_vertex + 1] - m_first_edge;
			}
			
			buf_size -= buf_alter;
		}
		else
#endif
		{
#if (COUNT_CLOCK == true)
			start_time = clock();
#endif
			//this does tb coop processing also
			num = wl.get_assignment(graph, m_assignment, warpid, work_count, total_work);
			unsigned ptr = agm_get_real_ptr(m_assignment);
			src_bag_id = agm_get_bag_id(m_assignment);

			if (wl_offset < num) {
				m_vertex = wl.pop_work(ptr + wl_offset);
				if (m_vertex < graph.nnodes) {
					m_first_edge = graph.row_start[m_vertex];
					m_size = graph.row_start[m_vertex + 1] - m_first_edge;
				}
			}

#if (COUNT_CLOCK == true)
			clock_queue += clock() - start_time;
#endif
		}

		local_cmp += m_size;

		//do tb coop assign, assign just one if multiple
		unsigned process_mask;
		process_mask = __ballot_sync(FULL_MASK, m_size >= tb_coop_threshold);
		unsigned tb_coop_lane = find_ms_bit(process_mask);
		if (tb_coop_lane != NOT_FOUND) {
			wl.tb_coop_assign(m_vertex, m_first_edge, m_assignment, m_size, tb_coop_lane);
		}

		//do high degree vertex
		process_mask = __ballot_sync(FULL_MASK, m_size >= HOR_EDGE);
		while (process_mask != 0) {
			unsigned leader = find_ms_bit(process_mask);
			int leader_first_edge = __shfl_sync(FULL_MASK, m_first_edge, leader);
			int leader_size = __shfl_sync(FULL_MASK, m_size, leader);
			int leader_vertex = __shfl_sync(FULL_MASK, m_vertex, leader);
			for (int offset = get_lane_id(); offset < leader_size; offset += 32) {
				index_type edge = leader_first_edge + offset;
				index_type dst = graph.edge_dst[edge];
				edge_data_type wt = graph.edge_data[edge];
				node_data_type new_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[leader_vertex])) + wt;
				node_data_type dst_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[dst]));
				if (dst_dist > new_dist) {
					if (atomicMin(&(graph.node_data[dst]), new_dist) > new_dist) {
#if (COUNT_CLOCK == true)
						start_time = clock();
#endif
						unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, new_dist);
						wl.push_work(dst_bag_id, dst);
#if (COUNT_CLOCK == true)
						clock_queue += clock() - start_time;
#endif
						total_work++;
					}
				}
			}
			process_mask = set_bits(process_mask, 0, leader, 1);
			if (!laneid)
				atomicAdd(cmp_count, leader_size);
		}
		if (m_size >= HOR_EDGE) {
			m_size = 0;
		}

		//do low degree vertex
		__syncwarp();
		int warp_offset;
		int warp_total;
		typedef cub::WarpScan<int> WarpScan;
		__shared__ typename WarpScan::TempStorage temp_storage[TB_SIZE / 32];
		WarpScan(temp_storage[warpid]).ExclusiveSum(m_size, warp_offset, warp_total);

		//write the target lane storage
		int write_idx = warp_offset;
		int write_end = warp_offset + m_size;
		while (write_idx < write_end) {
			leader_lane_storage[write_idx] = (unsigned char) get_lane_id();
			write_idx++;
		}

		total_edge_num += warp_total;
		total_edge_count++;

		__syncwarp();

		for (int read_idx_start = 0; read_idx_start < warp_total; read_idx_start += 32) {
			int m_read_idx = read_idx_start + get_lane_id();
			int leader = (int) leader_lane_storage[m_read_idx];
			int leader_warp_offset = __shfl_sync(FULL_MASK, warp_offset, leader);
			int leader_first_edge = __shfl_sync(FULL_MASK, m_first_edge, leader);
			int leader_vertex = __shfl_sync(FULL_MASK, m_vertex, leader);

			bool update_new = false;

			index_type dst = 0;
			
			if (m_read_idx < warp_total) {
				int offset = m_read_idx - leader_warp_offset;
				index_type edge = leader_first_edge + offset;
				dst = graph.edge_dst[edge];
				edge_data_type wt = graph.edge_data[edge];
				node_data_type new_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[leader_vertex])) + wt;
				node_data_type dst_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[dst]));
				if (dst_dist > new_dist) {
					if (atomicMin(&(graph.node_data[dst]), new_dist) > new_dist) {
						update_new = true;
						total_work++;
					}
				}
			}

#if (USE_BUFFER == true)
	#ifdef BUFFER_TYPE_FILTER
			unsigned cur_delta = get_current_delta();
			if (buf_size == 0)
			{
				if (!wl_offset)
				{
					buf_base[warpid] = MAX_EDGE_DATA;
					buf_base[warpid] = (unsigned)(graph.node_data[dst] / cur_delta) * cur_delta;
				}
				__syncwarp();
			}
			__syncwarp();

			const unsigned buf_delta = 100;
			bool update_local = update_new && graph.node_data[dst] <= buf_base[warpid] + buf_delta;
			bool update_global = update_new && graph.node_data[dst] > buf_base[warpid] + buf_delta;
			if (update_global)
			{
#if (COUNT_CLOCK == true)
				start_time = clock();
#endif
				unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, graph.node_data[dst]);
				wl.push_work(dst_bag_id, dst);
#if (COUNT_CLOCK == true)
				clock_queue += clock() - start_time;
#endif
			}

			unsigned write_mask = __ballot_sync(FULL_MASK, update_local);
			int write_num = count_bit(write_mask);
			int write_buffer_pos = count_bit(set_bits(write_mask, 0, wl_offset, 32));
			if (update_local)
				vertex_buf[buf_offset + buf_size + write_buffer_pos] = dst;
			__syncwarp();
			buf_size += write_num;

			__syncwarp();

			if (buf_size > LOCAL_SIZE / 2)
			{
				unsigned dst = vertex_buf[buf_offset + buf_size - LOCAL_SIZE / 2 + wl_offset];
				unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, graph.node_data[dst]);
				wl.push_work(dst_bag_id, dst);
				buf_size -= LOCAL_SIZE / 2;
			}
			__syncwarp();
	#else
			unsigned write_mask = __ballot_sync(FULL_MASK, update_new);
			int write_num = count_bit(write_mask);
			int write_buffer_pos = count_bit(set_bits(write_mask, 0, wl_offset, 32));
			if (update_new)
				vertex_buf[buf_offset + buf_size + write_buffer_pos] = dst;
			buf_size += write_num;
			__syncwarp();

			if (buf_size > LOCAL_SIZE / 2)
			{
				unsigned dst = vertex_buf[buf_offset + buf_size - LOCAL_SIZE / 2 + wl_offset];
				unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, graph.node_data[dst]);
				wl.push_work(dst_bag_id, dst);
				buf_size -= LOCAL_SIZE / 2;
			}
			__syncwarp();
	#endif
#else
			if (update_new)
			{
#if (COUNT_CLOCK == true)
				start_time = clock();
#endif
				unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, graph.node_data[dst]);
				wl.push_work(dst_bag_id, dst);
#if (COUNT_CLOCK == true)
				clock_queue += clock() - start_time;
#endif
			}
#endif


		}
		__syncwarp();

		if (!laneid)
		{
			atomicAdd(cmp_count, warp_total);
		}

		// if (!tid)
		// {
		// 	printf("current delta %u shifted %u\n", get_current_delta(), get_current_shifted());
		// 	//atomicAdd(cmp_count, warp_total);
		// }

#if (SHOW_GRA == true)
		if (!tid)
		{
			printf("average granularity %.2f\n", 1.0 * total_edge_num / total_edge_count);
			//atomicAdd(cmp_count, warp_total);
		}
#endif

#if (COUNT_CLOCK == true)
		start_time = clock();
#endif
		//free
#if (USE_BUFFER == true)
		if (num > 0 && !buf_size)
#else
		if (num > 0)
#endif
			wl.epilog(m_assignment, num, false);

#if (COUNT_CLOCK == true)
		clock_queue += clock() - start_time;
		clock_process = clock() - start_time2;
		if (!get_lane_id())
		{
			atomicAdd(debug_time, clock_queue / wl.num_warp / wl.num_tb_real);
			atomicAdd(debug_time + 1, clock_process / wl.num_warp / wl.num_tb_real);
		}
#endif
	}
}

__global__ void driver_kernel(int num_tb, int num_threads, worklist wl, CSRGraph gg, uint start_node, unsigned* work_count, unsigned *cmp_count, unsigned *debug_time) {

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	wl_kernel<<<1, 192, 0, s1>>>(wl, start_node);

	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	sssp_kernel<<<num_tb, num_threads, 0, s2>>>(gg, wl, work_count, cmp_count, debug_time);

}

void gg_main_pipe_1_wrapper(CSRGraph& hg, CSRGraph& gg, uint start_node, int num_tb, int num_threads, worklist wl, unsigned* work_count, unsigned *cmp_count, unsigned *debug_time) {
	// gg_main_pipe_1_gpu<<<1,1>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads,cl_curdelta,cl_i, enable_lb);
	//gg_cg_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(
	//		gg, glevel, curdelta, i, DELTA, pipe, cl_curdelta, cl_i, enable_lb);

	driver_kernel<<<1, 1>>>(num_tb, num_threads, wl, gg, start_node, work_count, cmp_count, debug_time);
	bench_cuda(cudaGetLastError());
	bench_cuda(cudaDeviceSynchronize());
}

#define I_TIME 4
#define N_TIME 2
__global__ void profiler_kernel(CSRGraph gg, float* bk_wt, int warp_edge_interval) {

	int tid = thread_id_x() + block_id_x() * block_dim_x();
	int warp_id = tid / 32;
	int num_warp = block_dim_x() * grid_dim_x() / 32;

	typedef cub::BlockReduce<float, 1024> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float thread_data[I_TIME];
	float tb_ave = 0;
	//find ave weight
	tb_ave = 0;
	for (int n = 0; n < N_TIME; n++) {
		for (int i = 0; i < I_TIME; i++) {
			unsigned warp_offset = ((n * I_TIME + i) * num_warp + warp_id) * warp_edge_interval;
			unsigned edge_id = warp_offset + cub::LaneId();
			if (edge_id < gg.nedges) {
				thread_data[i] = (float) gg.edge_data[edge_id];
				//if (!cub::LaneId()) printf("??? %.2f\n", gg.edge_data[edge_id]);
			} else {
				thread_data[i] = 0;
			}
		}
		//do a reduction
		float sum = BlockReduce(temp_storage).Sum(thread_data);
		tb_ave += (sum / 1024 / I_TIME);
	}
	tb_ave = tb_ave / N_TIME;
	//store it back
	if (threadIdx.x == 0) {
		bk_wt[blockIdx.x] = tb_ave;
	}
}

int node_data_check(int m, VALUE_TYPE *node_data, VALUE_TYPE *node_data_base);

int gg_main(CSRGraph& hg, CSRGraph& gg, VALUE_TYPE *reference) {
	const double allocation_start = bench_ms();
	const int repeats = bench_env_int("BENCH_REPEATS", RUN_LOOP, 1, 10000);
	const int warmups = bench_env_int("BENCH_WARMUPS", 0, 0, 10000);
	const int total_runs = repeats + warmups;

	struct cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, CUDA_DEVICE);

	int num_threads = TB_SIZE;

	int num_tb_per_sm;
	int max_num_threads;
	int min_grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_num_threads, sssp_kernel);
	if (num_threads > max_num_threads) {
		printf("error max threads is %d, specified is %d\n", max_num_threads, num_threads);
		fflush(0);
		exit(1);
	}
	printf("max_num_threads is %d\n", max_num_threads);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_tb_per_sm, sssp_kernel, num_threads, 0);
	num_tb_per_sm = 1;
	int num_sm = dev_prop.multiProcessorCount;
	int num_tb = num_tb_per_sm * (num_sm - 1);
	int num_warps = num_threads / 32;
	printf("sm count %u, tb per sm %u, tb size %u, num warp %u, total tb %u\n", num_sm, num_tb_per_sm, num_threads, num_warps, num_tb);

#define WL_SIZE_MUL 1.5f
	unsigned suggest_size = (unsigned)((float)hg.nedges * WL_SIZE_MUL) + (NUM_BAG * BLOCK_SIZE * 8);
	suggest_size = min(suggest_size, 536870912);
	unsigned* work_count;
	cudaMalloc((void **) &work_count, num_tb * num_threads * sizeof(unsigned));
	unsigned* cmp_count;
	cudaMalloc((void **) &cmp_count, sizeof(unsigned));
	unsigned* debug_time;
	cudaMalloc((void **) &debug_time, 2 * sizeof(unsigned));

	worklist wl;
	wl.alloc(num_tb, num_warps, suggest_size, hg.nnodes, hg.nedges);
	printf("alloc done!\n");
	bench_cuda(cudaGetLastError());
	bench_cuda(cudaDeviceSynchronize());
	printf("BENCH_SETUP algorithm=ADDS workspace_ms=%.6f repeats=%d\n",
		bench_ms() - allocation_start, total_runs);
	int failures = 0;

	unsigned long long agg_total_work = 0;
	float agg_time = 0;
	for (int loop = 0; loop < total_runs; loop++) {
		int other_num_tb = num_tb_per_sm * num_sm;
		const double reset_start = bench_ms();

		kernel<<<other_num_tb, 1024>>>(gg, start_node);
		cudaMemset(work_count, 0, num_tb * num_threads * sizeof(unsigned));
		cudaMemset(cmp_count, 0, sizeof(int));
		cudaMemset(debug_time, 0, 2 * sizeof(unsigned));
		wl.reinit();
		bench_cuda(cudaGetLastError());
		bench_cuda(cudaDeviceSynchronize());
		const double reset_ms = bench_ms() - reset_start;
		const double parameter_start = bench_ms();

		float elapsed_time;   // timing variables

		float ave_degree = (float) hg.nedges / (float) hg.nnodes;
		float ave_wt;
		{
			//find delta
			int total_warp = other_num_tb * 1024 / 32;
			int warp_edge_interval = hg.nedges / (total_warp * I_TIME * N_TIME);
			float* host_bk_wt = (float*) malloc(other_num_tb * sizeof(float));
			float* bk_wt;

			cudaMalloc((void **) &bk_wt, other_num_tb * sizeof(float));
			profiler_kernel<<<other_num_tb, 1024>>>(gg, bk_wt, warp_edge_interval);
			cudaMemcpy(host_bk_wt, bk_wt, other_num_tb * sizeof(float), cudaMemcpyDeviceToHost);

			ave_wt = 0;
			for (int i = 0; i < other_num_tb; i++) {
				ave_wt += host_bk_wt[i];
			}
			ave_wt /= other_num_tb;
			bench_cuda(cudaFree(bk_wt));
			free(host_bk_wt);
		}
		wl.set_param(ave_wt, ave_degree);
		bench_cuda(cudaGetLastError());
		bench_cuda(cudaDeviceSynchronize());
		const double parameter_ms = bench_ms() - parameter_start;

		cudaEvent_t start_event, stop_event;
		bench_cuda(cudaEventCreate(&start_event));
		bench_cuda(cudaEventCreate(&stop_event));
		bench_cuda(cudaEventRecord(start_event, 0));
		bench_cuda(cudaEventSynchronize(start_event));
		const double solve_start = bench_ms();

		gg_main_pipe_1_wrapper(hg, gg, start_node, num_tb, num_threads, wl, work_count, cmp_count, debug_time);
		const double solve_ms = bench_ms() - solve_start;

		bench_cuda(cudaEventRecord(stop_event, 0));
		bench_cuda(cudaEventSynchronize(stop_event));
		bench_cuda(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
		bench_cuda(cudaEventDestroy(start_event));
		bench_cuda(cudaEventDestroy(stop_event));
		// Only distances are output; immutable CSR need not be copied back.
		const double collection_start = bench_ms();
		bench_cuda(cudaMemcpy(hg.node_data, gg.node_data,
			hg.nnodes * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));
		const double collection_ms = bench_ms() - collection_start;
		const double query_wall_ms = bench_ms() - reset_start;
		const int incorrect = node_data_check(hg.nnodes, hg.node_data, reference);
		failures += incorrect;
		printf("BENCH algorithm=ADDS gpu_count=1 source=%d repeat=%d warmup=%d "
			"reset_ms=%.6f parameter_ms=%.6f solve_ms=%.6f collection_ms=%.6f "
			"query_components_ms=%.6f query_wall_ms=%.6f cuda_event_ms=%.6f correct=%d\n",
			start_node, loop, loop < warmups, reset_ms, parameter_ms, solve_ms, collection_ms,
			reset_ms + parameter_ms + solve_ms + collection_ms, query_wall_ms, elapsed_time, !incorrect);

		unsigned* work_count_host = (unsigned*) malloc(num_tb * num_threads * sizeof(unsigned));

		cudaMemcpy(work_count_host, work_count, num_tb * num_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);
		unsigned long long total_work = 0;
		for (int i = 0; i < num_tb * num_threads; i++) {
			total_work += work_count_host[i];
		}
		free(work_count_host);

		agg_time += elapsed_time;
		agg_total_work += total_work;

		int cmp_count_host;
		cudaMemcpy(&cmp_count_host, cmp_count, sizeof(int), cudaMemcpyDeviceToHost);

		unsigned debug_time_host[2];
		cudaMemcpy(&debug_time_host, debug_time, 2 * sizeof(unsigned), cudaMemcpyDeviceToHost);

		printf("%s Measured time for sample = %.6fs\n", hg.file_name, elapsed_time / 1000.0f);
		printf("total work is %llu\n", total_work);
		printf("compare work is %d\n", cmp_count_host);
#if (COUNT_CLOCK == true)
		float FRE = 1.35 * 1000 * 1000;
		float run_total_clock = debug_time_host[1] / FRE;
		float queue_prop = 1.0 * debug_time_host[0] / debug_time_host[1];
		printf("total run time %.4f queue proportion is %.4f process proportion is %.4f\n", 
		run_total_clock, queue_prop, 1 - queue_prop);
#endif
	}

	// Legacy summary includes warmups; formal statistics use BENCH warmup=0 rows only.
	float ave_time = agg_time / total_runs;
	long long unsigned ave_work = agg_total_work / total_runs;
	printf("ADDS summary graph=%s event_mean_ms=%.6f work_mean=%llu failures=%d\n",
		hg.file_name, ave_time, ave_work, failures);
	wl.free();
	cudaFree(work_count);
	cudaFree(cmp_count);
	cudaFree(debug_time);
	return failures ? 1 : 0;

//clean up
}

// struct on host, only for 
class node_struct_base
{
public:
    index_type id = 0;
	VALUE_TYPE dist;

    bool operator<(const node_struct_base& b) const;
    bool operator>(const node_struct_base& b) const;
    bool operator<=(const node_struct_base& b) const;
    bool operator>=(const node_struct_base& b) const;

    node_struct_base& operator=(const index_type id_in) { this->id = id_in; return *this; };

    VALUE_TYPE get_data() { return this->dist; };

    node_struct_base (index_type id_in, VALUE_TYPE dist_in) : id(id_in), dist(dist_in) {};

    node_struct_base () {};
};

bool node_struct_base::operator<(const node_struct_base& b) const { return this->dist < b.dist; }
bool node_struct_base::operator>(const node_struct_base& b) const { return this->dist > b.dist; }
bool node_struct_base::operator<=(const node_struct_base& b) const { return this->dist <= b.dist; }
bool node_struct_base::operator>=(const node_struct_base& b) const { return this->dist >= b.dist; }

void sssp_sequential(int m, int nnz, index_type *RowPtr, index_type *ColIdx, VALUE_TYPE *edge_data, VALUE_TYPE *node_data, index_type *prep, int src, int &total_work)
{
	std::priority_queue<node_struct_base, std::vector<node_struct_base>, std::greater<node_struct_base>> q;
	q.push(node_struct_base(src, 0));

	prep[src] = src;

	int count = 0;

	total_work = 0;

	std::vector<unsigned char> done(m, 0);

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
				q.push(node_struct_base(nid, node_data[tid] + edge_data[j]));
			}
		}
		count++;
	}
}

int node_data_check(int m, VALUE_TYPE *node_data, VALUE_TYPE *node_data_base)
{
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
	if (argc == 1) {
		usage(argc, argv);
		exit(1);
	}
	parse_args(argc, argv);
	const double input_start = bench_ms();
	CSRGraphTy g, gg;
	g.read(INPUT);
	start_node = bench_env_int("BENCH_SOURCE", start_node, 0, g.nnodes - 1);
	const double input_ms = bench_ms() - input_start;
	if (g.nnodes == 0 || start_node < 0 || (unsigned)start_node >= g.nnodes) {
		fprintf(stderr, "ADDS invalid graph/source\n");
		return 2;
	}
	// for (int i = 0; i < g.nedges; i++)
	// 	printf("%.2f ", g.edge_data[i]);
		

	// sequential run
	VALUE_TYPE *node_data_base = (VALUE_TYPE*)malloc(g.nnodes * sizeof(VALUE_TYPE));
	for (int i = 0; i < g.nnodes; i++)
	{
		node_data_base[i] = INT_MAX;
	}
	node_data_base[start_node] = 0;

	index_type *prep = (index_type*)malloc(g.nnodes * sizeof(index_type));

	int total_work = 0;
	struct timeval tv_begin, tv_end;
	gettimeofday(&tv_begin, NULL);
	sssp_sequential(g.nnodes, g.nedges, g.row_start, g.edge_dst, g.edge_data, node_data_base, prep, start_node, total_work);
	gettimeofday(&tv_end, NULL);
	float cpu_time = duration(tv_begin, tv_end) / 1000.0;
	// Reference remains host-only and is never supplied to the solver.
	const double upload_start = bench_ms();
	bench_cuda(cudaSetDevice(CUDA_DEVICE));
	g.copy_to_gpu(gg);
	bench_cuda(cudaDeviceSynchronize());
	printf("BENCH_SETUP algorithm=ADDS input_ms=%.6f device_upload_ms=%.6f "
		"reference_ms=%.6f layout=external\n", input_ms, bench_ms() - upload_start, cpu_time);
	const int run_status = gg_main(g, gg, node_data_base);
	output(g, OUTPUT);

	int show_n = 30;
	if (g.nnodes < show_n) show_n = g.nnodes;
	printf("gpu\n");
	for (int i = 0; i < show_n; i++)
		printf("%d ", g.node_data[i]);
	printf("\ncpu\n");
	for (int i = 0; i < show_n; i++)
		printf("%d ", node_data_base[i]);
	printf("\n");

	int checkres = node_data_check(g.nnodes, g.node_data, node_data_base);
	if (!checkres && !run_status) printf("ADDS sssp correct!\n");
	free(node_data_base);
	free(prep);

	return (checkres || run_status) ? 1 : 0;
}
