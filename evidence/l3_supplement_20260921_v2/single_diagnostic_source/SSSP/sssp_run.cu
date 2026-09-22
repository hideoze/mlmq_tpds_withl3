#include "sssp.cuh"
#include <vector>
#include "supplement_capacity.h"
#include "supplement_oracle.h"
__device__ unsigned long long g_dq_clamp;
#include "ml_queue.cuh"
#include <cub/cub.cuh>
#include <chrono>
#include <cstdlib>

extern VALUE_TYPE *node_data_base;
static double baseline_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void baseline_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "NO_L3 CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

int m, nnz;
int *RowPtr, *ColIdx;
VALUE_TYPE *edge_data;

// device node data ptr on host
VALUE_TYPE *node_data;
// device node data ptr on device
__device__ VALUE_TYPE *node_data_dev;
// indicate exiting signals;
int *global_exit;

#define work_count_type float

__device__ __forceinline__ bool node_struct :: operator<(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist < b.dist;
#else
    return node_data_dev[this->id] < node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator>(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist > b.dist;
#else
    return node_data_dev[this->id] > node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator<=(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist <= b.dist;
#else
    return node_data_dev[this->id] <= node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator>=(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist >= b.dist;
#else
    return node_data_dev[this->id] >= node_data_dev[b.id];
#endif
}

__device__ __forceinline__ VALUE_TYPE node_struct :: get_data()
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist;
#else
    return node_data_dev[this->id];
#endif
}

__device__ __forceinline__ bool node_struct :: filter()
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist > node_data_dev[this->id];
#else
    return false;
#endif
}

__host__ __device__ node_struct& node_struct :: operator=(const int id_in)
{
    id = id_in;
#if (USE_DIST_IN_STRUCT == true)
    dist = DIST_MAX;
#endif
    return *this;
}

__host__ __device__ node_struct :: node_struct(int id_in, VALUE_TYPE dist_in)
{
    id = id_in;
#if (USE_DIST_IN_STRUCT == true)
    dist = dist_in;
#endif
}

__host__ __device__ node_struct :: node_struct()
{}

// L0 vector queue
template <typename QUEUE_TYPE>
__device__ __forceinline__ void check_out_buffer(NODE_TYPE *node_out, int &node_out_num, 
QUEUE_TYPE mlmq, int block_id, int warp_id, int lane_id, unsigned *debug_time
#if (WORK_COUNT == true)
, int &total_work
#endif
)
{
    const int thresh = node_size;
    // write back for a whole node size
    while (node_out_num > thresh)
    {
        int fill_num = mlq_min(thresh / 2, node_out_num);
        // int fill_num = thresh / 2;

#if (WORK_CLOCK == true)
        if (!lane_id)
        {
            debug_time[1]++;
            debug_time[2]+=fill_num;
        }
#endif

        int write_s = mlmq.write(node_out + node_out_num - fill_num, fill_num, block_id, warp_id, lane_id, debug_time);

        if (write_s == 0)
        {
            node_out_num -= fill_num;
        }

        __syncwarp();
    }
}

template <typename QUEUE_TYPE>
__device__ int simple_process(int m, int nnz, int *RowPtr, int *ColIdx, 
VALUE_TYPE *edge_data, VALUE_TYPE *node_data, NODE_TYPE *node_in, NODE_TYPE *node_out, 
int &node_in_num, QUEUE_TYPE mlmq, int qshm_size, int block_id, int warp_id, int lane_id, unsigned *debug_time
#if (WORK_COUNT == true)
, int &total_work, int &total_comp
#endif
)
{
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
#endif

    int node_out_num = 0;

#if (WORK_CLOCK == true)
    start_time = clock();
#endif
    // __shared__ int node_prefix_buffer[node_size * WARP_NUM_PER_BLOCK];
    // __shared__ int node_id_buffer[node_size * WARP_NUM_PER_BLOCK];

    // each edge is processed by one thread
    for (int i = 0; i < node_in_num; i+=WARP_SIZE)
    {

#if (WORK_CLOCK == true)
        unsigned start_time2, end_time2;
#endif

        int src_v = 0;
        int i_idx = i + lane_id;
        if (i_idx < node_in_num && node_in[i_idx].get_data() <= node_data[node_in[i_idx].id]) // <
        {
            src_v = node_in[i_idx].id;
        }
        int node_len = 0;
        int first_edge = 0;
        
        if (src_v)
        {
            first_edge = RowPtr[src_v - 1];
            node_len = RowPtr[src_v] - first_edge;
        }
#if (WORK_COUNT == true)
        total_comp += node_len;
#endif

        __syncwarp();

        // cooperatively process large vertices
        unsigned large_mask = __ballot_sync(FULL_MASK, node_len >= LARGEV);
        while (large_mask != 0)
        {
            unsigned large_lane = find_ms_bit(large_mask);
            int large_st = __shfl_sync(FULL_MASK, first_edge, large_lane);
            int large_len = __shfl_sync(FULL_MASK, node_len, large_lane);
            int large_v = __shfl_sync(FULL_MASK, src_v, large_lane);

            for (int iter_idx = 0; iter_idx < large_len; iter_idx += WARP_SIZE)
            {
                bool coop_update = false;
                int coop_idx = iter_idx + lane_id;
                int dst_v = 0;
                VALUE_TYPE new_dist = 0;
                if (coop_idx < large_len)
                {
                    dst_v = ColIdx[large_st + coop_idx] + 1;
                    //new_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[large_v]) + edge_data[large_st + coop_idx];
                    new_dist = node_data[large_v] + edge_data[large_st + coop_idx];
                    //VALUE_TYPE old_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[dst_v]);
                    VALUE_TYPE old_dist = node_data[dst_v];

                    if (new_dist < old_dist)
                    {

                        VALUE_TYPE update_res;
#ifdef TYPE_INT
                        update_res = atomicMin(&node_data[dst_v], new_dist);
#else
                        update_res = atomicMin_float(&node_data[dst_v], new_dist);
#endif
                    
                        if (new_dist < update_res)
                        {
                            coop_update = true;

#if (WORK_COUNT == true)
                            total_work++;
#endif
                        }
                    }
                }
                __syncwarp();

                unsigned update_mask = __ballot_sync(FULL_MASK, coop_update);
                // push into node_in
                int update_num = count_bit(update_mask);
                int update_pos = node_out_num + count_bit(set_bits(update_mask, 0, lane_id, 32));
                node_out_num += update_num;
                if (coop_update)
                    node_out[update_pos] = node_struct(dst_v, new_dist);

                __syncwarp();

#if (WORK_CLOCK == true)
                start_time2 = clock();
#endif

#if (WORK_COUNT == true)
                check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time, total_work);
#else
                check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time);
#endif
                __syncwarp();
#if (WORK_CLOCK == true)
                end_time2 = clock();
                debug_time[0] += end_time2 - start_time2;
#endif

            }
            large_mask = set_bits(large_mask, 0, large_lane, 1);
        }
        if (node_len >= LARGEV) node_len = 0;
        __syncwarp();

        // process small vertices with load balance
        extern __shared__ int s[];
        VALUE_TYPE *edge_lane_buffer = (VALUE_TYPE *)(s + qshm_size / sizeof(int));

        int nl_st = warp_id * LARGEV * WARP_SIZE;

        typedef cub::WarpScan<int> WarpScan;
        __shared__ typename WarpScan::TempStorage temp_storage[WARP_NUM_PER_BLOCK];
        int edge_offset, total_edges;
        WarpScan(temp_storage[warp_id]).ExclusiveSum(node_len, edge_offset, total_edges);
        // if (!lane_id && !global_wid) printf("size %d\n", total_edges);

#if (WORK_CLOCK == true)
        // if (!lane_id)
        // {
        //     debug_time[1]++;
        //     debug_time[2]+=total_edges;
        // }
#endif

		//write the target lane storage
		int edge_idx = edge_offset;
		while (edge_idx < edge_offset + node_len) {
			edge_lane_buffer[nl_st + edge_idx] = lane_id;
			edge_idx++;
		}

        __syncwarp();

// #if (WORK_COUNT == true)
//         if (!lane_id)
//             total_work += total_edges;
// #endif

        for (int iter_idx = 0; iter_idx < total_edges; iter_idx += WARP_SIZE)
        {
            edge_idx = iter_idx + lane_id;
            int leader = 0;
            if (edge_idx < total_edges) leader = edge_lane_buffer[nl_st + edge_idx];
            int leader_edge_offset = __shfl_sync(FULL_MASK, edge_offset, leader);
            int leader_first_edge = __shfl_sync(FULL_MASK, first_edge, leader);
            int leader_vertex = __shfl_sync(FULL_MASK, src_v, leader);

#if (WORK_CLOCK == true)
            unsigned start_time2, end_time2;
#endif

            bool coop_update = false;
            int dst_v = 0;
            VALUE_TYPE new_dist = 0;
            if (edge_idx < total_edges)
            {
                int global_idx = edge_idx - leader_edge_offset + leader_first_edge;
                dst_v = ColIdx[global_idx] + 1;
                //new_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[leader_vertex]) + edge_data[global_idx];
                new_dist = node_data[leader_vertex] + edge_data[global_idx];
                //VALUE_TYPE old_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[dst_v]);
                VALUE_TYPE old_dist = node_data[dst_v];

                if (new_dist < old_dist)
                {

                    VALUE_TYPE update_res;
#ifdef TYPE_INT
                    update_res = atomicMin(&node_data[dst_v], new_dist);
#else
                    update_res = atomicMin_float(&node_data[dst_v], new_dist);
#endif
                
                    if (new_dist < update_res)
                    {
                        coop_update = true;

#if (WORK_COUNT == true)
                        total_work++;
#endif
                    }
                }
                
            }

            __syncwarp();

            unsigned update_mask = __ballot_sync(FULL_MASK, coop_update);
            // push into node_in
            int update_num = count_bit(update_mask);
            int update_pos = node_out_num + count_bit(set_bits(update_mask, 0, lane_id, 32));
            node_out_num += update_num;
            if (coop_update)
            {
                node_out[update_pos] = node_struct(dst_v, new_dist);
            }

            __syncwarp();

#if (WORK_CLOCK == true)
            start_time2 = clock();
#endif

#if (WORK_COUNT == true)
            check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time, total_work);
#else
            check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time);
#endif

            __syncwarp();

#if (WORK_CLOCK == true)
            end_time2 = clock();
            debug_time[0] += end_time2 - start_time2;
#endif
        }
    }

#if (WORK_CLOCK == true)
    start_time = clock();
#endif

    while (node_out_num > 0)
    {
        int fill_num = node_out_num;

#if (WORK_CLOCK == true)
        if (!lane_id)
        {
            debug_time[1]++;
            debug_time[2]+=fill_num;
        }
#endif

        int write_s = mlmq.write(node_out, fill_num, block_id, warp_id, lane_id, debug_time);

        if (write_s == 0)
        {
            node_out_num -= fill_num;
        }
    }
    node_in_num = 0;

#if (WORK_CLOCK == true)
    end_time = clock();
    debug_time[0] += end_time - start_time;
#endif

    return 0;
}

template <typename QUEUE_TYPE>
__global__ void 
work_block_kernel(int m, int nnz, int *RowPtr, int *ColIdx, VALUE_TYPE *edge_data, VALUE_TYPE *node_data, int src,
QUEUE_TYPE mlmq, mlmq_setup setup, int qshm_size, int nshm_size, int *global_exit, work_count_type *global_work_count, int *global_comp_count, int *profile)
{

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    if (tid >= THREAD_NUM_PER_BLOCK + WARP_SIZE) return;

    __shared__ bool local_exit;

    extern __shared__ int s[];
    NODE_TYPE *node_buf = (NODE_TYPE*)(s + qshm_size / sizeof(int));

    NODE_TYPE *node_in = node_buf + 2 * node_size * wid;
    NODE_TYPE *node_out = node_buf + 2 * node_size * WARP_NUM_PER_BLOCK + wid * node_size * 2;

    // if (!bid && !tid) printf("begin %d", bid);

    if (tid < THREAD_NUM_PER_BLOCK)
    { 
        mlmq.init_device(bid, wid, lane_id, setup);
    }
    else
    {
        if (!lane_id)
            local_exit = false;
    }

    __syncthreads();

    // Local manager
    if (tid >= THREAD_NUM_PER_BLOCK)
    {
        while (*global_exit == 0)
        {
            mlmq.update_local_info(lane_id);
            __threadfence();
        }
        local_exit = true;
        __threadfence();
        return;
    }

unsigned debug_time[3];
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
    unsigned read_time = 0, process_time = 0;
    unsigned total_time = 0;
    for (int i = 0; i < 3; i++)
        debug_time[i] = 0;
#endif

    if (global_wid == 0)
    {
        if (!lane_id) node_data[src + 1] = 0;
        node_in[0] = node_struct(src + 1, 0);
        int write_num = 1;
        __syncwarp();
        mlmq.write_through(node_in, write_num, bid, wid, lane_id, debug_time);
        
        if (!lane_id){
            *(mlmq.run_begin) = 1;
            // printf("write through\n");
        }
        __threadfence();
    }
    else
    {
        while (!*(mlmq.run_begin))
        {
            __threadfence();
        }
    }

    // nodes that are already in node_in buffer
    int node_in_num = 0;

    int total_work = 0;
    int total_comp = 0;
    int on_the_fly_num = 0;

#if (WORK_CLOCK == true)
        unsigned start_time2 = clock();
#endif

    // local_exit: true for end
    while (!local_exit)
    {
#if (WORK_CLOCK == true)
        start_time = clock();
#endif

        // if (!global_wid && !lane_id)
        //     printf("before mlmq reading data: fly %d size %d\n", on_the_fly_num, mlmq.get_global_queue_size());

        if (node_in_num == 0)
        {  

            mlmq.read(node_in, node_in_num, on_the_fly_num, bid, wid, lane_id, debug_time);
            // if (!lane_id && node_in_num > 0)
            //     printf("%d read out %d\n", global_wid, node_in_num);
            // if(!lane_id) printf("mlmq node_in %d on_the_fly %d\n", node_in_num, on_the_fly_num);
            // if (on_the_fly_num && node_in_num == 0)//&& node_in_num == 0
            // {
            //     if (!lane_id)
            //     {
            //         mlmq.update_done(on_the_fly_num);
            //         on_the_fly_num = 0;
            //     }
            // }
        }

        // if ( !lane_id)
        // printf("after mlmq reading data num %d fly %d size %d\n", node_in_num, on_the_fly_num, mlmq.get_global_queue_size());
        // if(!lane_id) printf("on_the_fly_num %d\n", on_the_fly_num);
        __syncwarp();
#if (WORK_CLOCK == true)
        end_time = clock();
        read_time += end_time - start_time;
        start_time = clock();
#endif

        if (node_in_num > 0)
        {
#if (WORK_COUNT == true)
            simple_process<QUEUE_TYPE>(m, nnz, RowPtr, ColIdx, edge_data, node_data, 
            node_in, node_out, node_in_num, mlmq, qshm_size + nshm_size, bid, wid, lane_id, debug_time, total_work, total_comp);
#else
            simple_process<QUEUE_TYPE>(m, nnz, RowPtr, ColIdx, edge_data, node_data,
            node_in, node_out, node_in_num, mlmq, qshm_size + nshm_size, bid, wid, lane_id, debug_time);
#endif
       }

#if (WORK_CLOCK == true)
        end_time = clock();
        process_time += end_time - start_time;
#endif

        // if (!lane_id)
        //     printf("After processing fly %d local_size %d global_size %d\n", on_the_fly_num, mlmq.get_local_queue_size(wid),
        //     mlmq.get_global_queue_size());
    //    __syncwarp();
    //    if(!lane_id) printf("on_the_fly_num %d local queue %d\n", on_the_fly_num, mlmq.get_local_queue_size(wid));
        // __syncwarp();
        if (on_the_fly_num && mlmq.get_local_queue_size(wid) == 0)
        {
            if (!lane_id)
            {
                mlmq.update_done(on_the_fly_num);
                on_the_fly_num = 0;
            }
        }

        // if(!lane_id){
        //     // printf("update done");
        //     if (on_the_fly_num && mlmq.get_local_queue_size(wid) == 0){
        //         // printf("update done %d\n", on_the_fly_num);
        //         mlmq.update_done(on_the_fly_num);
        //         on_the_fly_num = 0;
        //     }
        // }
        __syncwarp();
        // printf("%d still running\n", lane_id);
    }

#if (WORK_CLOCK == true)
        total_time += clock() - start_time2;
#endif

#if (PROFILE_COUNT == true)
    if (!lane_id)
    {
        atomicAdd(profile, (debug_time[0] + read_time) / WORK_WARP_NUM);
        atomicAdd(profile + 1, total_time / WORK_WARP_NUM);
    }
#endif

#if (WORK_COUNT == true)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        total_work += __shfl_down_sync(0xffffffff, total_work, offset);
        total_comp += __shfl_down_sync(0xffffffff, total_comp, offset);
    }

    if (!lane_id)
    {
        atomicAdd(global_work_count, total_work);
        atomicAdd(global_comp_count, total_comp);
    }
#endif

#if (WORK_CLOCK == true)
    // Performance profiling
    // Notice: When profiling is enabled, performance will decrease due to calls to printf
    if (!global_wid && !lane_id)
    {
        printf("average write back granularity %.3f\n", 1.0 * debug_time[2] / debug_time[1]);
    }
#endif

}

template <typename QUEUE_TYPE>
__global__ void manage_block_kernel(int m, int nnz, int *RowPtr, int *ColIdx, int src, QUEUE_TYPE mlmq, int *global_exit)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;

    __shared__ int manager_end;
    if (!local_wid && !lane_id)
    {
        manager_end = 0;
        while (!*(mlmq.run_begin))
        { __threadfence(); }
    }
    __syncwarp();

    __syncthreads();

    // if (!lane_id)
    //     printf("manager begin\n");

    if (local_wid == 0)
    {
        while (true)
        {
            // if (!lane_id) printf("size %d\n", mlmq.get_global_queue_size());

            if (mlmq.get_global_queue_size() == 0)
                break;
            __threadfence();

            // if (!lane_id) printf("size %d\n", mlmq.get_global_queue_size());
        }

        manager_end = 1;
        *global_exit = 1;
        __threadfence();

    }
    // l2 manage warp
    else if (local_wid <= mlmq.manage_warp_num())
    {
        int vec_id = local_wid - 1;
        while (!manager_end)
        {
            mlmq.l2_manager(vec_id, lane_id);
            __threadfence();
        }
    }

    __syncthreads();
}

int sssp_init(int m_in, int nnz_in, int *RowPtr_in, int *ColIdx_in, VALUE_TYPE *edge_data_in)
{
    m = m_in;
    nnz = nnz_in;

    RowPtr = RowPtr_in;
    ColIdx = ColIdx_in;
    edge_data = edge_data_in;
    //node_data = node_data_in;
    cudaMalloc(&node_data, sizeof(VALUE_TYPE) * (m + 1));
    
    VALUE_TYPE init_max = DIST_MAX;

    // VALUE_TYPE node_data_h[m + 1];
    VALUE_TYPE *node_data_h= new VALUE_TYPE[m + 1];
    for (int i = 0; i <= m; i++)
        node_data_h[i] = init_max;
    cudaMemcpy(node_data, node_data_h, sizeof(VALUE_TYPE) * (m + 1), cudaMemcpyHostToDevice);
    //cudaMemset(node_data + 1, 0, sizeof(VALUE_TYPE) * m);
    delete[] node_data_h;

    //printf("node_data %d\n", node_data);
    //cudaMemcpy(&node_data_dev, &node_data, sizeof(VALUE_TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(node_data_dev, &node_data, sizeof(VALUE_TYPE*));

    cudaMalloc(&global_exit, sizeof(int));
    cudaMemset(global_exit, 0, sizeof(int));

    cudaDeviceSynchronize();

    return 0;
}

int sssp_re_init()
{
    VALUE_TYPE init_max;
    if (typeid(VALUE_TYPE) == typeid(int))
        init_max = INT_MAX;
    else if (typeid(VALUE_TYPE) == typeid(float))
        init_max = FLT_MAX;
    else if (typeid(VALUE_TYPE) == typeid(double))
        init_max = DBL_MAX;
    else
    {
        printf("VALUE TYPE error!\n");
        return -1;
    }

    // VALUE_TYPE node_data_h[m + 1];
    VALUE_TYPE *node_data_h = new VALUE_TYPE[m + 1];
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    //     return EXIT_FAILURE;
    // }

    for (int i = 0; i <= m; i++)
        node_data_h[i] = init_max;
    cudaMemcpy(node_data, node_data_h, sizeof(VALUE_TYPE) * (m + 1), cudaMemcpyHostToDevice);

    cudaMemset(global_exit, 0, sizeof(int));
    delete[] node_data_h;

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    //     return EXIT_FAILURE;
    // }

    cudaDeviceSynchronize();
    return 0;
}

int sssp_cp_data(VALUE_TYPE **node_data_h)
{
    *node_data_h = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    cudaMemcpy(*node_data_h, node_data + 1, sizeof(VALUE_TYPE) * m, cudaMemcpyDeviceToHost);

    return 0;
}

template <typename QUEUE_TYPE>
void kernel_adaptive(int src, mlmq_setup setup)
{
    NODE_TYPE init_limits = node_struct(0, DIST_MAX);

    work_count_type *global_work_count = nullptr;
    int *global_comp_count = nullptr;
    int *profile = nullptr;
#if (WORK_COUNT == true)
    cudaMalloc(&global_work_count, sizeof(work_count_type));
    cudaMemset(global_work_count, 0, sizeof(work_count_type));
    cudaMalloc(&global_comp_count, sizeof(int));
    cudaMemset(global_comp_count, 0, sizeof(int));
    cudaMalloc(&profile, 2 * sizeof(int));
    cudaMemset(profile, 0, 2 * sizeof(int));
#endif

    cudaDeviceSynchronize();

    QUEUE_TYPE mlmq;
    mlmq.init_host(GPU_MEMORY, init_limits, setup);

    int qshm_size = (mlmq.get_shm_size() + 1023) / 1024 * 1024;
#if (node_size < WARP_SIZE)
    int nshm_size = WARP_SIZE * WARP_NUM_PER_BLOCK * 4 * sizeof(NODE_TYPE);
#else
    int nshm_size = node_size * WARP_NUM_PER_BLOCK * 4 * sizeof(NODE_TYPE);
#endif
    int rshm_size = LARGEV * WARP_SIZE * WARP_NUM_PER_BLOCK * sizeof(int);

    const int warmups = getenv("BENCH_WARMUPS") ? atoi(getenv("BENCH_WARMUPS")) : 2;
    const int repeats = getenv("BENCH_REPEATS") ? atoi(getenv("BENCH_REPEATS")) : 3;
    if (warmups < 0 || repeats < 1) std::exit(2);
    int REPEAT_TIME = warmups + repeats;
    for (int repeat = 0; repeat < REPEAT_TIME; repeat++)
    {
        const double query_start = baseline_ms();
        sssp_re_init();
        mlmq.reinit_host(GPU_MEMORY, init_limits, setup);
        cudaEvent_t start, stop;
        float elapsedTime = 0.0;

        int total_shm_size = (qshm_size + nshm_size + rshm_size + 1023) / 1024 * 1024;
        // Load BOTH persistent kernels before either starts. Lazy-loading the
        // manager after work starts can synchronize against never-ending work.
        baseline_cuda(cudaFuncSetAttribute(work_block_kernel<QUEUE_TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, total_shm_size));
        cudaFuncAttributes manager_attributes;
        baseline_cuda(cudaFuncGetAttributes(&manager_attributes, manage_block_kernel<QUEUE_TYPE>));

        struct cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, 0);

        cudaDeviceSynchronize();

        printf("shm_size %d qshm_size %d nshm_size %d rshm_size %d\n", total_shm_size, qshm_size, nshm_size, rshm_size);

        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        int work_block_num = dev_prop.multiProcessorCount - 1;
        // int work_block_num = 1;

        cudaStream_t s1;
        baseline_cuda(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        cudaStream_t s2;
        baseline_cuda(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        baseline_cuda(cudaGetLastError());
        const double solve_start = baseline_ms();
        work_block_kernel<QUEUE_TYPE><<<work_block_num, ALIGN_THREAD_PER_BLOCK + WARP_SIZE, total_shm_size, s1>>>(m, nnz, RowPtr, ColIdx, edge_data, node_data,
        src, mlmq, setup, qshm_size, nshm_size, global_exit, global_work_count, global_comp_count, profile);
        baseline_cuda(cudaGetLastError());

        int manage_thread_num = WARP_SIZE * (1 + mlmq.manage_warp_num() + 1);

        manage_block_kernel<QUEUE_TYPE><<<1, manage_thread_num, 0, s2>>>(m, nnz, RowPtr, ColIdx, src, mlmq, global_exit);
        baseline_cuda(cudaGetLastError());

        baseline_cuda(cudaDeviceSynchronize());
        const double solve_end = baseline_ms();
        supplement_capacity(mlmq.q2,0);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("Elapse time: %.2f ms\n", elapsedTime);
        VALUE_TYPE *actual = nullptr;
        sssp_cp_data(&actual);
        baseline_cuda(cudaGetLastError());
        const double query_end = baseline_ms();
        bool correct = true;
        for (int i = 0; i < m; ++i) if (actual[i] != node_data_base[i]) {
            fprintf(stderr, "Error at node %d: GPU=%d CPU=%d\n", i, int(actual[i]), int(node_data_base[i]));
            correct = false; break;
        }
        correct = !supplement_oracle_check(actual,m) && correct;
        free(actual);
        printf("BENCH algorithm=MLMQ baseline=original_no_l3 gpu_count=1 source=%d repeat=%d warmup=%d queue=L1SLF_L2DQ solve_ms=%.6f query_wall_ms=%.6f correct=%d\n",
               src, repeat, repeat < warmups, solve_end-solve_start, query_end-query_start, int(correct));
        fflush(stdout);
        if (!correct) std::exit(2);
        baseline_cuda(cudaStreamDestroy(s1));
        baseline_cuda(cudaStreamDestroy(s2));
        baseline_cuda(cudaEventDestroy(start));
        baseline_cuda(cudaEventDestroy(stop));

#if (WORK_COUNT == true)
        work_count_type global_work_count_host = 0;
        cudaMemcpy(&global_work_count_host, global_work_count, sizeof(work_count_type), cudaMemcpyDeviceToHost);
        if (typeid(global_work_count_host) == typeid(float))
            printf("Float total work count %.2f\n", global_work_count_host);
        else
            printf("Total work count %d\n", global_work_count_host);
#endif

#if (WORK_CLOCK == true)
        int global_comp_count_host = 0;
        int profile_host[2];
        cudaMemcpy(&global_comp_count_host, global_comp_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(profile_host, profile, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Total comp count %d\n", global_comp_count_host);
        float profile_total = profile_host[1];
        printf("total time %.4f read proportion %.4f process proportion %.4f\n", 
        profile_total / FRE, profile_host[0] / profile_total, (profile_total - profile_host[0]) / profile_total);
#endif
    }

}

int sssp_run_adaptive(int src, graph_info info)
{

    mlmq_setup setup;

    setup.init_setup();
    //setup.init_setup_adaptive(info);
    setup.type = L1SLF_L2DQ;
    setup.s_l2_delta = getenv("BENCH_DELTA") ? atoi(getenv("BENCH_DELTA")) : 200000;
    if (setup.s_l2_delta <= 0) std::exit(2);
    printf("NO_L3_CONFIG workers=%d delta=%d queue=L1SLF_L2DQ\n", THREAD_NUM_PER_BLOCK, setup.s_l2_delta);

    switch (setup.type)
    {
        case L1N_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1V_L2DQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1NF_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1V_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1N_L2V:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1NF_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1FQ_L2DQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue_new<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1V_L2PQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1NF_L2PQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1FQ_L2PQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1V_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1NF_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1FQ_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1HQ_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_hop_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1HQ_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_hop_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1FQ_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1N_L2MV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_multi_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1N_L2BV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_batch_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1V_L2BV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_batch_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1SLF_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_SLF_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(src, setup);
            break;
        case L1SLF_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_SLF_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(src, setup);
            break;
        default:
            printf("MLMQ type not implemented!\n");
    }

    return 0;
}
