
// l2_bgpq_queue
template <typename eletype>
init_status l2_bgpq_queue<eletype> :: host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
{
    batchSize = setup.s_l2_batch_size;
    int node_num = max_size / sizeof(eletype) / batchSize;
    BGPQ_Heap<eletype> bgpq_tmp = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    cudaMalloc(&bgpq, sizeof(BGPQ_Heap<eletype>));
    cudaMemcpy(bgpq, &bgpq_tmp, sizeof(BGPQ_Heap<eletype>), cudaMemcpyHostToDevice);

    cudaMalloc(&read_done, sizeof(int));
    cudaMemset(read_done, 0, sizeof(int));
    cudaMalloc(&write_reserve, sizeof(int));
    cudaMemset(write_reserve, 0, sizeof(int));

    int smemOffset_host[WARP_NUM_PER_BLOCK];
    cudaMalloc(&smemOffset, sizeof(int) * WARP_NUM_PER_BLOCK);
    for (int i = 0; i < WARP_NUM_PER_BLOCK; i++)
    {
        smemOffset_host[i] = mdata.l2_queue_offset + i * mdata.l2_bufsize_perwarp;
    }
    cudaMemcpy(smemOffset, smemOffset_host, sizeof(int) * WARP_NUM_PER_BLOCK, cudaMemcpyHostToDevice);

    return INIT_SUCCESS;
}

template <typename eletype>
init_status l2_bgpq_queue<eletype> :: host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
{
    batchSize = setup.s_l2_batch_size;
    int node_num = max_size / sizeof(eletype) / batchSize;
    BGPQ_Heap<eletype> bgpq_tmp = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    cudaMemcpy(bgpq, &bgpq_tmp, sizeof(BGPQ_Heap<eletype>), cudaMemcpyHostToDevice);

    cudaMemset(read_done, 0, sizeof(int));
    cudaMemset(write_reserve, 0, sizeof(int));

    int smemOffset_host[WARP_NUM_PER_BLOCK];
    for (int i = 0; i < WARP_NUM_PER_BLOCK; i++)
    {
        smemOffset_host[i] = mdata.l2_queue_offset + i * mdata.l2_bufsize_perwarp;
    }
    cudaMemcpy(smemOffset, smemOffset_host, sizeof(int) * WARP_NUM_PER_BLOCK, cudaMemcpyHostToDevice);

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l2_bgpq_queue<eletype> :: read(eletype *node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
    start_time = clock();
#endif

    //printf("%d reading left\n", lane_id);

    // delete items from heap
    int bgpq_status = bgpq->deleteRoot(node_in, read_num, wid, lane_id, debug_time);

#if (WORK_CLOCK == true)
    end_time = clock();
    debug_time[6] += end_time - start_time;
#endif

    __syncwarp();

    if (bgpq_status == true) {

        // if (!lane_id)
        //     printf("before bgpq removing read_num %d size %d\n", read_num, get_queue_size());

        if (read_num >= batchSize)
        {
            bgpq->deleteUpdate(wid, lane_id, smemOffset[wid]);
        }

        // if (!lane_id)
        //     printf("after bgpq removing read_num %d size %d\n", read_num, get_queue_size());

        return READ_SUCCESS;
    }
    else
    {
        return READ_EMPTY;
    }
}

template <typename eletype>
__device__ write_status l2_bgpq_queue<eletype> :: write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
    //if (!lane_id) printf("%d %d\n", wid, smemOffset[wid]);
    for (int i = 0; i < write_num; i += l2_batch_size)
    {
        int tmp_write_num = mlq_min(l2_batch_size, write_num - i);

        bgpq->insertion_shared(node_out + i, tmp_write_num, wid, lane_id, smemOffset[wid], debug_time);
        
        if (!lane_id)
        {
            atomicAdd(write_reserve, tmp_write_num);
        }
        __syncwarp();
    }

    return WRITE_SUCCESS;
}

template <typename eletype>
__device__ init_status l2_bgpq_queue<eletype> :: device_init(int bid, int wid, int lane_id)
{
    return INIT_SUCCESS;
}

template <typename eletype>
__device__ void l2_bgpq_queue<eletype> :: show_bgpq()
{
    int heap_size = *(bgpq->batchCount) * batchSize;
    printf("bgpq heap: size %d\n", heap_size);
    for (int i = batchSize; i < heap_size + batchSize; i++)
    {
        //if (bgpq->heapItems[i] != bgpq->init_limits)
            printf("%d ", bgpq->heapItems[i]);
    }
    printf("\n");
}

template <typename eletype>
__device__ int l2_bgpq_queue<eletype> :: get_queue_size()
{
    return *write_reserve - *read_done;
}

template <typename eletype>
__device__ int l2_bgpq_queue<eletype> :: get_available_queue_size()
{
    // BGPQ exposes no published-vs-reserved cursor pair; preserve the
    // existing credit metric as the compatibility fallback.
    const int available = *write_reserve - *read_done;
    return available > 0 ? available : 0;
}

template <typename eletype>
__device__ void l2_bgpq_queue<eletype> :: manager_run(int wid, int lane_id)
{
    //printf("read_done, write_reserve %d %d\n", *read_done, *write_reserve);
}

template <typename eletype>
__device__ void l2_bgpq_queue<eletype> :: update_done(int on_the_fly_num)
{
    atomicAdd(read_done, on_the_fly_num);
}

template <typename eletype>
__device__ void l2_bgpq_queue<eletype> :: update_local_info(int lane_id)
{
}

template <typename eletype>
__host__ __device__ int l2_bgpq_queue<eletype> :: manage_warp_num()
{
    return 1;
}

// l2_multi_queue
template <typename eletype>
init_status l2_multi_queue<eletype> :: host_init(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
{
    batchSize = setup.s_l2_batch_size;
    int node_num = max_size / sizeof(eletype) / batchSize;

    NUM_PQ = L2_MQ_NUM;

    BGPQ_Heap<eletype> *bgpq_tmp = (BGPQ_Heap<eletype>*)malloc(sizeof(BGPQ_Heap<eletype>) * NUM_PQ);
    //BGPQ_Heap<eletype> bgpq_tmp = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    for (int i = 0; i < NUM_PQ; i++)
    {
        bgpq_tmp[i] = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    }

    cudaMalloc(&bgpq, sizeof(BGPQ_Heap<eletype>) * NUM_PQ);
    cudaMemcpy(bgpq, bgpq_tmp, sizeof(BGPQ_Heap<eletype>) * NUM_PQ, cudaMemcpyHostToDevice);

    cudaMalloc(&read_done, sizeof(int));
    cudaMemset(read_done, 0, sizeof(int));
    cudaMalloc(&write_reserve, sizeof(int));
    cudaMemset(write_reserve, 0, sizeof(int));

    int smemOffset_host[WARP_NUM_PER_BLOCK];
    cudaMalloc(&smemOffset, sizeof(int) * WARP_NUM_PER_BLOCK);
    buffer_size = mdata.l2_bufsize_perwarp;
    for (int i = 0; i < WARP_NUM_PER_BLOCK; i++)
    {
        smemOffset_host[i] = mdata.l2_queue_offset + i * buffer_size;
    }
    cudaMemcpy(smemOffset, smemOffset_host, sizeof(int) * WARP_NUM_PER_BLOCK, cudaMemcpyHostToDevice);

    return INIT_SUCCESS;
}

template <typename eletype>
init_status l2_multi_queue<eletype> :: host_reinit(int max_size, mlmq_mdata &mdata, eletype init_limits, mlmq_setup setup)
{
    batchSize = setup.s_l2_batch_size;
    int node_num = max_size / sizeof(eletype) / batchSize;

    NUM_PQ = L2_MQ_NUM;

    BGPQ_Heap<eletype> *bgpq_tmp = (BGPQ_Heap<eletype>*)malloc(sizeof(BGPQ_Heap<eletype>) * NUM_PQ);
    //BGPQ_Heap<eletype> bgpq_tmp = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    for (int i = 0; i < NUM_PQ; i++)
    {
        bgpq_tmp[i] = BGPQ_Heap<eletype>(node_num, init_limits, batchSize);
    }

    cudaMemcpy(bgpq, bgpq_tmp, sizeof(BGPQ_Heap<eletype>) * NUM_PQ, cudaMemcpyHostToDevice);

    cudaMemset(read_done, 0, sizeof(int));
    cudaMemset(write_reserve, 0, sizeof(int));

    int smemOffset_host[WARP_NUM_PER_BLOCK];
    buffer_size = mdata.l2_bufsize_perwarp;
    for (int i = 0; i < WARP_NUM_PER_BLOCK; i++)
    {
        smemOffset_host[i] = mdata.l2_queue_offset + i * buffer_size;
    }
    cudaMemcpy(smemOffset, smemOffset_host, sizeof(int) * WARP_NUM_PER_BLOCK, cudaMemcpyHostToDevice);

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l2_multi_queue<eletype> :: read(eletype *node_in, int &read_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
    start_time = clock();
#endif

    int qid = bid % NUM_PQ;

    // delete items from heap
    int bgpq_status = bgpq[qid].deleteRoot(node_in, read_num, wid, lane_id, debug_time);

#if (WORK_CLOCK == true)
    end_time = clock();
    debug_time[6] += end_time - start_time;
#endif

    __syncwarp();

    if (bgpq_status == true) {

        if (read_num >= batchSize)
        {
            bgpq[qid].deleteUpdate(wid, lane_id, smemOffset[wid]);
        }

        return READ_SUCCESS;
    }
    else
    {
        return READ_EMPTY;
    }
}

template <typename eletype>
__device__ write_status l2_multi_queue<eletype> :: write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
    for (int i = 0; i < write_num; i += l2_batch_size)
    {
        int tmp_write_num = mlq_min(l2_batch_size, write_num - i);

        bgpq[write_id].insertion_shared(node_out + i, tmp_write_num, wid, lane_id, smemOffset[wid], debug_time);
        write_id = (write_id + 1) % NUM_PQ;
        
        if (!lane_id)
            atomicAdd(write_reserve, tmp_write_num);
        __syncwarp();
    }

    return WRITE_SUCCESS;
}

template <typename eletype>
__device__ init_status l2_multi_queue<eletype> :: device_init(int bid, int wid, int lane_id)
{
    write_id = bid % NUM_PQ;

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ int l2_multi_queue<eletype> :: get_queue_size()
{
    return *write_reserve - *read_done;
}

template <typename eletype>
__device__ int l2_multi_queue<eletype> :: get_available_queue_size()
{
    // The heap queue likewise has no published cursor that can be sampled
    // without changing its read protocol.
    const int available = *write_reserve - *read_done;
    return available > 0 ? available : 0;
}

template <typename eletype>
__device__ void l2_multi_queue<eletype> :: manager_run(int wid, int lane_id)
{
    //printf("read_done, write_reserve %d %d\n", *read_done, *write_reserve);
}

template <typename eletype>
__device__ void l2_multi_queue<eletype> :: update_done(int on_the_fly_num)
{
    atomicAdd(read_done, on_the_fly_num);
}

template <typename eletype>
__device__ void l2_multi_queue<eletype> :: update_local_info(int lane_id)
{
}

template <typename eletype>
__host__ __device__ int l2_multi_queue<eletype> :: manage_warp_num()
{
    return 1;
}
