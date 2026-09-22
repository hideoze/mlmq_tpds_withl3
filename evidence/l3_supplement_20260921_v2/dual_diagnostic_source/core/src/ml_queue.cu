//#include "../include/ml_queue.cuh"

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
init_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
init_host(int max_size, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    min_read_gra = setup.s_min_read_gra;

    cudaMalloc((void**)&run_begin, sizeof(int));
    cudaMemset(run_begin, 0, sizeof(int));

    mdata.l1_queue_offset = 0;
    // Calculate shared memory volume
    if (typeid(l1_queue_type) == typeid(l1_none_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_none_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_vector_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_vector_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_near_far_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_near_far_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_filter_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_filter_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_filter_queue_new<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_filter_queue_new<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_hop_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_hop_queue<eletype>);
    }
    else if(typeid(l1_queue_type) == typeid(l1_SLF_queue<eletype>)){
        mdata.l1_bufsize_perwarp = sizeof(l1_SLF_queue<eletype>);
    }
    else
    {
        printf("l1 type not supported!\n");
        return INIT_FAILED;
    }

    init_status status;
    mdata.l2_queue_offset = (mdata.l1_bufsize_perwarp + mdata.l1_bufsize_perwarp) * WARP_NUM_PER_BLOCK;
    mdata.l2_bufsize_extra = 0;
    if (typeid(l2_queue_type) == typeid(l2_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_batch_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_multi_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_delta_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2 * setup.s_BNUM;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int) * setup.s_BNUM;
    }
    else if (typeid(l2_queue_type) == typeid(l2_bgpq_queue<eletype>))
    {
        mdata.l2_bufsize_perwarp = setup.s_l2_batch_size * sizeof(eletype) * 6;
    }
    else if (typeid(l2_queue_type) == typeid(l2_multi_queue<eletype>))
    {
        mdata.l2_bufsize_perwarp = setup.s_l2_batch_size * sizeof(eletype) * 6;
    }
    else
    {
        printf("l2 type not supported!\n");
        return INIT_FAILED;
    }
    status = q2.host_init(max_size, mdata, init_limits, setup);

    if (status == INIT_FAILED)
    {
        printf("q2 initilization failed!\n");
        return INIT_FAILED;
    }

    return INIT_SUCCESS;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
init_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
reinit_host(int max_size, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    min_read_gra = setup.s_min_read_gra;

    cudaMemset(run_begin, 0, sizeof(int));

    mdata.l1_queue_offset = 0;
    // Calculate shared memory volume
    if (typeid(l1_queue_type) == typeid(l1_none_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_none_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_vector_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_vector_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_near_far_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_near_far_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_filter_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_filter_queue<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_filter_queue_new<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_filter_queue_new<eletype>);
    }
    else if (typeid(l1_queue_type) == typeid(l1_hop_queue<eletype>))
    {
        mdata.l1_bufsize_perwarp = sizeof(l1_hop_queue<eletype>);
    }
    else if(typeid(l1_queue_type) == typeid(l1_SLF_queue<eletype>)){
        mdata.l1_bufsize_perwarp = sizeof(l1_SLF_queue<eletype>);
    }
    else
    {
        printf("l1 type not supported!\n");
        return INIT_FAILED;
    }

    init_status status;
    mdata.l2_queue_offset = (mdata.l1_bufsize_perwarp + mdata.l1_bufsize_perwarp) * WARP_NUM_PER_BLOCK;
    mdata.l2_bufsize_extra = 0;
    if (typeid(l2_queue_type) == typeid(l2_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_batch_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_multi_vector_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int);
    }
    else if (typeid(l2_queue_type) == typeid(l2_delta_queue<eletype>))
    {
        // local_read_ptr & local_dst_read_ptr
        mdata.l2_bufsize_perwarp = sizeof(int) * 2 * setup.s_BNUM;
        // for local_read_pos
        mdata.l2_bufsize_extra = sizeof(int) * setup.s_BNUM;
    }
    else if (typeid(l2_queue_type) == typeid(l2_bgpq_queue<eletype>))
    {
        mdata.l2_bufsize_perwarp = setup.s_l2_batch_size * sizeof(eletype) * 6;
    }
    else if (typeid(l2_queue_type) == typeid(l2_multi_queue<eletype>))
    {
        mdata.l2_bufsize_perwarp = setup.s_l2_batch_size * sizeof(eletype) * 6;
    }
    else
    {
        printf("l2 type not supported!\n");
        return INIT_FAILED;
    }
    status = q2.host_reinit(max_size, mdata, init_limits, setup);

    if (status == INIT_FAILED)
    {
        printf("q2 initilization failed!\n");
        return INIT_FAILED;
    }

    return INIT_SUCCESS;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
int ml_queue<eletype, l1_queue_type, l2_queue_type> :: get_shm_size()
{
    int shm_size = mdata.l1_bufsize_perwarp * WARP_NUM_PER_BLOCK
    + mdata.l2_bufsize_perwarp * WARP_NUM_PER_BLOCK + mdata.l2_bufsize_extra;
    return shm_size;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ init_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
init_device(int bid, int wid, int lane_id, mlmq_setup setup)
{
    init_status status;

    extern __shared__ int s[];
    l1_queue_type *q1 = (l1_queue_type*)(s);
    status = q1[wid].init(wid, lane_id, init_limits, setup);
    if (status != INIT_SUCCESS) return status;

    status = q2.device_init(bid, wid, lane_id);
    if (status != INIT_SUCCESS) return status;
    
    return INIT_SUCCESS;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ read_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
read(eletype *node_in, int &read_num, int &on_the_fly_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
#if (ACCESS_THROUGH == true)
    while (read_num < min_read_gra)
    {
        int new_read_num = 0;
        read_status l2_status = q2.read(node_in + read_num, new_read_num, bid, wid, lane_id, debug_time);
        __syncwarp();
        read_num += new_read_num;
        on_the_fly_num += new_read_num;
        if (!new_read_num) break;
    }
#else
    read_num = 0;
    int local_read_num;
    int buffer_num;
    while (read_num < min_read_gra)
    {
        extern __shared__ int s[];
        l1_queue_type *q1 = (l1_queue_type*)(s);
        local_read_num = 0;
        read_status l1_status = q1[wid].read(node_in + read_num, local_read_num, lane_id);
        read_num += local_read_num;

        __syncwarp();
        if (local_read_num == 0)
        {
            buffer_num = 0;

            read_status l2_status = q2.read(q1[wid].buffer, buffer_num, bid, wid, lane_id, debug_time);
            on_the_fly_num += buffer_num;
            
            // if(!lane_id) printf("q2 %d %d\n", q2.get_queue_size(), l2_status);
            __syncwarp();

            if (l2_status == READ_SUCCESS)
            {
                int tmp_buffer_num = 0;
                __shared__ eletype buffer2[WARP_NUM_PER_BLOCK * l2_batch_size];
                eletype *buffer2_ptr = buffer2 + l2_batch_size * wid;
                write_status l1_write_status = q1[wid].write(q1[wid].buffer, buffer2_ptr, buffer_num, tmp_buffer_num, lane_id, debug_time);
                __syncwarp();

                // if (!lane_id)
                // {
                //     printf("q2 size %d on_the_fly %d\n", q2.get_queue_size(), on_the_fly_num);
                //     printf("q1 twice write done b %d b2 %d local_s %d q2 size %d\n", buffer_num, tmp_buffer_num, q1[wid].get_queue_size(), q2.get_queue_size());
                //     // for (int i = 0; i < tmp_buffer_num; i++)
                //     //     printf("(%d, %d) ", buffer2_ptr[i].id, buffer2_ptr[i].dist);
                //     // printf("\n");
                // }

                if (l1_write_status == WRITE_SUCCESS)
                {
                    if (tmp_buffer_num > 0)
                        q2.write(buffer2_ptr, tmp_buffer_num, bid, wid, lane_id, debug_time);
                    local_read_num = 0;
                    l1_status = q1[wid].read(node_in + read_num, local_read_num, lane_id);
                    read_num += local_read_num;
                    // if (!lane_id)
                    // {
                    //     printf("q2 size %d on_the_fly %d read_num %d\n", q2.get_queue_size(), on_the_fly_num, read_num);
                    // }
                }
                else
                {
                    for (int i = lane_id; i < tmp_buffer_num; i+= WARP_SIZE)
                        node_in[read_num + i] = buffer2_ptr[i];
                    read_num += tmp_buffer_num;
                    __syncwarp();
                    if (!tmp_buffer_num)
                        break;
                }

                __syncwarp();
            }
            else
            {  
                __syncwarp();
                break;
            }
        }
    }
#endif

    __syncwarp();
    if (!read_num)
    {
        return READ_EMPTY;
    }
    else
    {
        return READ_SUCCESS;
    }
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ write_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
write(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
#if (ACCESS_THROUGH == true)
    write_status l2_status = q2.write(node_out, write_num, bid, wid, lane_id, debug_time);
#else

    extern __shared__ int s[];
    l1_queue_type *q1 = (l1_queue_type*)(s);

    int buffer_num = 0;
    // if (!lane_id)
    //     printf("before write to q1 w %d b %d s %d\n", write_num, buffer_num, q1[wid].get_queue_size());
    write_status l1_status = q1[wid].write(node_out, q1[wid].buffer, write_num, buffer_num, lane_id, debug_time);

    // if (!lane_id)
    //     printf("after write to q1 w %d b %d s %d\n", write_num, buffer_num, q1[wid].get_queue_size());

    __syncwarp();

    if (buffer_num > 0)
    {
        // if (!lane_id)
        //     printf("before write to q2 %d %d\n", buffer_num, q2.get_queue_size());
        write_status l2_status = q2.write(q1[wid].buffer, buffer_num, bid, wid, lane_id, debug_time);
        // if (!lane_id)
        //     printf("before write to q2 %d %d\n", buffer_num, q2.get_queue_size());
        __syncwarp();
        return l2_status;
    }
#endif

    return WRITE_SUCCESS;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ write_status ml_queue<eletype, l1_queue_type, l2_queue_type> :: 
write_through(eletype *node_out, int &write_num, int bid, int wid, int lane_id, unsigned *debug_time)
{
    write_status l2_status = q2.write(node_out, write_num, bid, wid, lane_id, debug_time);
    // if(!lane_id)printf("write through %d\n", l2_status);
    return l2_status;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ int ml_queue<eletype, l1_queue_type, l2_queue_type> ::
spill_local_to_l2(int bid, int wid, int lane_id, unsigned *debug_time)
{
    extern __shared__ int s[];
    l1_queue_type *q1 = (l1_queue_type*)(s);
    int spilled = 0;
    while (q1[wid].get_queue_size() > 0)
    {
        int count = 0;
        q1[wid].read(q1[wid].buffer, count, lane_id);
        __syncwarp();
        // The supported delta queue publishes all count items before return.
        // No update_done is permitted between removing L1 and publishing L2.
        q2.write(q1[wid].buffer, count, bid, wid, lane_id, debug_time);
        __syncwarp();
        spilled += count;
    }
    return spilled;
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ int ml_queue<eletype, l1_queue_type, l2_queue_type> :: get_global_queue_size()
{
    return q2.get_queue_size();
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ int ml_queue<eletype, l1_queue_type, l2_queue_type> :: get_available_queue_size()
{
    return q2.get_available_queue_size();
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ int ml_queue<eletype, l1_queue_type, l2_queue_type> :: get_local_queue_size(int wid)
{
    extern __shared__ int s[];
    l1_queue_type *q1 = (l1_queue_type*)(s);
    return q1[wid].get_queue_size();
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ void ml_queue<eletype, l1_queue_type, l2_queue_type> :: update_done(int on_the_fly_num)
{
    return q2.update_done(on_the_fly_num);
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ void ml_queue<eletype, l1_queue_type, l2_queue_type> :: l2_manager(int wid, int lane_id)
{
    q2.manager_run(wid, lane_id);
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__device__ void ml_queue<eletype, l1_queue_type, l2_queue_type> :: update_local_info(int lane_id)
{
    q2.update_local_info(lane_id);
}

template <typename eletype, typename l1_queue_type, typename l2_queue_type>
__host__ __device__ int ml_queue<eletype, l1_queue_type, l2_queue_type> :: manage_warp_num()
{
    return q2.manage_warp_num();
}
