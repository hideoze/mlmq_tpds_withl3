#include <cassert>
template <typename eletype>
__device__ init_status l1_none_queue<eletype> :: init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    batchSize = node_size;
    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_none_queue<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
{
    __syncwarp();
    return READ_EMPTY;
}

template <typename eletype>
__device__ write_status l1_none_queue<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    for (int cpy_iter = lane_id; cpy_iter < write_num; cpy_iter+=WARP_SIZE)
    {
        buffer[cpy_iter] = node_out[cpy_iter];
    }
    buffer_num += write_num;
    __syncwarp();

    return WRITE_EMPTY;
}

template <typename eletype>
__device__ int l1_none_queue<eletype> :: get_queue_size()
{
    return 0;
}

// l1 vector
template <typename eletype>
__device__ init_status l1_vector_queue<eletype> :: init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    batchSize = setup.s_l1_vector_size;
    data_size = 0;
    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_vector_queue<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
{
    if (data_size > 0)
    {
        read_num = mlq_min(data_size, l1_vector_size / 2);
        for (int i = lane_id; i < read_num; i+=WARP_SIZE)
            node_in[i] = data[i];
        __syncwarp();
        for (int i = read_num + lane_id; i < data_size; i+=WARP_SIZE)
            data[i - read_num] = data[i];
        __syncwarp();
        if (lane_id == 0)
            data_size -= read_num;
        __syncwarp();
        return READ_SUCCESS;
    }
    else
    {
        __syncwarp();
        return READ_EMPTY;
    }
}

template <typename eletype>
__device__ write_status l1_vector_queue<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    for (int i = lane_id; i < write_num; i+=WARP_SIZE)
    {
        data[data_size + i] = node_out[i];
    }
    if (!lane_id)
        data_size += write_num;
    __syncwarp();
    int write_min = batchSize / 2;
    //int write_back_num = (data_size + 1) / 2;
    int write_back_num = data_size;
    if (data_size > write_min)
    {
        for (int i = lane_id; i < write_back_num; i+=WARP_SIZE)
            buffer[i] = data[data_size - write_back_num + i];
        buffer_num = write_back_num;
        __syncwarp();
        if (!lane_id)
            data_size -= write_back_num;
        __syncwarp();
    }

    return WRITE_SUCCESS;
}

template <typename eletype>
__device__ int l1_vector_queue<eletype> :: get_queue_size()
{
    return data_size;
}

template <typename eletype>
__device__ init_status l1_filter_queue<eletype> :: init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    data_size = 0;
    batchSize = setup.s_l1_filter_size;
    base = 0;
    init_limits = init_limits_in;

    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_filter_queue<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
{
    if (data_size > 0)
    {
        read_num += mlq_min(node_size, data_size);
        
        coop_mem_cpy<eletype>(node_in, data + data_size - read_num, read_num, lane_id);
        __syncwarp();

        if (!lane_id)
            data_size -= read_num;
        __syncwarp();

        return READ_SUCCESS;
    }
    else
    {
        return READ_EMPTY;
    }
}

template <typename eletype>
__device__ write_status l1_filter_queue<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    if (data_size == 0)
    {
        if (!lane_id)
        {
            base = (int)(node_out[0].get_data() / l1_filter_delta) * l1_filter_delta;
        }
    }
    __syncwarp();

    if (data_size >= batchSize / 2)
    {
        //int write_back_num = batchSize / 2;
        int write_back_num = data_size;

        for (int i = lane_id; i < write_back_num; i+=WARP_SIZE)
        {
            buffer[buffer_num + i] = data[data_size - write_back_num + i];
        }
        __syncwarp();
        buffer_num += write_back_num;

        if (!lane_id)
        {
            data_size -= write_back_num;
        }
    }
    __syncwarp();

    for (int idx = 0; idx < write_num; idx+=WARP_SIZE)
    {
        int i = idx + lane_id;

        bool valid = i < write_num;

        unsigned valid_mask = __ballot_sync(FULL_MASK, valid);

        bool update_local = valid && node_out[i].get_data() < base + l1_filter_delta;
        unsigned write_local_mask = __ballot_sync(valid_mask, update_local);
        int local_write_num = count_bit(write_local_mask);
        int local_update_pos = count_bit(set_bits(write_local_mask, 0, lane_id, 32));

        if (update_local)
        {
            data[data_size + local_update_pos] = node_out[i];
        }
        __syncwarp();
        if (!lane_id)
        {
            data_size += local_write_num;
        }

        bool update_global = valid && node_out[i].get_data() >= base + l1_filter_delta;
        unsigned write_global_mask = __ballot_sync(valid_mask, update_global);
        int global_update_pos = count_bit(set_bits(write_global_mask, 0, lane_id, 32));

        if (update_global)
            buffer[buffer_num + global_update_pos] = node_out[i];
        buffer_num += count_bit(write_global_mask);
        __syncwarp();
    }
    __syncwarp();

    return WRITE_SUCCESS;

}

template <typename eletype>
__device__ int l1_filter_queue<eletype> :: get_queue_size()
{
    return data_size;
}

template <typename eletype>
__device__ init_status l1_SLF_queue<eletype> :: 
init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    batchSize = setup.s_l1_SLF_size;
    l = r = data_size = 0;
    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_SLF_queue<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
{
    // if(!lane_id) printf("read start\n");
    // if (data_size > 0)
    // {
    //     read_num = mlq_min(data_size, batchSize / 2);
    //     for (int i = lane_id; i < read_num; i+=WARP_SIZE)
    //         node_in[i] = data[i];
    //     __syncwarp();
    //     for (int i = read_num + lane_id; i < data_size; i+=WARP_SIZE)
    //         data[i - read_num] = data[i];
    //     __syncwarp();
    //     if (lane_id == 0)
    //         data_size -= read_num;
    //     __syncwarp();
    //     return READ_SUCCESS;
    // }
    // else
    // {
    //     __syncwarp();
    //     return READ_EMPTY;
    // }

    if (data_size > 0)
    {
        // assert(read_num==0);
        read_num = mlq_min(data_size, MAX_L1SLF_BATCH_SIZE/2); // /2
        for (int i = lane_id; i < read_num; i+=WARP_SIZE){
            int pos = l + i;
            if(pos >= MAX_L1SLF_BATCH_SIZE) pos -= MAX_L1SLF_BATCH_SIZE;
            // __syncwarp();
            node_in[i] = data[pos];
        }
        __syncwarp();

        if (lane_id == 0){
            data_size -= read_num;
            l += read_num;
            if(l >= MAX_L1SLF_BATCH_SIZE) l -= MAX_L1SLF_BATCH_SIZE;
        }
        __syncwarp();
        // if(!lane_id) printf("read end %d\n", data_size);
        return READ_SUCCESS;
    }
    else
    {
        __syncwarp();
        // if(!lane_id) printf("read end %d\n", data_size);
        return READ_EMPTY;
    }
}

// todo: 调用大小trace
template <typename eletype>
__device__ write_status l1_SLF_queue<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    // if(!lane_id) printf("write start\n");
    for (int idx = 0; idx < write_num; idx+=WARP_SIZE)
    {
        int i = idx + lane_id;
        // if(!lane_id) printf("%d %d\n", i, write_num);
        // __syncwarp();
        bool valid = i < write_num;
        unsigned valid_mask = __ballot_sync(FULL_MASK, valid);

        bool is_left = data_size? node_out[i].get_data() < data[l].get_data() : node_out[i].get_data() < node_out[0].get_data();
        bool update_left = valid && is_left, update_right = valid && !is_left;
        unsigned write_left_mask = __ballot_sync(valid_mask, update_left), write_right_mask = __ballot_sync(valid_mask, update_right);
        int write_left_num = count_bit(write_left_mask), write_right_num = count_bit(write_right_mask);
        int update_left_pos = count_bit(set_bits(write_left_mask, 0, lane_id, 32)), update_right_pos = count_bit(set_bits(write_right_mask, 0, lane_id, 32));
        // printf("%d %d %d\n",lane_id, update_left_pos, update_right_pos);
        if (update_left)
        {
            int pos = l - (update_left_pos+1);
            if(pos<0) pos += MAX_L1SLF_BATCH_SIZE;
            // __syncwarp();
            data[pos] = node_out[i];
        }
        else if (update_right){
            int pos = r + update_right_pos;
            if(pos >= MAX_L1SLF_BATCH_SIZE) pos -= MAX_L1SLF_BATCH_SIZE;
            // __syncwarp();
            data[pos] = node_out[i];
        }
        __syncwarp();

        if(!lane_id){
            l -= write_left_num;
            if(l < 0) l += MAX_L1SLF_BATCH_SIZE;
            r += write_right_num;
            if(r >= MAX_L1SLF_BATCH_SIZE) r -= MAX_L1SLF_BATCH_SIZE;
            // printf("left num %d right num %d\n", write_left_num, write_right_num);
        }
        __syncwarp();
    }
    __syncwarp();

    if (!lane_id)
        data_size += write_num;
    __syncwarp();
 
    // todo: write back
    int write_min = MAX_L1SLF_BATCH_SIZE / 2;
    int write_back_num = data_size;
    if (data_size > write_min)
    {
        for (int i = lane_id; i < write_back_num; i+=WARP_SIZE){
            int pos = r - (write_back_num - i);
            if(pos < 0) pos += MAX_L1SLF_BATCH_SIZE;
            // __syncwarp();
            buffer[i] = data[pos];
        }
        buffer_num = write_back_num;
        __syncwarp();
        if (!lane_id){
            r -= write_back_num;
            if(r < 0) r += MAX_L1SLF_BATCH_SIZE;
            data_size -= write_back_num;
        }
        __syncwarp();
    }

    // // if(!lane_id) printf("write end %d\n", data_size);
    return WRITE_SUCCESS;
}

template <typename eletype>
__device__ int l1_SLF_queue<eletype> :: get_queue_size()
{
    return data_size;
}

template <typename eletype>
__device__ init_status l1_near_far_queue<eletype> :: 
init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    if (!lane_id)
    {
        near_num = 0;
        far_num = 0;
        init_limits = init_limits_in;
        base = 0;
        delta = setup.s_l1_near_far_delta;

        near_bucket_size = setup.s_l1_near_far_size;
        far_bucket_size = setup.s_l1_near_far_size;
    }
    __syncwarp();
    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_near_far_queue<eletype> ::
read(eletype *node_in, int &read_num, int lane_id)
{

    if (near_num == 0 && far_num == 0)
        return READ_EMPTY;

    __syncwarp();

    if (near_num > 0)
    {
        int near_read_num = mlq_min(near_num, node_size);

        // queue
        if (near_read_num > 0)
        {
            for (int cpy_iter = lane_id; cpy_iter < near_read_num; cpy_iter+=WARP_SIZE)
            {
                node_in[cpy_iter] = data[near_num - near_read_num + cpy_iter];
            }
            __syncwarp();
        }
        __syncwarp();

        read_num += near_read_num;

        if (lane_id == 0)
        {
            near_num = near_num - read_num;
        }
        __syncwarp();
    }

    // If not enough, read from far bucket
    if (read_num < READ_MIN)
    {
        int far_read_num = mlq_min(far_num, node_size - read_num);

        // queue
        if (far_read_num > 0)
        {
            for (int cpy_iter = lane_id; cpy_iter < far_read_num; cpy_iter+=WARP_SIZE)
            {
                node_in[read_num + cpy_iter] = data[near_bucket_size + far_num - far_read_num + cpy_iter];
            }
            __syncwarp();
        }
        __syncwarp();

        read_num += far_read_num;

        if (lane_id == 0)
        {
            far_num = far_num - far_read_num;
        }
        __syncwarp();

    }

    __syncwarp();

    return READ_SUCCESS;
}

template <typename eletype>
__device__ write_status l1_near_far_queue<eletype> ::
write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    eletype update_min = init_limits;

    buffer_num = 0;

#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
    start_time = clock();
#endif
    for (int i = 0; i < write_num; i+=WARP_SIZE)
    {
        int node_idx = i + lane_id;

#if (WORK_CLOCK == true)
        unsigned start_time2, end_time2;
#endif
        
        bool valid = node_idx < write_num;
        bool near_valid = valid && node_out[node_idx].get_data() < base + delta;
        bool far_valid = valid && node_out[node_idx].get_data() >= base + delta;

        unsigned near_mask = __ballot_sync(FULL_MASK, near_valid);
        unsigned far_mask = __ballot_sync(FULL_MASK, far_valid);

        int update_near_num = count_bit(near_mask);
        int update_far_num = count_bit(far_mask);
        int update_near_pos = count_bit(set_bits(near_mask, 0, lane_id, 32));
        int update_far_pos = count_bit(set_bits(far_mask, 0, lane_id, 32));
        // When bucket is full, write back half the bucket

#if (WORK_CLOCK == true)
        start_time2 = clock();
#endif

        // near bucket
        if (near_num + update_near_num > N_TH)
        {
            int write_buffer_num = mlq_min(N_WB, near_num);
            int start_pos = near_num - write_buffer_num;

            for (int cpy_iter = lane_id; cpy_iter < write_buffer_num; cpy_iter += WARP_SIZE)
            {
                buffer[buffer_num + cpy_iter] = data[start_pos + cpy_iter];
            }

            buffer_num += write_buffer_num;

            __syncwarp();
            if (!lane_id)
                near_num = near_num - write_buffer_num;

        }

        // far bucket
        if (far_num + update_far_num > F_TH)
        {
            int write_buffer_num = mlq_min(F_WB, far_num);
            int start_pos = far_num - write_buffer_num;
            for (int cpy_iter = lane_id; cpy_iter < write_buffer_num; cpy_iter += WARP_SIZE)
            {
                buffer[buffer_num + cpy_iter] = data[near_bucket_size + start_pos + cpy_iter];
            }
            buffer_num += write_buffer_num;

            if (!lane_id)
                far_num = far_num - write_buffer_num;
        }

        __syncwarp();

#if (WORK_CLOCK == true)
        end_time2 = clock();
        debug_time[11] += end_time2 - start_time2;
#endif

#if (WORK_CLOCK == true)
        start_time2 = clock();
#endif
        
        if (near_valid)
        {
            data[near_num + update_near_pos] = node_out[node_idx];
        }
        else if (far_valid)
        {
            data[near_bucket_size + far_num + update_far_pos] = node_out[node_idx];
            // update_min = mlq_min(update_min, node_out[node_idx]);
        }
        __syncwarp();
        if (!lane_id)
        {
            near_num = near_num + update_near_num;
            far_num = far_num + update_far_num;
        }

        __syncwarp();

#if (WORK_CLOCK == true)
        end_time2 = clock();
#endif
    }

#if (WORK_CLOCK == true)
    start_time = clock();
#endif

    // change base and refresh the bucket
    if (near_num == 0 && far_num > 0)
    {

        VALUE_TYPE update_data_min = data[near_bucket_size].get_data();
        // update_min.get_data();
        // reduction for minimum
        // for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        // {
        //     VALUE_TYPE neighbour_data_min = __shfl_down_sync(FULL_MASK, update_data_min, offset);
        //     update_data_min = mlq_min(neighbour_data_min, update_data_min);
        // }

        // set new base
        if (!lane_id)
        {
            base = (int)(update_data_min / delta) * delta;
        }
        __syncwarp();

        int new_near_num = 0;
        int new_far_num = 0;

        for (int i = 0; i < far_num; i += WARP_SIZE)
        {
            int node_idx = i + lane_id;
        
            bool valid = node_idx < far_num;
            bool near_valid = valid && data[near_bucket_size + node_idx].get_data() < base + delta;
            bool far_valid = valid && data[near_bucket_size + node_idx].get_data() >= base + delta;

            unsigned near_mask = __ballot_sync(FULL_MASK, near_valid);
            unsigned far_mask = __ballot_sync(FULL_MASK, far_valid);

            int update_near_num = count_bit(near_mask);
            int update_far_num = count_bit(far_mask);
            int update_near_pos = count_bit(set_bits(near_mask, 0, lane_id, 32));
            int update_far_pos = count_bit(set_bits(far_mask, 0, lane_id, 32));
            
            if (near_valid)
            {
                data[new_near_num + update_near_pos] = data[near_bucket_size + node_idx];
            }
            __syncwarp();
            if (far_valid)
            {
                data[near_bucket_size + new_far_num + update_far_pos] = data[near_bucket_size + node_idx];
            }
            __syncwarp();

            new_near_num += update_near_num;
            new_far_num += update_far_num;
        }
        __syncwarp();
        if (!lane_id)
        {
            near_num = new_near_num;
            far_num = new_far_num;
        }
#if (WORK_CLOCK == true)
        end_time = clock();
#endif
    }

    __syncwarp();

    return WRITE_SUCCESS;
    
}

template <typename eletype>
__device__ int l1_near_far_queue<eletype> ::
get_queue_size()
{
    return near_num + far_num;
}