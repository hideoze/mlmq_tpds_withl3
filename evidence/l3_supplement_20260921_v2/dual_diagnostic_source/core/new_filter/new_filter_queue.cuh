#define LARGE_BUFFER_SIZE 16
template <typename eletype>
class l1_filter_queue_new
{
public:
    eletype data[MAX_L1FQ_BATCH_SIZE];
    eletype large_data[MAX_L1FQ_BATCH_SIZE];
    eletype buffer[MAX_L1FQ_BATCH_SIZE];
    eletype init_limits;
    VALUE_TYPE base;
    int batchSize;
    int data_size;
    int large_size;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};


template <typename eletype>
__device__ init_status l1_filter_queue_new<eletype> :: init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    data_size = 0;
    large_size = 0;
    batchSize = setup.s_l1_filter_size;
    base = 0;
    init_limits = init_limits_in;

    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_filter_queue_new<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
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
        if (large_size > 0)
        {
            for (int i = lane_id; i < large_size; i+=WARP_SIZE)
                data[i] = large_data[i];

            __syncwarp();
            if (!lane_id)
            {
                data_size = large_size;
                large_size = 0;
            }

            __syncwarp();
        }
        else
            return READ_EMPTY;
    }
}

template <typename eletype>
__device__ write_status l1_filter_queue_new<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    if (data_size == 0)
    {
        if (!lane_id)
        {
            base = (int)(node_out[0].get_data() / l1_filter_delta) * l1_filter_delta;
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
            large_data[large_size + global_update_pos] = node_out[i];
        __syncwarp();

        if (!lane_id)
        {
            large_size += count_bit(write_global_mask);
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
    if (buffer_num > 0 || large_size >= LARGE_BUFFER_SIZE)
    {
        int write_back_num = large_size;
        for (int i = lane_id; i < write_back_num; i+=WARP_SIZE)
        {
            buffer[buffer_num + i] = large_data[i];
        }
        buffer_num += write_back_num;
        __syncwarp();
        large_size = 0;
    }
    __syncwarp();

    return WRITE_SUCCESS;

}

template <typename eletype>
__device__ int l1_filter_queue_new<eletype> :: get_queue_size()
{
    return data_size + large_size;
}