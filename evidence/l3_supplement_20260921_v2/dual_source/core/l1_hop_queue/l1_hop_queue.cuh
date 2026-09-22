#define CLEAR_THRESH 2
template <typename eletype>
class l1_hop_queue
{
public:
    eletype data[MAX_L1V_BATCH_SIZE];
    eletype buffer[MAX_L1V_BATCH_SIZE];
    eletype init_limits;
    int batchSize;
    int data_size;
    int hop = 0;

    __device__ init_status init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup);

    __device__ read_status read(eletype *node_in, int &read_num, int lane_id);

    __device__ write_status write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time);

    __device__ int get_queue_size();
};

template <typename eletype>
__device__ init_status l1_hop_queue<eletype> :: init(int warp_id, int lane_id, eletype init_limits_in, mlmq_setup setup)
{
    init_limits = init_limits_in;
    batchSize = setup.s_l1_vector_size;
    data_size = 0;
    hop = 0;
    __syncwarp();

    return INIT_SUCCESS;
}

template <typename eletype>
__device__ read_status l1_hop_queue<eletype> :: read(eletype *node_in, int &read_num, int lane_id)
{
    if (data_size > 0)
    {
        read_num = data_size;
        for (int i = lane_id; i < read_num; i+=WARP_SIZE)
            node_in[i] = data[i];
        // __syncwarp();
        // for (int i = read_num; i < data_size; i+=WARP_SIZE)
        //     data[i - read_num] = data[i];
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
__device__ write_status l1_hop_queue<eletype> :: write(eletype *node_out, eletype *buffer, int &write_num, int &buffer_num, int lane_id, unsigned *debug_time)
{
    for (int i = lane_id; i < write_num; i+=WARP_SIZE)
    {
        data[data_size + i] = node_out[i];
    }
    if (!lane_id)
    {
        data_size += write_num;
        hop += 1;
    }
    __syncwarp();
    int write_min = batchSize / 2;
    //int write_back_num = (data_size + 1) / 2;
    int write_back_num = data_size;
    if (data_size > write_min || hop >= CLEAR_THRESH)
    {
        for (int i = lane_id; i < write_back_num; i+=WARP_SIZE)
            buffer[i] = data[data_size - write_back_num + i];
        buffer_num = write_back_num;
        __syncwarp();
        if (!lane_id)
        {
            data_size -= write_back_num;
            hop = 0;
        }
    }
    __syncwarp();

    return WRITE_SUCCESS;
}

template <typename eletype>
__device__ int l1_hop_queue<eletype> :: get_queue_size()
{
    return data_size;
}