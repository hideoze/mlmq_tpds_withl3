#ifndef HEAPUTIL_CUH
#define HEAPUTIL_CUH

#include "../include/GPU_setup.h"

template<typename K>
__inline__ __device__ void batchCopy_warp(K *dest, K *source, int size, int lane_id)
{
    for (int i = lane_id; i < size; i += WARP_SIZE) {
        dest[i] = source[i];
        //printf("wid %d source %d dest %d\n", lane_id, dest[i], source[i]);
    }
}

template<typename K>
__inline__ __device__ void batchCopy_warp_reset(K *dest, K *source, int size, int lane_id, K init_limits = 0)
{
    for (int i = lane_id; i < size; i += WARP_SIZE) {
        dest[i] = source[i];
        source[i] = init_limits;
    }
}

template<typename K>
__inline__ __device__ void batchCopy(K *dest, K *source, int size, bool reset = false, K init_limits = 0)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dest[i] = source[i];
        if (reset) source[i] = init_limits;
    }
    __syncthreads();
}


template<typename T>
__inline__ __device__ void _swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename T>
__inline__ __device__ void icmp_and_swap(T &a, T &b, short bit) {
    T tmpa = a;
    T tmpb = b;
    if (bit)
    {
        a = (a < b)? tmpa: tmpb;
        b = (a < b)? tmpb: tmpa;
    }
    else
    {
        a = (a < b)? tmpb: tmpa;
        b = (a < b)? tmpa: tmpb;
    }
}

template<typename K>
__inline__ __device__ void ibitonicSort_warp(K *items, int lane_id, int batchSize) {
    for (int k = 2; k <= batchSize; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            int i = lane_id;
            for (int i = lane_id; i < batchSize; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                }
                __syncwarp();
            }
        }
    }
    __syncwarp();
}

// hard coded 32-element warp level sort
template<typename K>
__inline__ __device__ void ibitonicSort_warp_32(K *items, int lane_id)
{
    int i = lane_id;
    short bit0 = i ^ 1;
    short bit1 = i ^ 2;
    short bit2 = i ^ 4;
    short bit3 = i ^ 8;
    short bit4 = i ^ 16;

    icmp_and_swap<K>(items[i], items[bit0], bit0);
    icmp_and_swap<K>(items[i], items[bit1], bit1);
    icmp_and_swap<K>(items[i], items[bit0], bit0);
    icmp_and_swap<K>(items[i], items[bit2], bit2);
    icmp_and_swap<K>(items[i], items[bit1], bit1);
    icmp_and_swap<K>(items[i], items[bit0], bit0);
    icmp_and_swap<K>(items[i], items[bit3], bit3);
    icmp_and_swap<K>(items[i], items[bit2], bit2);
    icmp_and_swap<K>(items[i], items[bit1], bit1);
    icmp_and_swap<K>(items[i], items[bit0], bit0);
    icmp_and_swap<K>(items[i], items[bit4], bit4);
    icmp_and_swap<K>(items[i], items[bit3], bit3);
    icmp_and_swap<K>(items[i], items[bit2], bit2);
    icmp_and_swap<K>(items[i], items[bit1], bit1);
    icmp_and_swap<K>(items[i], items[bit0], bit0);
}

template<typename K>
__inline__ __device__ void ibitonicSort(K *items, int size) {

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename K>
__inline__ __device__ void dbitonicSort(K *items, int size) {

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] < items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] > items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename K>
__inline__ __device__ void dbitonicMerge(K *items, int size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] < items[ixj]))
                _swap<K>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

template<typename K>
__inline__ __device__ void ibitonicMerge(K *items, int size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] > items[ixj]))
                _swap<K>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

// merge 2 arrays of size batchSize
template<typename K>
__inline__ __device__ void imergePath_warp(K *aItems, K *bItems,
                                      K *smallItems, K *largeItems, K *tmpItems, int lane_id, int batchSize) {

    // extern __shared__ int s[];
    // K *tmpItems = (K *)&s[smemOffset];

    int lengthPerThread = batchSize * 2 / WARP_SIZE;

    int index = lane_id * lengthPerThread;
    int aTop = (index > batchSize) ? batchSize : index;
    int bTop = (index > batchSize) ? index - batchSize : 0;
    int aBottom = bTop;
    
    int offset, aI, bI;

    // if (lane_id == 0)
    // {
    //     int count = 0;
    //     int i = 0;
    //     int j = 0;
    //     while (i < batchSize && j < batchSize)
    //     {
    //         if (aItems[i] < bItems[j])
    //             tmpItems[count++] = aItems[i++];
    //         else
    //             tmpItems[count++] = bItems[j++];
    //     }
    //     while (i < batchSize)
    //     {
    //         tmpItems[count++] = aItems[i++];
    //     }
    //     while (j < batchSize)
    //     {
    //         tmpItems[count++] = bItems[j++];
    //     }
    // }
    
    // binary search for diagonal intersections
    while (1) {
        offset = (aTop - aBottom) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        // if (aItems[aI] > bItems[bI - 1])
        // {
        //     if (aItems[aI - 1] <= bItems[bI])
        //     {
        //         break;
        //     }
        //     else
        //     {
        //         aTop = aI - 1;
        //         bTop = bI + 1;
        //     }
        // }
        // else
        // {
        //     aBottom = aI + 1;
        // }

        if (aTop == aBottom || (bI < batchSize && (aI == batchSize || aItems[aI] > bItems[bI]))) {
            if (aTop == aBottom || aItems[aI - 1] <= bItems[bI]) {
                break;
            }
            else {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else {
            aBottom = aI;
        }
    }

    __syncwarp();

     // start from [aI, bI], found a path with lengthPerThread
    for (int i = lengthPerThread * lane_id; i < lengthPerThread * lane_id + lengthPerThread; ++i) {
        if (bI == batchSize || (aI < batchSize && aItems[aI] <= bItems[bI])) {
            tmpItems[i] = aItems[aI];
            aI++;
        }
        else if (aI == batchSize || (bI < batchSize && aItems[aI] > bItems[bI])) {
            tmpItems[i] = bItems[bI];
            bI++;
        }
    }

    __syncwarp();

    batchCopy_warp<K>(smallItems, tmpItems, batchSize, lane_id);
    batchCopy_warp<K>(largeItems, tmpItems + batchSize, batchSize, lane_id);

    __syncwarp();
}

template<typename K>
__inline__ __device__ void imergePath_warp_bak(K *aItems, K *bItems,
                                      K *smallItems, K *largeItems, K *tmpItems, int lane_id, int batchSize) {
    K tmpa = aItems[lane_id];
    K tmpb = bItems[lane_id];
    if (aItems[lane_id] < bItems[lane_id])
    {
        smallItems[lane_id] = tmpa;
        largeItems[lane_id] = tmpb;
    }
    else
    {
        smallItems[lane_id] = tmpb;
        largeItems[lane_id] = tmpa;
    }
    __syncwarp();
}

template<typename K>
__inline__ __device__ void imergePath(K *aItems, K *bItems,
                                      K *smallItems, K *largeItems,
                                      int size, int smemOffset) {

    extern __shared__ int s[];
    K *tmpItems = (K *)&s[smemOffset];

    int lengthPerThread = size * 2 / blockDim.x;

    int index= threadIdx.x * lengthPerThread;
    int aTop = (index > size) ? size : index;
    int bTop = (index > size) ? index - size : 0;
    int aBottom = bTop;
    
    int offset, aI, bI;
    
    // binary search for diagonal intersections
    while (1) {
        offset = (aTop - aBottom) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        if (aTop == aBottom || (bI < size && (aI == size || aItems[aI] > bItems[bI]))) {
            if (aTop == aBottom || aItems[aI - 1] <= bItems[bI]) {
                break;
            }
            else {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else {
            aBottom = aI;
        }
     }

     // start from [aI, bI], found a path with lengthPerThread
    for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
        if (bI == size || (aI < size && aItems[aI] <= bItems[bI])) {
            tmpItems[i] = aItems[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aItems[aI] > bItems[bI])) {
            tmpItems[i] = bItems[bI];
            bI++;
        }
    }
    __syncthreads();

    batchCopy(smallItems, tmpItems, size);
    batchCopy(largeItems, tmpItems + size, size);
}

#endif
