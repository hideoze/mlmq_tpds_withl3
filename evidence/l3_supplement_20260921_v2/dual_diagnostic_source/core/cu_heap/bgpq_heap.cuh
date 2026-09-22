#ifndef HEAP_CUH
#define HEAP_CUH

// derived from BGPQ:
// Chen, Yanhao, et al. "BGPQ: A Heap-Based Priority Queue Design for GPUs."
// Proceedings of the 50th International Conference on Parallel Processing. 2021.

#include "heaputil.cuh"
#include <stdio.h>
#include <algorithm>

#define AVAIL 0
#define INUSE 1
#define TARGET 2
#define MARKED 3

#define BUSY_WAITING_FENCE __threadfence()

// element type K, the size of the node batchSize
template<typename K>
class BGPQ_Heap {
    public:

        K init_limits;

        int batchNum;
        int batchSize;

        // Number of nodes in the heap
        int *batchCount;
        // Size of partial buffer, 
        int *partialBufferSize;
#ifdef HEAP_SORT
        int *deleteCount;
#endif
#ifdef PBS_MODEL
        int *globalBenefit;
        uint32_t *tbstate;
        uint32_t *terminate;
#endif
        // 0 is partial buffer, 1 ~ batchCount is the heap
        K *heapItems;
        int *status;

        // shared memory buffer of each 
        K *sm_buffer;
        int sm_buffer_size;

        //BGPQ_Heap() {};

        BGPQ_Heap(int _batchNum, K _init_limits, int _batchSize)
        {
            batchSize = _batchSize;
            batchNum = _batchNum;
            init_limits = _init_limits;
            sm_buffer_size = batchSize * 4;

            // prepare device heap
            cudaMalloc((void **)&heapItems, sizeof(K) * batchSize * (batchNum + 1));
            // initialize heap items with max value
            K *tmp = new K[batchSize * (batchNum + 1)];
            std::fill(tmp, tmp + batchSize * (batchNum + 1), init_limits);
            cudaMemcpy(heapItems, tmp, sizeof(K) * batchSize * (batchNum + 1), cudaMemcpyHostToDevice);
            delete []tmp; tmp = NULL;

            cudaMalloc((void **)&status, sizeof(int) * (batchNum + 1));
            cudaMemset(status, AVAIL, sizeof(int) * (batchNum + 1));

            cudaMalloc((void **)&batchCount, sizeof(int));
            cudaMemset(batchCount, 0, sizeof(int));
            cudaMalloc((void **)&partialBufferSize, sizeof(int));
            cudaMemset(partialBufferSize, 0, sizeof(int));
#ifdef HEAP_SORT
            cudaMalloc((void **)&deleteCount, sizeof(int));
            cudaMemset(deleteCount, 0, sizeof(int));
#endif
#ifdef PBS_MODEL
            cudaMalloc((void **)&globalBenefit, sizeof(int));
            cudaMemset(globalBenefit, 0, sizeof(int));
            cudaMalloc((void **)&tbstate, 1024 * sizeof(uint32_t));
            cudaMemset(tbstate, 0, 1024 * sizeof(uint32_t));
            uint32_t tmp1 = 1;
            cudaMemcpy(tbstate, &tmp1, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&terminate, sizeof(uint32_t));
            cudaMemset(terminate, 0, sizeof(uint32_t));
#endif
        };

        // determine the next batch when insert operation updating the heap
        // given the current batch index and the target batch index
        // return the next batch index to the target batch
        __device__ int getNextIdxToTarget(int currentIdx, int targetIdx) {
            return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
        }

        // __device__ void print_node(K *addr, int size = batchSize)
        // {
        //     printf("node: ");
        //     for (int i = 0; i < size; i++)
        //         printf("%d ", addr[i]);
        //     printf("\n");
        // }

        // __device__ void print_node_data(K *addr, int size = batchSize)
        // {
        //     printf("node: ");
        //     for (int i = 0; i < size; i++)
        //         printf("%d ", addr[i].get_data());
        //     printf("\n");
        // }

        __device__ bool changeStatus(int *_status, int oriS, int newS) {
            while (atomicCAS(_status, oriS, newS) != oriS){
                BUSY_WAITING_FENCE;
            }
            return true;
        }

        __inline__ __device__ bool deleteRoot(K *items, int &size, int wid, int lane_id, unsigned *debug_time) {

            // if read thread directly require block, write thread may starve
            if (*batchCount == 0 && *partialBufferSize == 0)
            {
                return false;
            }

            if (lane_id == 0) {
                changeStatus(&status[0], AVAIL, INUSE);

            }

            __syncwarp();

            int deleteOffset = 0;

            if (*batchCount == 0 && *partialBufferSize == 0) {
                if (lane_id == 0) {
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                __syncwarp();
                return false;
            }

            // printf("%d check\n", lane_id);

            if (*batchCount == 0 && *partialBufferSize != 0) {
                // only partial batch has items
                // output the partial batch
                size = *partialBufferSize;
                batchCopy_warp_reset<K>(items + deleteOffset, heapItems, size, lane_id, init_limits);
                if (lane_id == 0) {
                    __threadfence();
                    *partialBufferSize = 0;
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                __syncwarp();
                return true;
            }

            if (lane_id == 0) {
                changeStatus(&status[1], AVAIL, INUSE);

            }
            __syncwarp();

            size = batchSize;
            batchCopy_warp<K>(items + deleteOffset, heapItems + batchSize, size, lane_id);

            return true;
        }
        
        // deleteUpdate is used to update the heap
        // it will fill the empty root batch
        __device__ void deleteUpdate(int wid, int lane_id, int smemOffset) {

            extern __shared__ int s[];

            smemOffset = smemOffset / sizeof(int);

            K *sMergedItems = (K *)&s[smemOffset];
            // What is tmpIdx?
            volatile int *tmpIdx = (int *)&s[smemOffset];

            smemOffset += sizeof(K) * 3 * batchSize / sizeof(int);
            int *tmpType = (int *)&s[smemOffset - 1];

            K *tmpBuffer = (K*)(s + smemOffset);

            if (lane_id == 0) {
                *tmpIdx = atomicSub(batchCount, 1);
                // if no more batches in the heap
                if (*tmpIdx == 1) {
                    changeStatus(&status[1], INUSE, AVAIL);
                    changeStatus(&status[0], INUSE, AVAIL);
                }
            }
            __syncwarp();

            int lastIdx = *tmpIdx;
            __syncwarp();

            if (lastIdx == 1) return;
            
            if (lane_id == 0) {
                int debug_count = 0;
                while(1) {
                    if (atomicCAS(&status[lastIdx], AVAIL, INUSE) == AVAIL) {
                        *tmpType = 0;
                        break;
                    }
                    if (atomicCAS(&status[lastIdx], TARGET, MARKED) == TARGET) {
                        *tmpType = 1;
                        break;
                    }
                    BUSY_WAITING_FENCE;
                }
            }
            __syncwarp();

            if (*tmpType == 1) {
                // wait for insert worker
                if (lane_id == 0) {
                    while (atomicCAS(&status[lastIdx], TARGET, AVAIL) != TARGET) {BUSY_WAITING_FENCE;}
                }
                __syncwarp();

                batchCopy_warp<K>(sMergedItems, heapItems + batchSize, batchSize, lane_id);
            }
            else if (*tmpType == 0){

                batchCopy_warp_reset<K>(sMergedItems, heapItems + lastIdx * batchSize, batchSize, lane_id, init_limits);

                if (lane_id == 0) {
                    changeStatus(&status[lastIdx], INUSE, AVAIL);
                }
                __syncwarp();
            }

            /* start handling partial batch */
            if (*partialBufferSize)
            {
                batchCopy_warp<K>(sMergedItems + batchSize, heapItems, batchSize, lane_id);

                imergePath_warp<K>(sMergedItems, sMergedItems + batchSize,
                        sMergedItems, heapItems, tmpBuffer, lane_id, batchSize);
                __syncwarp();
            }

            __syncwarp();

            if (lane_id == 0) {
                changeStatus(&status[0], INUSE, AVAIL);
            }
            __syncwarp();
            /* end handling partial batch */
            // to finish

            int currentIdx = 1;
            while (1) {
                int leftIdx = currentIdx << 1;
                int rightIdx = leftIdx + 1;

                // Wait until status[] are not locked
                // After that if the status become unlocked, than child exists
                // If the status is not unlocked, than no valid child
                if (lane_id == 0) {
                    int leftStatus, rightStatus;
                    leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                    while (leftStatus == INUSE) {
                        leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                        BUSY_WAITING_FENCE;
                    }
                    int current_size = *batchCount;
                    if (leftStatus != AVAIL || leftIdx >= current_size) {
                        *tmpType = 0;
                        leftStatus = atomicCAS(&status[leftIdx], INUSE, AVAIL);
                    }
                    else {
                        rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                        while (rightStatus == INUSE) {
                            rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                            BUSY_WAITING_FENCE;
                        }
                        if (rightStatus != AVAIL || rightIdx >= current_size) {
                            *tmpType = 1;
                            rightStatus = atomicCAS(&status[rightIdx], INUSE, AVAIL);
                        }
                        else {
                            *tmpType = 2;
                        }
                    }
                }

                __syncwarp();

                int deleteType = *tmpType;

                if (deleteType == 0) { // no children
                    // move shared memory to currentIdx
                    batchCopy_warp<K>(heapItems + currentIdx * batchSize, sMergedItems, batchSize, lane_id);
                    if (lane_id == 0) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                    }
                    return;
                }
                else if (deleteType == 1) { // only has left child and left child is a leaf batch
                    // move leftIdx to shared memory
                    batchCopy_warp<K>(sMergedItems + batchSize, heapItems + leftIdx * batchSize, batchSize, lane_id);

                    imergePath_warp<K>(sMergedItems, sMergedItems + batchSize, heapItems + currentIdx * batchSize, 
                                heapItems + leftIdx * batchSize, tmpBuffer, lane_id, batchSize);
                    __syncwarp();

                    if (lane_id == 0) {
                        // unlock batch[currentIdx] & batch[leftIdx]
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                        changeStatus(&status[leftIdx], INUSE, AVAIL);
                    }
                    __syncwarp();
                    return;
                }

                // move leftIdx and rightIdx to shared memory
                batchCopy_warp<K>(sMergedItems + batchSize,
                          heapItems + leftIdx * batchSize,
                          batchSize, lane_id);
                batchCopy_warp<K>(sMergedItems + 2 * batchSize,
                          heapItems + rightIdx * batchSize,
                          batchSize, lane_id);

                int largerIdx = (heapItems[leftIdx * batchSize + batchSize - 1] < heapItems[rightIdx * batchSize + batchSize - 1]) ? rightIdx : leftIdx;
                int smallerIdx = 4 * currentIdx - largerIdx + 1;
                __syncwarp();

                imergePath_warp<K>(sMergedItems + batchSize, sMergedItems + 2 * batchSize,
                           sMergedItems + batchSize, heapItems + largerIdx * batchSize,
                           tmpBuffer, lane_id, batchSize);
                __syncwarp();

                if (lane_id == 0) {
                    changeStatus(&status[largerIdx], INUSE, AVAIL);
                }
                __syncwarp();

                if (sMergedItems[0] >= sMergedItems[2 * batchSize - 1]) {
                    batchCopy_warp<K>(heapItems + currentIdx * batchSize,
                              sMergedItems + batchSize,
                              batchSize, lane_id);
                }
                else if (sMergedItems[batchSize - 1] <= sMergedItems[batchSize]) {
                    batchCopy_warp<K>(heapItems + currentIdx * batchSize, 
                              sMergedItems,
                              batchSize, lane_id);
                    batchCopy_warp<K>(heapItems + smallerIdx * batchSize, 
                              sMergedItems + batchSize,
                              batchSize, lane_id);
                    if (lane_id == 0) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                        changeStatus(&status[smallerIdx], INUSE, AVAIL);
                    }
                    __syncwarp();
                    return;
                }
                else {
                    imergePath_warp<K>(sMergedItems, sMergedItems + batchSize,
                               heapItems + currentIdx * batchSize, sMergedItems,
                               tmpBuffer, lane_id, batchSize);
                }
                __syncwarp();

                if (lane_id == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                }
                __syncwarp();
                currentIdx = smallerIdx;

            }

        }

        // Items are already in smem
        // Notification: the function call will destroy the shared memory space items!
        __device__ void insertion_shared(K *items, int size, int wid, int lane_id, int smemOffset, unsigned *debug_time)
        {
            // to do: when size != batchSize
            extern __shared__ int s[];
            K *sMergedItems1;
            K *sMergedItems2;
            int *tmpIdx;
            K* tmpBuffer;

            if (size < batchSize)
            {
                smemOffset = smemOffset / sizeof(int);
                sMergedItems1 = (K *)&s[smemOffset];
                sMergedItems2 = (K *)&sMergedItems1[batchSize];
                smemOffset += sizeof(K) * 2 * batchSize / sizeof(int);
                tmpIdx = (int *)&s[smemOffset - 1];
                tmpBuffer = (K*)(s + smemOffset);
                for (int i = lane_id; i < batchSize; i += WARP_SIZE)
                    sMergedItems1[i] = i < size ? items[i] : init_limits;
            }
            else
            {
                sMergedItems1 = (K*)&items[0];
                smemOffset = smemOffset / sizeof(int);
                sMergedItems2 = (K*)(s + smemOffset);
                smemOffset += sizeof(K) * batchSize / sizeof(int);
                tmpIdx = (int *)&s[smemOffset - 1];
                tmpBuffer = (K*)(s + smemOffset);
            }

            __syncwarp();

            ibitonicSort_warp<K>(sMergedItems1, lane_id, batchSize);

            __syncwarp();

            if (lane_id == 0) {
                changeStatus(&status[0], AVAIL, INUSE);
            }
            __syncwarp();

            /* start handling partial batch */
            // Case 1: the heap has no full batch
            // TODO current not support size > batchSize, app should handle this
            if (*batchCount == 0 && size < batchSize) {
                // Case 1.1: partial batch is empty
                if (*partialBufferSize == 0) {
                    batchCopy_warp<K>(heapItems, sMergedItems1, batchSize, lane_id);
                    if (lane_id == 0) {
                        *partialBufferSize = size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncwarp();
                    // printf("%d first batch\n", lane_id);
                    return;
                }
                // Case 1.2: no full batch is generated
                // imergePath_warp is to modified to support for items_size < batchSize, by hzd
                else if (size + *partialBufferSize < batchSize) {
                    batchCopy_warp<K>(sMergedItems2, heapItems, batchSize, lane_id);

                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                               heapItems, sMergedItems1, tmpBuffer, lane_id, batchSize);
                    if (lane_id == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncwarp();
                    return;
                }
                // Case 1.3: a full batch is generated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy_warp<K>(sMergedItems2, heapItems, batchSize, lane_id);
                    if (lane_id == 0) {
                        atomicAdd(batchCount, 1);
                        changeStatus(&status[1], AVAIL, TARGET);
                        changeStatus(&status[1], TARGET, INUSE);
                    }
                    __syncwarp();
                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                               heapItems + batchSize, heapItems, tmpBuffer, lane_id, batchSize);
                    
                    if (lane_id == 0) {
                        *partialBufferSize += (size - batchSize);
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    __syncwarp();
                    return;
                }
            }
            // Case 2: the heap is non empty
            else {
                // Case 2.1: no full batch is generated
                if (size + *partialBufferSize < batchSize) {
                    batchCopy_warp<K>(sMergedItems2, heapItems, batchSize, lane_id);

                    // Merge insert batch with partial batch
                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                               sMergedItems1, sMergedItems2, tmpBuffer, lane_id, batchSize);
                    if (!lane_id) {
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncwarp();
                    batchCopy_warp(sMergedItems2, heapItems + batchSize, batchSize, lane_id);
                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                               heapItems + batchSize, heapItems, tmpBuffer, lane_id, batchSize);
                    if (lane_id == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    return;
                }
                // Case 2.2: a full batch is generated and needed to be propogated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy_warp(sMergedItems2, heapItems, batchSize, lane_id);
                    // Merge insert batch with partial batch, leave larger half in the partial batch
                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                            sMergedItems1, heapItems, tmpBuffer, lane_id, batchSize);
                    if (lane_id == 0) {
                        *partialBufferSize += (size - batchSize);
                    }
                    __syncwarp();
                }
            }
            /* end handling partial batch */

            if (lane_id == 0) {
                *tmpIdx = atomicAdd(batchCount, 1) + 1;
                changeStatus(&status[*tmpIdx], AVAIL, TARGET);
                if (*tmpIdx != 1) {
                    changeStatus(&status[1], AVAIL, INUSE);
                }
            }

            __syncwarp();

            int currentIdx = 1;
            int targetIdx = *tmpIdx;

            while(currentIdx != targetIdx) {

                if (lane_id == 0) {
                    *tmpIdx = 0;
                    if (status[targetIdx] == MARKED) {
                        *tmpIdx = 1;
                    }
                }
                __syncwarp();

                if (*tmpIdx == 1) break;

                if (lane_id == 0) {
                    changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                }
                __syncwarp();

                // move batch to shard memory
                batchCopy_warp<K>(sMergedItems2, 
                          heapItems + currentIdx * batchSize, batchSize, lane_id);

                if (sMergedItems1[batchSize - 1] <= sMergedItems2[0]) {
                    batchCopy_warp<K>(heapItems + currentIdx * batchSize,
                              sMergedItems1, batchSize, lane_id);

                    batchCopy_warp<K>(sMergedItems1, sMergedItems2, batchSize, lane_id);
                }
                else if (sMergedItems2[batchSize - 1] > sMergedItems1[0]) {
                    imergePath_warp<K>(sMergedItems1, sMergedItems2,
                            heapItems + currentIdx * batchSize, sMergedItems1, tmpBuffer, lane_id, batchSize);
                }
                currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
                if (lane_id == 0) {
                    if (currentIdx != targetIdx) {
                        changeStatus(&status[currentIdx], AVAIL, INUSE);
                    }
                }
                __syncwarp();
                //__syncwarp();
                //__syncthreads();
            }

            if (lane_id == 0) {
                atomicCAS(&status[targetIdx], TARGET, INUSE);
            }
            __syncwarp();

            if (status[targetIdx] == MARKED) {
                batchCopy_warp<K>(heapItems + batchSize, sMergedItems1, batchSize, lane_id);
                if (lane_id == 0) {
                    changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                    if (targetIdx != currentIdx) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                    }
                    changeStatus(&status[targetIdx], MARKED, TARGET);
                }
                __syncwarp();
                return;
            }

            batchCopy_warp<K>(heapItems + targetIdx * batchSize, sMergedItems1, batchSize, lane_id);

            if (lane_id == 0) {
                changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                changeStatus(&status[currentIdx], INUSE, AVAIL);
            }
            __syncwarp();

        }
};

#endif
