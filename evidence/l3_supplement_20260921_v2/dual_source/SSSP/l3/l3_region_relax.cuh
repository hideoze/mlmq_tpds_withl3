#pragma once
#if (L3_REGION_RELAX == true)
// One warp owns a region closure while retaining the original input L2 credit.
// Its register vector starts from authoritative upper bounds and reaches the
// internal Bellman fixed point in <=32 rounds (nonnegative weights, <=32 nodes).
// Concurrent lower updates retain their own notifications/recovery obligation.
// No cross-GPU distance store or new queue/ACK protocol is introduced.
template <typename QUEUE_TYPE>
__device__ __forceinline__ int l3_region_process(
    int *RowPtr,int *ColIdx,VALUE_TYPE *edge_data,VALUE_TYPE *node_data,
    NODE_TYPE *node_in,NODE_TYPE *node_out,int &node_in_num,QUEUE_TYPE mlmq,
    int block_id,int warp_id,int lane_id,unsigned *debug_time,int v_begin,int v_end,
    VALUE_TYPE *remote_cand,unsigned *remote_mark,unsigned *mark_hint,unsigned *mark_hint2,
    VALUE_TYPE *peer_cache,int peer_v_begin,VALUE_TYPE *last_processed)
{
    int node_out_num=0;
    const auto index=g_l3_regions;
    for(int input=0;input<node_in_num;input+=WARP_SIZE) {
        int region=-1;
        if(input+lane_id<node_in_num) {
            auto item=node_in[input+lane_id];const int local=item.id-v_begin;
            if(local>0&&local<=v_end-v_begin&&(!last_processed||item.get_data()<*((volatile VALUE_TYPE *)(last_processed+local))))
                region=index.map[local]&~31;
        }
        unsigned pending=__ballot_sync(FULL_MASK,region>=0);
        while(pending) {
            int leader=__ffs(pending)-1;
            const int base=__shfl_sync(FULL_MASK,region,leader);
            pending &= ~__ballot_sync(FULL_MASK,region==base);
            const int flat=base+lane_id,id=index.members[flat];
            VALUE_TYPE distance=id?*((volatile VALUE_TYPE *)(node_data+id-v_begin)):DIST_MAX;
            const int first=index.rows[flat],degree=index.rows[flat+1]-first;
            const int max_degree=__reduce_max_sync(FULL_MASK,unsigned(degree));
            int rounds=0;
            #pragma unroll 1
            for(int turn=0;turn<32;++turn) {
                VALUE_TYPE next=distance;
                for(int j=0;j<max_degree;++j) {
                    int source=j<degree?index.sources[first+j]:lane_id;
                    const VALUE_TYPE other=__shfl_sync(FULL_MASK,distance,source);
                    if(j<degree&&other!=DIST_MAX) {
                        const long long sum=(long long)other+index.weights[first+j];
                        if(sum<next)next=VALUE_TYPE(sum);
                    }
                }
                ++rounds;
                const bool changed=__any_sync(FULL_MASK,next<distance);
                distance=next;if(!changed)break;
            }
            bool active=id&&distance!=DIST_MAX&&(!last_processed||distance<*((volatile VALUE_TYPE *)(last_processed+id-v_begin)));
            bool materialized=false;
            if(id&&distance<*((volatile VALUE_TYPE *)(node_data+id-v_begin)))
                materialized=distance<atomicMin(node_data+id-v_begin,distance);
            const int edge_begin=active?RowPtr[id-v_begin-1]:0;
            const int edges=active?RowPtr[id-v_begin]-edge_begin:0;
            const int max_edges=__reduce_max_sync(FULL_MASK,unsigned(edges));
#if (L3_REGION_DIAG == true)
            unsigned long long external_attempts=0;
#endif
            for(int j=0;j<max_edges;++j) {
                bool won=false;int dst=0;VALUE_TYPE value=DIST_MAX;
                if(j<edges) {
                    const int e=edge_begin+j;dst=ColIdx[e]+1;
                    const bool local=dst-1>=v_begin&&dst-1<v_end;
                    if(!local||(index.map[dst-v_begin]&~31)!=base) {
                        const long long sum=(long long)distance+edge_data[e];
                        if(sum<DIST_MAX) {
                            value=VALUE_TYPE(sum);
                            relax_dst(dst,value,node_data,v_begin,v_end,remote_cand,remote_mark,mark_hint,mark_hint2,peer_cache,peer_v_begin,won);
#if (L3_REGION_DIAG == true)
                            ++external_attempts;
#endif
                        }
                    }
                }
                const unsigned winners=__ballot_sync(FULL_MASK,won);
                int position=node_out_num+__popc(winners&((1u<<lane_id)-1u));
                node_out_num+=__popc(winners);
                if(won)node_out[position]=node_struct(dst,value);
                __syncwarp();
                check_out_buffer<QUEUE_TYPE>(node_out,node_out_num,mlmq,block_id,warp_id,lane_id,debug_time);
                __syncwarp();
            }
            // Internal edges are closed, external winners are either in the
            // retained output buffer/L1/L2 or authoritative remote_mark.
            if(active&&last_processed)atomicMin(last_processed+id-v_begin,distance);
#if (L3_REGION_DIAG == true)
            unsigned updates=__popc(__ballot_sync(FULL_MASK,materialized));
            for(int step=16;step;step/=2)external_attempts+=__shfl_down_sync(FULL_MASK,external_attempts,step);
            if(!lane_id) {
                atomicAdd(g_l3_region_counts,1ull);atomicAdd(g_l3_region_counts+1,(unsigned long long)rounds);
                atomicAdd(g_l3_region_counts+2,(unsigned long long)updates);atomicAdd(g_l3_region_counts+3,external_attempts);
            }
#endif
            __syncwarp();
        }
    }
    while(node_out_num>0) {
        int fill=node_out_num;
        if(mlmq.write(node_out,fill,block_id,warp_id,lane_id,debug_time)==0)node_out_num-=fill;
    }
    node_in_num=0;return 0;
}
#endif
