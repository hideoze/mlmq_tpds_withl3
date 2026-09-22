#pragma once
#if (L3_CONTINUATION == true)
static_assert(L3_TILE_LOAN_RESULT_CAP>=L3_TILE_LOAN_MAX_SEEDS,"journal must retain every seed");
static_assert(L3_CONTINUATION_ROWS>0,"continuation must have a finite positive row budget");
#if (L3_CONTINUATION_WARP == true)
// Same single-slot ownership, admission, budget and row certificates. Only
// journal minimum/ID searches become warp-cooperative; lane0 alone mutates it.
__device__ __forceinline__ void l3_continuation_execute_warp(
    const l3_channel_view &c,int lane)
{
    NODE_TYPE *j=c.loan_in.results;
    VALUE_TYPE *cert=c.loan_in.result_dists;
    int seeds=0,count=0,rows=0,continued=0;
    if(!lane) {
        atomicExch(c.loan_in.ready,0);
        seeds=atomicAdd(c.loan_in.seed_count,0);
        for(int s=0;s<seeds;++s) {
            int v=c.loan_in.tasks[s].id,pos=0;
            VALUE_TYPE d=c.loan_in.task_dists[s];
            while(pos<count && j[pos].id!=v) ++pos;
            if(pos==count) { j[count]=node_struct(v,d); cert[count++]=DIST_MAX; }
            else if(d<j[pos].get_data()) j[pos]=node_struct(v,d);
        }
    }
    count=__shfl_sync(FULL_MASK,count,0); __syncwarp();
    for(int step=0;step<L3_CONTINUATION_ROWS;++step) {
        int pick=L3_TILE_LOAN_RESULT_CAP;
        VALUE_TYPE d=DIST_MAX;
        for(int i=lane;i<count;i+=WARP_SIZE) {
            int local=j[i].id-1-c.peer_v_begin;
            VALUE_TYPE candidate=j[i].get_data();
            if(local>=0 && local<c.peer_v_local && candidate<cert[i] &&
               (candidate<d || (candidate==d && i<pick))) { d=candidate; pick=i; }
        }
        for(int offset=16;offset;offset/=2) {
            VALUE_TYPE other=__shfl_down_sync(FULL_MASK,d,offset);
            int index=__shfl_down_sync(FULL_MASK,pick,offset);
            if(lane+offset<WARP_SIZE && (other<d || (other==d && index<pick))) { d=other; pick=index; }
        }
        pick=__shfl_sync(FULL_MASK,pick,0); d=__shfl_sync(FULL_MASK,d,0);
        if(pick==L3_TILE_LOAN_RESULT_CAP) break;
        int first=0,end=0;
        if(!lane) {
            int local=j[pick].id-1-c.peer_v_begin;
            first=c.peer_row_ptr[local]; end=c.peer_row_ptr[local+1];
        }
        first=__shfl_sync(FULL_MASK,first,0); end=__shfl_sync(FULL_MASK,end,0);
        if(end-first>L3_TILE_LOAN_RESULT_CAP-count) break;
        for(int e=first;e<end;++e) {
            int v=0; VALUE_TYPE w=0;
            if(!lane) { v=c.peer_col_idx[e]+1; w=c.peer_edge_data[e]; }
            v=__shfl_sync(FULL_MASK,v,0); w=__shfl_sync(FULL_MASK,w,0);
            if(w>DIST_MAX-d) continue;
            int pos=L3_TILE_LOAN_RESULT_CAP;
            for(int i=lane;i<count;i+=WARP_SIZE) if(j[i].id==v) pos=min(pos,i);
            for(int offset=16;offset;offset/=2) {
                int other=__shfl_down_sync(FULL_MASK,pos,offset);
                if(lane+offset<WARP_SIZE) pos=min(pos,other);
            }
            pos=__shfl_sync(FULL_MASK,pos,0);
            if(!lane) {
                VALUE_TYPE nd=d+w;
                if(pos==L3_TILE_LOAN_RESULT_CAP) { j[count]=node_struct(v,nd); cert[count++]=DIST_MAX; }
                else if(nd<j[pos].get_data()) j[pos]=node_struct(v,nd);
            }
            count=__shfl_sync(FULL_MASK,count,0); __syncwarp();
        }
        if(!lane) {
            bool seed=false;
            for(int s=0;s<seeds;++s) seed=seed || c.loan_in.tasks[s].id==j[pick].id;
            continued+=!seed; cert[pick]=min(cert[pick],d); ++rows;
        }
        __syncwarp();
    }
    if(!lane) {
        atomicExch(c.loan_in.result_count,count);
        atomicExch(c.loan_in.requeue_count,0);
        auto m=c.loan_in.continuation_metrics;
        m[0]+=1; m[1]+=seeds; m[2]+=rows; m[3]+=continued; m[4]+=count;
        __threadfence_system();
        l3_loan_state_store(c.loan_in.state,L3_TILE_LOAN_RETURNED);
    }
    __syncwarp();
}
#endif
// Correctness-first production adapter. Return buffers are helper-local.
// results={id,label}, result_dists=full-row certificate (INF if unfinished).
// One existing designated helper owns the journal; ordinary L1 is untouched.
__device__ __forceinline__ bool l3_continuation_service(
    const l3_channel_view &c,int lane,int &active_generation)
{
    int state=0;
    if(!lane) state=l3_tile_loan_load_state(c.loan_in.state);
    state=__shfl_sync(FULL_MASK,state,0);
    if(state==L3_TILE_LOAN_APPLIED) {
        if(!lane && active_generation!=0 &&
           active_generation==atomicAdd(c.loan_in.generation,0))
            l3_loan_state_store(c.loan_in.state,L3_TILE_LOAN_ACKED);
        __syncwarp();
        return true;
    }
    if(state!=L3_TILE_LOAN_READY) return false;
    int claimed=0;
    if(!lane) claimed=l3_loan_state_cas(c.loan_in.state,L3_TILE_LOAN_READY,
        L3_TILE_LOAN_EXECUTING)==L3_TILE_LOAN_READY;
    claimed=__shfl_sync(FULL_MASK,claimed,0);
    if(!claimed) return false;
    int generation=0;
    if(!lane) generation=atomicAdd(c.loan_in.generation,0);
    active_generation=__shfl_sync(FULL_MASK,generation,0);
#if (L3_CONTINUATION_WARP == true)
    l3_continuation_execute_warp(c,lane);
#else
    if(!lane) {
        atomicExch(c.loan_in.ready,0);
        NODE_TYPE *j=c.loan_in.results;
        VALUE_TYPE *cert=c.loan_in.result_dists;
        int seeds=atomicAdd(c.loan_in.seed_count,0), count=0,rows=0,continued=0;
        for(int s=0;s<seeds;++s) {
            int v=c.loan_in.tasks[s].id, pos=0;
            while(pos<count && j[pos].id!=v) ++pos;
            VALUE_TYPE d=c.loan_in.task_dists[s];
            if(pos==count) { j[count]=node_struct(v,d); cert[count++]=DIST_MAX; }
            else if(d<j[pos].get_data()) j[pos]=node_struct(v,d);
        }
        for(int step=0;step<L3_CONTINUATION_ROWS;++step) {
            int pick=-1;
            for(int i=0;i<count;++i) {
                int local=j[i].id-1-c.peer_v_begin;
                if(local>=0 && local<c.peer_v_local && j[i].get_data()<cert[i] &&
                   (pick<0 || j[i].get_data()<j[pick].get_data())) pick=i;
            }
            if(pick<0) break;
            int local=j[pick].id-1-c.peer_v_begin;
            int first=c.peer_row_ptr[local],end=c.peer_row_ptr[local+1];
            // Conservative complete-row admission. Never lose a partial row.
            if(end-first>L3_TILE_LOAN_RESULT_CAP-count) break;
            VALUE_TYPE d=j[pick].get_data();
            for(int e=first;e<end;++e) {
                int v=c.peer_col_idx[e]+1;
                VALUE_TYPE w=c.peer_edge_data[e];
                if(w>DIST_MAX-d) continue; // no signed distance overflow
                VALUE_TYPE nd=d+w;
                int pos=0; while(pos<count && j[pos].id!=v) ++pos;
                if(pos==count) { j[count]=node_struct(v,nd); cert[count++]=DIST_MAX; }
                else if(nd<j[pos].get_data()) j[pos]=node_struct(v,nd);
            }
            bool seed=false;
            for(int s=0;s<seeds;++s) seed=seed || c.loan_in.tasks[s].id==j[pick].id;
            continued+=!seed;
            cert[pick]=min(cert[pick],d); ++rows;
        }
        atomicExch(c.loan_in.result_count,count);
        atomicExch(c.loan_in.requeue_count,0); // persistent owner enqueue cursor
        // Fixed writer per metric; host reads only after query completion.
        auto m=c.loan_in.continuation_metrics;
        m[0]+=1; m[1]+=seeds; m[2]+=rows; m[3]+=continued; m[4]+=count;
        __threadfence_system();
        l3_loan_state_store(c.loan_in.state,L3_TILE_LOAN_RETURNED);
#if (L3_TILE_LOAN_DIAG == true)
        printf("L3_CONT_EXEC gen=%d seeds=%d records=%d rows=%d continued=%d\n",generation,seeds,count,rows,continued);
#endif
    }
#endif
    __syncwarp();
    return true;
}

template<typename Q>
__device__ __forceinline__ bool l3_continuation_apply(Q mlmq,
    const l3_channel_view &c,int bid,int wid,int lane,unsigned *debug,
    int begin,int end,VALUE_TYPE *dist,NODE_TYPE *scratch,int &work)
{
    int state=0;
    if(!lane) state=l3_tile_loan_load_state(c.loan_out.state);
    state=__shfl_sync(FULL_MASK,state,0);
    if(state!=L3_TILE_LOAN_RETURNED) return false;
    int count=0,cursor=0;
    if(!lane) {
        count=atomicAdd(c.loan_out.result_count,0);
        cursor=atomicAdd(c.loan_out.requeue_count,0);
    }
    count=__shfl_sync(FULL_MASK,count,0);
    cursor=__shfl_sync(FULL_MASK,cursor,0);
    // Replaying this first pass is idempotent. Crucially, enqueue is based on
    // remaining distance/certificate debt, NOT this pass's atomicMin winners.
    for(int i=lane;i<count;i+=WARP_SIZE) {
        NODE_TYPE r=c.loan_out.results[i];
        if(r.id>begin && r.id<=end) atomicMin(&dist[r.id-begin],r.get_data());
        else {
            bool ignored=false;
#if (WORK_COUNT == true)
            relax_dst(r.id,r.get_data(),dist,begin,end,c.candidate_values,
                c.candidate_mark,c.candidate_hint,c.candidate_hint2,c.peer_cache,
                c.peer_v_begin,ignored,work);
#else
            relax_dst(r.id,r.get_data(),dist,begin,end,c.candidate_values,
                c.candidate_mark,c.candidate_hint,c.candidate_hint2,c.peer_cache,
                c.peer_v_begin,ignored);
#endif
        }
    }
    __threadfence_system(); __syncwarp(); // all lanes' output before certification
    if(cursor<count) {
        int i=cursor+lane;
        bool need=false; NODE_TYPE r=node_struct(0,0);
        if(i<count) {
            r=c.loan_out.results[i];
            if(r.id>begin && r.id<=end) {
                VALUE_TYPE current=*((volatile VALUE_TYPE*)&dist[r.id-begin]);
                need=current<c.loan_out.result_dists[i];
                r=node_struct(r.id,current);
            }
        }
        unsigned mask=__ballot_sync(FULL_MASK,need);
        int n=count_bit(mask),pos=count_bit(set_bits(mask,0,lane,WARP_SIZE));
        if(need) scratch[pos]=r;
        __syncwarp();
        write_status status=WRITE_SUCCESS;
        if(n) { int num=n; status=mlmq.write_through(scratch,num,bid,wid,lane,debug); }
        bool ok=__ballot_sync(FULL_MASK,status==WRITE_SUCCESS)==FULL_MASK;
        if(!lane && ok) atomicExch(c.loan_out.requeue_count,min(count,cursor+WARP_SIZE));
        __syncwarp();
        return true; // on failure retain cursor, journal and original q2 credit
    }
#if (L3_COMPLETED_ROWS == true)
    // Every output and required q2 write succeeded before reaching this point.
    // Each journal vertex is unique; the evidence survives slot reuse.
    for(int i=lane;i<count;i+=WARP_SIZE) {
        int v=c.loan_out.results[i].id;
        if(v>begin && v<=end) l3_publish_completed_row(v-begin,c.loan_out.result_dists[i]);
    }
    __threadfence(); __syncwarp();
#endif
    if(!lane) {
        int seeds=atomicAdd(c.loan_out.seed_count,0);
        mlmq.update_done(seeds);
        c.loan_out.continuation_metrics[5]+=seeds;
    }
    __threadfence_system(); __syncwarp();
    if(!lane) {
        l3_loan_state_store(c.loan_out.state,L3_TILE_LOAN_APPLIED);
#if (L3_TILE_LOAN_DIAG == true)
        printf("L3_CONT_APPLY gen=%d records=%d released=%d\n",
            atomicAdd(c.loan_out.generation,0),count,atomicAdd(c.loan_out.seed_count,0));
#endif
    }
    return true;
}
#endif
