#pragma once
// Host-only, after both solver threads have joined and before any query reset.
// No repair, device write, reference seeding or timing-path instrumentation.
#ifdef TYPE_INT
void sssp_audit_final(int n_gpu, int n, const int *rows, const int *dst,
                      const VALUE_TYPE *weights, const VALUE_TYPE *actual,
                      const VALUE_TYPE *reference)
{
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) {
            fprintf(stderr, "FINAL_AUDIT_ERROR %s\n", cudaGetErrorString(status));
            std::exit(2);
        }
    };
    std::vector<VALUE_TYPE> processed(n, DIST_MAX);
    std::vector<std::vector<VALUE_TYPE>> candidates(n_gpu), caches(n_gpu);
    std::vector<std::vector<unsigned>> marks(n_gpu);
    for (int gpu=0; gpu<n_gpu; ++gpu) {
        auto &c = gctx[gpu];
        check(cudaSetDevice(gpu));
        check(cudaDeviceSynchronize());
        if (c.last_processed)
            check(cudaMemcpy(processed.data()+c.v_begin,c.last_processed+1,
                             c.v_local*sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost));
#if (L3_COMPLETED_ROWS == true)
        std::vector<VALUE_TYPE> completed(c.v_local);
        check(cudaMemcpy(completed.data(),c.continuation_completed+1,
                         c.v_local*sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost));
        size_t certified=0,raw_pending=0;
        for(int i=0;i<c.v_local;++i) {
            raw_pending+=actual[c.v_begin+i]<processed[c.v_begin+i];
            certified+=completed[i]<DIST_MAX;
            processed[c.v_begin+i]=std::min(processed[c.v_begin+i],completed[i]);
        }
        printf("FINAL_COMPLETED gpu=%d entries=%zu raw_pending=%zu\n",gpu,certified,raw_pending);
#endif
        size_t pending_local=0,marked_words=0,stranded=0,cache_ahead=0;
        for(int u=c.v_begin;u<c.v_end;++u) pending_local+=actual[u]<processed[u];
        if (c.remote_cand && c.remote_mark) {
            candidates[gpu].resize(c.peer_v_local+1);
            caches[gpu].assign(c.peer_v_local+1,DIST_MAX);
            marks[gpu].resize((c.peer_v_local+31)/32);
            check(cudaMemcpy(candidates[gpu].data(),c.remote_cand,
                             candidates[gpu].size()*sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost));
            if(c.peer_cache) check(cudaMemcpy(caches[gpu].data(),c.peer_cache,
                             caches[gpu].size()*sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(marks[gpu].data(),c.remote_mark,
                             marks[gpu].size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
            for(auto word:marks[gpu]) marked_words+=word!=0;
            for(int i=1;i<=c.peer_v_local;++i) {
                int v=c.peer_v_begin+i-1;
                bool lost=candidates[gpu][i]<actual[v];
                bool ahead=caches[gpu][i]<actual[v];
                if ((lost || ahead) && stranded+cache_ahead<32)
                    printf("FINAL_CAND gpu=%d dst=%d candidate=%d cache=%d actual=%d reference=%d marked=%u\n",
                           gpu,v,candidates[gpu][i],caches[gpu][i],actual[v],reference[v],
                           (marks[gpu][(i-1)/32]>>((i-1)%32))&1u);
                stranded+=lost; cache_ahead+=ahead;
            }
        }
        int term=-1,request=-1,exit_code=-1;
        if(c.l3_term_state) check(cudaMemcpy(&term,c.l3_term_state,sizeof(int),cudaMemcpyDeviceToHost));
        if(c.l3_term_req) check(cudaMemcpy(&request,c.l3_term_req,sizeof(int),cudaMemcpyDeviceToHost));
        if(c.global_exit) check(cudaMemcpy(&exit_code,c.global_exit,sizeof(int),cudaMemcpyDeviceToHost));
        printf("FINAL_STATE gpu=%d term=%d request=%d exit=%d pending_local=%zu marked_words=%zu stranded=%zu cache_ahead=%zu\n",
               gpu,term,request,exit_code,pending_local,marked_words,stranded,cache_ahead);
#if (L3_TILE_LOAN == true)
        int loan_generation = -1, loan_state = -1, loan_seeds = -1;
        if (c.loan_out.generation)
            check(cudaMemcpy(&loan_generation, c.loan_out.generation,
                             sizeof(int), cudaMemcpyDeviceToHost));
        if (c.loan_out.state)
            check(cudaMemcpy(&loan_state, c.loan_out.state,
                             sizeof(int), cudaMemcpyDeviceToHost));
        if (c.loan_out.seed_count)
            check(cudaMemcpy(&loan_seeds, c.loan_out.seed_count,
                             sizeof(int), cudaMemcpyDeviceToHost));
        printf("FINAL_LOAN gpu=%d generation=%d state=%d last_seeds=%d\n",
               gpu, loan_generation, loan_state, loan_seeds);
#if (L3_MULTI_PRODUCER == true)
        unsigned long long gate=0,attempts=0,published=0,nonzero=0;
        std::vector<unsigned long long> producers(L3_PRODUCER_COUNTER_WORDS);
        check(cudaMemcpy(&gate,c.loan_gate,sizeof(gate),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(producers.data(),c.loan_producers,L3_PRODUCER_COUNTER_WORDS*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
        int active=0;
        for(int b=0;b<4096;++b) {
            attempts+=producers[b]; published+=producers[4096+b];
            active+=producers[4096+b]!=0;
            if(b) nonzero+=producers[4096+b];
        }
        unsigned count=unsigned(gate)&0x7fffffffu,closed=(gate>>31)&1u;
        printf("FINAL_PRODUCER gpu=%d epoch=%u gate_epoch=%u closed=%u registered=%u attempts=%llu published=%llu active_blocks=%d nonzero_block=%llu\n",
            gpu,c.loan_epoch,unsigned(gate>>32),closed,count,attempts,published,active,nonzero);
#if (L3_PRODUCER_STOP_POLL == true)
        unsigned latched=0;
        for(int b=0;b<4096;++b) latched+=producers[8192+b]!=0;
        printf("FINAL_POLL gpu=%d latched_blocks=%u\n",gpu,latched);
#endif
        if(count || unsigned(gate>>32)!=c.loan_epoch || published!=static_cast<unsigned long long>(loan_generation) ||
           (n_gpu>1 && g_l3_tile_loan_enabled && !closed)) {
            fprintf(stderr,"FINAL_AUDIT_ERROR producer gate or publication mismatch\n"); std::exit(2);
        }
#endif
#if (L3_CONTINUATION == true)
        unsigned long long cont[6]={};
        check(cudaMemcpy(cont,c.loan_out.continuation_metrics,sizeof(cont),cudaMemcpyDeviceToHost));
        printf("FINAL_CONT gpu=%d exec=%llu seeds=%llu rows=%llu continued=%llu records=%llu released=%llu\n",
            gpu,cont[0],cont[1],cont[2],cont[3],cont[4],cont[5]);
#endif
#endif
        if(c.bulk_inbox_state) {
            int state[2],ack[2],epoch[2],generation[2];
            check(cudaMemcpy(state,c.bulk_inbox_state,sizeof(state),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(ack,c.bulk_inbox_ack,sizeof(ack),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(epoch,c.bulk_inbox_epoch,sizeof(epoch),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(generation,c.bulk_inbox_generation,sizeof(generation),cudaMemcpyDeviceToHost));
            for(int slot=0;slot<2;++slot)
                printf("FINAL_INBOX gpu=%d slot=%d state=%d ack=%d epoch=%d generation=%d\n",
                       gpu,slot,state[slot],ack[slot],epoch[slot],generation[slot]);
        }
    }
    size_t residuals=0,cross_residuals=0,mismatches=0;
    auto owner = [&](int v) { for(int gpu=0;gpu<n_gpu;++gpu)
        if(v>=gctx[gpu].v_begin && v<gctx[gpu].v_end) return gpu; return -1; };
    for(int u=0;u<n;++u) {
        mismatches+=actual[u]!=reference[u];
        if(actual[u]==DIST_MAX) continue;
        for(int e=rows[u];e<rows[u+1];++e) {
            int v=dst[e]; long long relaxed=(long long)actual[u]+weights[e];
            if(relaxed>=actual[v]) continue;
            bool cross=owner(u)!=owner(v); cross_residuals+=cross;
            if(residuals<32)
                printf("FINAL_EDGE src=%d dst=%d weight=%d src_actual=%d src_processed=%d src_reference=%d dst_actual=%d dst_reference=%d cross=%d\n",
                       u,v,weights[e],actual[u],processed[u],reference[u],actual[v],reference[v],int(cross));
            ++residuals;
        }
    }
    printf("FINAL_AUDIT mismatches=%zu residual_edges=%zu cross_residual_edges=%zu\n",mismatches,residuals,cross_residuals);
}
#endif
