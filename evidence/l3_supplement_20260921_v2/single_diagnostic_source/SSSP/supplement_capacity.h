#pragma once
template<class Q>static void supplement_capacity(const Q&,int){std::exit(2);}
template<class E>static void supplement_capacity(const l2_delta_queue<E>&q,int gpu){
    std::vector<int>w(q.bucketNum),r(q.bucketNum);int done=0;
    if(cudaMemcpy(w.data(),q.write_reserve,4*q.bucketNum,cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(r.data(),q.bucket_read_done,4*q.bucketNum,cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(&done,q.read_done,4,cudaMemcpyDeviceToHost)!=cudaSuccess)std::exit(2);
    long long total=0;for(int b=0;b<q.bucketNum;++b){
        if(w[b]<0 || w[b]>q.total_size || r[b]!=w[b])std::exit(2);total+=w[b];
        printf("CAPACITY_BUCKET gpu=%d bucket=%d capacity=%d writes=%d reads=%d no_wrap=1\n",gpu,b,q.total_size,w[b],r[b]);
    }if(total!=done || total>=INT_MAX)std::exit(2);
    printf("CAPACITY_QUERY gpu=%d total=%lld completed=%d guarded=1\n",gpu,total,done);
}
