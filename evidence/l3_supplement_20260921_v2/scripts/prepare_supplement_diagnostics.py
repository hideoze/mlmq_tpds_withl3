#!/usr/bin/env python3
"""Isolated capacity guards and sequential-source test adapter; not timed data."""
import hashlib,json,shutil,subprocess,tarfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/l3_supplement_20260921')
def patch(p,a,b):
    s=p.read_text();assert s.count(a)==1,(p,a,s.count(a));p.write_text(s.replace(a,b))
for kind in ('single','dual'):
    src=BASE/(kind+'_diagnostic_source');src.mkdir(exist_ok=False)
    for p in (BASE/(kind+'_source')).rglob('*'):
        if p.is_file() and p.suffix in ('.h','.cuh','.cu'):
            dest=src/p.relative_to(BASE/(kind+'_source'));dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Every atomic reservation is checked BEFORE payload indexing. Each increment <=32.
    # Thus a completed checked query cannot have crossed capacity or wrapped its write counter.
    p=src/'core/cu_delta_queue/cu_delta_queue.cuh'
    patch(p,'current_reserve = atomicAdd(&write_reserve[dst_bucket_id], write_bucket_num);','current_reserve = atomicAdd(&write_reserve[dst_bucket_id], write_bucket_num);\n                assert(current_reserve >= 0 && (long long)current_reserve + write_bucket_num <= total_size);')
    # Read reservation counters can advance ahead of writes: bound them independently.
    patch(p,'local_read_ptr[vec_id] = atomicAdd(&read_ptr[vec_id], current_read_size);','local_read_ptr[vec_id] = atomicAdd(&read_ptr[vec_id], current_read_size);\n                assert(local_read_ptr[vec_id] >= 0 && (long long)local_read_ptr[vec_id]+current_read_size < INT_MAX);')
    header='''#pragma once
template<class Q>static void supplement_capacity(const Q&,int){std::exit(2);}
template<class E>static void supplement_capacity(const l2_delta_queue<E>&q,int gpu){
    std::vector<int>w(q.bucketNum),r(q.bucketNum);int done=0;
    if(cudaMemcpy(w.data(),q.write_reserve,4*q.bucketNum,cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(r.data(),q.bucket_read_done,4*q.bucketNum,cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(&done,q.read_done,4,cudaMemcpyDeviceToHost)!=cudaSuccess)std::exit(2);
    long long total=0;for(int b=0;b<q.bucketNum;++b){
        if(w[b]<0 || w[b]>q.total_size || r[b]!=w[b])std::exit(2);total+=w[b];
        printf("CAPACITY_BUCKET gpu=%d bucket=%d capacity=%d writes=%d reads=%d no_wrap=1\\n",gpu,b,q.total_size,w[b],r[b]);
    }if(total!=done || total>=INT_MAX)std::exit(2);
    printf("CAPACITY_QUERY gpu=%d total=%lld completed=%d guarded=1\\n",gpu,total,done);
}
'''
    (src/'SSSP/supplement_capacity.h').write_text(header)
    p=src/'SSSP/sssp_run.cu'
    patch(p,'#include "sssp.cuh"','#include "sssp.cuh"\n#include <vector>\n#include "supplement_capacity.h"')
    if kind=='dual':
        patch(p,'#if (MLMQ_WORKER_THREADS == 384 || MLMQ_WORKER_THREADS == 320)','#if (MLMQ_WORKER_THREADS == 512 || MLMQ_WORKER_THREADS == 384 || MLMQ_WORKER_THREADS == 320)')
        # This hook is after device synchronization; overload applies only to DQ.
        patch(p,'    std::vector<int> reads(q.bucketNum), writes(q.bucketNum);','    supplement_capacity(q,gpu);\n    std::vector<int> reads(q.bucketNum), writes(q.bucketNum);')
        p=src/'SSSP/main.cu'
        patch(p,'#include "common.h"','#include "common.h"\n#include <sstream>')
        patch(p,'\tfor (int sample = 0; sample < repeats + warmups; ++sample) {','''
    std::vector<int> supplement_sources;
    if(const char *sequence=std::getenv("L3_SUPPLEMENT_SOURCES")) {
        std::stringstream stream(sequence);std::string token;
        while(std::getline(stream,token,',')){int v=std::stoi(token);if(v<0 || v>=g.nnodes)return 2;supplement_sources.push_back(v);}
        if(supplement_sources.size()!=size_t(repeats+warmups))return 2;
    }
    for (int sample = 0; sample < repeats + warmups; ++sample) {
    if(!supplement_sources.empty()) {
        src=supplement_sources[sample];
        const char *dir=std::getenv("L3_SUPPLEMENT_ORACLE_DIR");if(!dir)return 2;
        std::string path=std::string(dir)+"/s"+std::to_string(src)+".i32";
        setenv("L3_SUPPLEMENT_ORACLE",path.c_str(),1);
        FILE *f=fopen(path.c_str(),"rb");if(!f)return 2;
        bool ok=fread(node_data_base,sizeof(int),g.nnodes,f)==size_t(g.nnodes) && fgetc(f)==EOF;
        fclose(f);if(!ok)return 2;
        printf("SEQUENTIAL_L3_QUERY sample=%d source=%d gpu_count=%d\\n",sample,src,n_gpu);
    }
''')
    else:
        patch(p,'        const double solve_end = baseline_ms();','        const double solve_end = baseline_ms();\n        supplement_capacity(mlmq.q2,0);')
    build=BASE/(kind+'_diagnostic_build');build.mkdir(exist_ok=False)
    cmd=json.loads((BASE/(kind+'_build/command.json')).read_text())
    cmd=[x.replace(str(BASE/(kind+'_source')),str(src)).replace(str(BASE/(kind+'_build')),str(build)) for x in cmd]
    cmd+=['--maxrregcount=64','-Xptxas=-v']
    if kind=='dual':cmd+=['-DQUERY_WORKSPACE_DIAG=true']
    (build/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with tarfile.open(build/'source.tgz','w:gz') as t:
        for folder in ('SSSP','core'):t.add(src/folder,arcname=folder)
    with (build/'build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    (build/'status.json').write_text(json.dumps(dict(rc=r.returncode)))
    if r.returncode:raise RuntimeError(str(build/'build.log'))
    print(kind,'DIAGNOSTIC_BUILD_PASS',flush=True)
