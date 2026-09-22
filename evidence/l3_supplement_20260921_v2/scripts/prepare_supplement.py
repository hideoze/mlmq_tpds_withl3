#!/usr/bin/env python3
"""Freeze isolated adapters. No solver algorithm in the workspace is edited."""
import hashlib,json,shutil,subprocess,tarfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/l3_supplement_20260921')
OUT.mkdir(exist_ok=False)
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def patch(p,a,b):
    s=p.read_text();assert s.count(a)==1,(p,a,s.count(a));p.write_text(s.replace(a,b))
with tarfile.open(ROOT/'tmp/nol3_213_preload/source.tgz') as t:
    t.extractall(OUT/'single_source',filter='data')
shutil.copytree(ROOT/'core',OUT/'single_source/core',dirs_exist_ok=True)
for d in ('SSSP','core'):
    for p in (ROOT/d).rglob('*'):
        if p.is_file() and p.suffix in ('.h','.cuh','.cu'):
            dest=OUT/'dual_source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for kind in ('single','dual'):
    src=OUT/(kind+'_source')
    shutil.copy2(ROOT/'scripts/multigpu/supplement_oracle.h',src/'SSSP/supplement_oracle.h')
    # Both adapters explicitly use the same representable budget, queue code and workers.
    p=src/'SSSP/sssp.cuh'
    if '#define GPU_MEMORY (2.0' in p.read_text():
        patch(p,'#define GPU_MEMORY (2.0 * 1024 * 1024 * 1024)','#define GPU_MEMORY INT_MAX')
    assert '#define GPU_MEMORY INT_MAX' in p.read_text()
    p=src/'SSSP/main.cu';patch(p,'#include "common.h"','#include "common.h"\n#include "supplement_oracle.h"')
    # In the dual adapter this covers every query; no timer boundary is moved.
    patch(p,'int node_data_check(int m, VALUE_TYPE *node_data, VALUE_TYPE *node_data_base)\n{','int node_data_check(int m, VALUE_TYPE *node_data, VALUE_TYPE *node_data_base)\n{\n    if(supplement_oracle_check(node_data,m)) return 1;')
    if kind=='single':
        p=src/'SSSP/sssp_run.cu'
        patch(p,'#include "sssp.cuh"','#include "sssp.cuh"\n#include "supplement_oracle.h"\n__device__ unsigned long long g_dq_clamp;')
        patch(p,'        free(actual);','        correct = !supplement_oracle_check(actual,m) && correct;\n        free(actual);')
    # Outside solve, disclose the physical capacity once per allocation.
    p=src/'core/cu_delta_queue/cu_delta_queue.cuh'
    patch(p,'        return INIT_SUCCESS;\n    }\n\n    init_status host_reinit','        printf("L2_CAPACITY budget=%d record_bytes=%zu buckets=%d per_bucket=%d allocated_records=%d counter_bits=%zu\\n",max_size,sizeof(eletype),bucketNum,total_size,all_total_size,8*sizeof(int));\n        return INIT_SUCCESS;\n    }\n\n    init_status host_reinit')
    build=OUT/(kind+'_build');build.mkdir()
    flags=['-DWORK_COUNT=false','-DMLMQ_WORKER_THREADS=512']
    if kind=='dual':flags += ['-D'+f for f in ('L3_COOPERATIVE_COLLECT=true','L3_DIRECT_RX=true','L3_RETAIN_TX=true','L3_WINDOW_MODE=2','L3_WORKER_RECOVERY=true','L3_TERM_WAIT_ACK=true','L3_ACK_SCAN=true','L3_ACK_WIDE_SCAN=true','L3_BOUNDARY_INDEX=true','L3_L2_FINAL_COUNTS=true','L3_CHAIN_SHORTCUTS=false','L3_IDLE_TOKEN_PROBE=true')]
    cmd=['nvcc',*[str(src/'SSSP'/f) for f in ('main.cu','csr_graph.cu','sssp_run.cu')],'-o',str(build/'mlmq'),*flags,'-O3','-m64','-gencode=arch=compute_80,code=sm_80','-rdc=true','-lcuda','-lcudart','-w','-I'+str(src/'core/include'),'-I/a100-data/wyh/boost_1_87_0/','-lcusparse']
    save(build/'command.json',cmd)
    with tarfile.open(build/'source.tgz','w:gz') as t:
        for name in ('SSSP','core'):t.add(src/name,arcname=name)
    with (build/'build.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    save(build/'status.json',dict(rc=r.returncode))
    if r.returncode:raise RuntimeError(str(build/'build.log'))
    save(build/'hashes.json',{f:sha(build/f) for f in ('mlmq','source.tgz')})
    print(kind,'BUILD_PASS',flush=True)
subprocess.run(['g++','-std=c++17','-O3',str(ROOT/'scripts/multigpu/supplement_graph.cpp'),'-o',str(OUT/'export_graph')],check=True)
print('PREPARE_PASS',flush=True)
