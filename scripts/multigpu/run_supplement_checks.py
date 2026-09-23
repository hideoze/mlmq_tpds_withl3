#!/usr/bin/env python3
"""Run only after same-graph matrix; separate checked runs from performance."""
import json,os,struct,subprocess,sys,shutil
from pathlib import Path
from run_usa_road_matrix import run_one,sha256,parse_bench
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/l3_supplement_20260921');OUT=BASE/'checks'
assert os.environ.get('SLURM_JOB_ID') and (BASE/'matrix/complete.json').exists()
OUT.mkdir(exist_ok=False)
shutil.copy2(__file__,OUT/'runner.py')
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def command(name,cmd,env=None,timeout=120):
    try:r=subprocess.run(list(map(str,cmd)),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout);rc,raw=r.returncode,r.stdout
    except subprocess.TimeoutExpired as e:rc,raw=124,e.stdout or b'';raw=raw.decode(errors='replace') if isinstance(raw,bytes) else raw
    (OUT/(name+'.log')).write_text(raw+f'\nRUN_RC={rc}\n');save(name+'.json',dict(command=list(map(str,cmd)),rc=rc));return rc,raw
for name,cmd in [('gpu',['nvidia-smi']),('topology',['nvidia-smi','topo','-m']),('apps',['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])]:
    rc,raw=command(name,cmd);assert rc==0
    if name=='apps':assert not raw.strip(),'GPU occupied'
save('manifest.json',dict(job=os.environ['SLURM_JOB_ID'],binaries={str(p.relative_to(BASE)):sha256(p) for p in [BASE/'test_primitives',BASE/'test_dq',BASE/'test_capacity',BASE/'single_diagnostic_build/mlmq',BASE/'dual_diagnostic_build/mlmq']}))
problems=[]
for name,args in [('primitives',[BASE/'test_primitives']),('dq_publication',[BASE/'test_dq']),('capacity_full',[BASE/'test_capacity']),('capacity_overflow',[BASE/'test_capacity','overflow'])]:
    rc,raw=command(name,args)
    if rc or 'PASS' not in raw:problems.append(dict(test=name,rc=rc))
    print('SMALL_TEST',name,'rc',rc,flush=True)

# A small connected component spans both owners; vertex 2050 is isolated.
# Parallel high/low cost fan-in and zero cycles create candidate contention.
fixture=OUT/'fixture.gr';n=2052;adj=[[] for _ in range(n)]
for u in range(2049):
    adj[u].append((u+1,1+u%3));adj[u+1].append((u,1+u%5))
for u in range(1,513):
    adj[0].append((u,1000+u));adj[u].append((1025,u%7));adj[1025].append((u,2+u%5))
for u in (5,1030):adj[u].append((u+1,0));adj[u+1].append((u,0))
m=sum(map(len,adj));row=[];count=0;dest=[];weights=[]
for edges in adj:
    count+=len(edges);row.append(count)
    for v,w in edges:dest.append(v);weights.append(w)
with fixture.open('wb') as f:
    f.write(struct.pack('<4Q',1,4,n,m));f.write(struct.pack('<'+str(n)+'Q',*row));f.write(struct.pack('<'+str(m)+'I',*dest));
    if m%2:f.write(b'\0'*4)
    f.write(struct.pack('<'+str(m)+'i',*weights))
for source in (0,1030,2050):
    rc,raw=command('export_s'+str(source),[BASE/'export_graph',fixture,source,n//2,OUT/f's{source}.gr',OUT/f's{source}.i32']);assert rc==0
env={k:v for k,v in os.environ.items() if not k.startswith(('MLMQ_','BENCH_','L3_SUPPLEMENT_'))}
env.update(MLMQ_BENCH='1',MLMQ_WORK_BLOCKS='107',BENCH_DELTA='200000',BENCH_QUEUE='L1SLF_L2DQ',BENCH_SOURCE='0',BENCH_WARMUPS='0',BENCH_REPEATS='4',MLMQ_CUT_PERCENT='50',L3_SUPPLEMENT_SOURCES='0,1030,2050,0',L3_SUPPLEMENT_ORACLE_DIR=str(OUT))
# Fixed G+ constructed with source 0; changing source must not rebuild the graph.
rc,raw=command('sequential_reset',[BASE/'dual_diagnostic_build/mlmq','-i',OUT/'s0.gr','-n','2','-d','200000'],env)
rows=parse_bench(raw)
# The dual-GPU diagnostic prints one WIDE_ORACLE line per GPU per query.
reset_ok=(rc==0 and [int(r['source']) for r in rows]==[0,1030,2050,0]
          and all(r['correct']=='1' and r['gpu_count']=='2' for r in rows)
          and raw.count('WIDE_ORACLE ')==2*len(rows)
          and raw.count('CAPACITY_QUERY ')==8)
if not reset_ok:problems.append(dict(test='sequential_reset',rc=rc))
print('RESET_TEST',reset_ok,flush=True)
if not reset_ok:
    save('results.json',dict(problems=problems,reset_pass=False,capacity_queries=0,valid=False))
    raise RuntimeError('Stop after reset failure; preserve evidence before broader capacity runs')
for key in ('L3_SUPPLEMENT_SOURCES','L3_SUPPLEMENT_ORACLE_DIR'):env.pop(key)
env['BENCH_REPEATS']='1'
graphs=json.loads((BASE/'matrix/graphs.json').read_text());records=[]
for graph,d in graphs.items():
    for variant,(build,view,n) in {'M1-original':('single','original',1),'M1-shortcut':('single','augmented',1),'M2-original':('dual','original',2),'M2-shortcut':('dual','augmented',2)}.items():
        env.update(BENCH_SOURCE=str(d['source']),MLMQ_CUT_PERCENT=str(d['cut_percent']),L3_SUPPLEMENT_ORACLE=d['oracle'])
        prefix=OUT/(graph+'_'+variant)
        r=run_one(BASE/(build+'_diagnostic_build/mlmq'),Path(d[view]),'MLMQ',n,prefix,env,d['source'],'L1SLF_L2DQ',0,1,400,200000)
        raw=prefix.with_suffix('.log').read_text()
        if raw.count('CAPACITY_QUERY ')!=n:r.update(valid=False,reason='missing capacity audit')
        records.append(dict(graph=graph,variant=variant,**r));save('capacity_records.json',records)
        if not r['valid']:problems.append(dict(test=graph+'_'+variant,reason=r['reason']))
        print('CAPACITY_TEST',graph,variant,r['valid'],flush=True)
        if not r['valid']:
            save('results.json',dict(problems=problems,reset_pass=reset_ok,capacity_queries=len(records),valid=False))
            raise RuntimeError('Stop after capacity failure; preserve evidence')
save('results.json',dict(problems=problems,reset_pass=reset_ok,capacity_queries=len(records),valid=not problems))
print('SUPPLEMENT_CHECKS_DONE problems='+str(len(problems)),flush=True)
sys.exit(bool(problems))
