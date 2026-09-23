#!/usr/bin/env python3
"""Re-read raw outputs, calculate dispersion, and archive source provenance."""
import csv,hashlib,json,math,re,statistics,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/l3_supplement_20260921_v2')
OUT=(ROOT/'knowledgebase/l3_supplementary_execution_20260921' if BASE==ROOT/'tmp/l3_supplement_20260921_v2' else BASE/'audit');OUT.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fields(raw,tag):return [dict(t.split('=',1) for t in line.split()[1:] if '=' in t) for line in raw.splitlines() if line.startswith(tag+' ')]
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
records=json.loads((BASE/'matrix/records.json').read_text());graphs=json.loads((BASE/'matrix/graphs.json').read_text())
assert len(records)==64
seen=set();rows=[];formal=0;wide=0;l2=0
for r in records:
    identity=(r['round'],r['graph'],r['variant']);assert identity not in seen;seen.add(identity)
    raw=(BASE/'matrix'/f'r{r["round"]}_{r["graph"]}_{r["variant"]}.log').read_text()
    samples=fields(raw,'BENCH');assert samples==r['samples'] and r['valid'] and r['rc']==0 and 'RUN_RC=0' in raw
    n=1 if r['variant'].startswith('M1') else 2
    assert len(samples)==6 and len(fields(raw,'WIDE_ORACLE'))>=6
    for i,s in enumerate(samples):
        assert s['gpu_count']==str(n) and s['correct']=='1' and s['source']==str(graphs[r['graph']]['source'])
        assert s['repeat']==str(i) and s['warmup']==str(int(i==0)) and s['queue']=='L1SLF_L2DQ'
        assert math.isfinite(float(s['solve_ms'])) and float(s['solve_ms'])>0
    formal+=sum(s['warmup']=='0' for s in samples);wide+=len(fields(raw,'WIDE_ORACLE'));l2+=len(fields(raw,'L2_FINAL'))
    # Each GPU logs capacity for two queue instances: expect two records per
    # single-GPU process and four for a dual-GPU process.
    capacities=fields(raw,'L2_CAPACITY');assert len(capacities)==2*n
    for c in capacities:
        assert int(c['record_bytes'])==8 and int(c['buckets'])==16 and int(c['per_bucket'])==16776704
        assert int(c['per_bucket'])*int(c['buckets'])<=int(c['allocated_records'])
    if n==2:
        parts=re.findall(r'GPU(\d) partition: \[(\d+), (\d+)\)',raw)
        assert sorted(parts)==[('0','0',str(graphs[r['graph']]['cut'])),('1',str(graphs[r['graph']]['cut']),str(graphs[r['graph']]['vertices']))]
        finals=fields(raw,'L2_FINAL');assert len(finals)==12
        assert all(x['reads']==x['writes']==x['completed'] for x in finals)
for g in graphs:
    for variant in ('M1-original','M1-shortcut','M2-original','M2-shortcut'):
        xs=[float(s['solve_ms']) for r in records if r['graph']==g and r['variant']==variant for s in r['samples'] if s['warmup']=='0']
        assert len(xs)==10;med=statistics.median(xs);q1,_,q3=statistics.quantiles(xs,n=4,method='inclusive')
        rows.append(dict(graph=g,variant=variant,n=10,median_ms=med,q25_ms=q1,q75_ms=q3,iqr_ms=q3-q1,mad_ms=statistics.median(abs(x-med) for x in xs),min_ms=min(xs),max_ms=max(xs)))
with (OUT/'timing_dispersion.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for g,d in graphs.items():
    for key in ('original','augmented','oracle'):assert sha(Path(d[key]))==d['hashes'][key]
single=BASE/'single_source/core';dual=BASE/'dual_source/core';core_files=[]
for p in sorted(single.rglob('*')):
    if p.is_file():
        q=dual/p.relative_to(single);assert p.read_bytes()==q.read_bytes(),p
        core_files.append(dict(path=str(p.relative_to(single)),sha256=sha(p)))
save('identical_core_files.json',core_files)
source_hashes={}
for kind in ('single','dual','single_diagnostic','dual_diagnostic'):
    build=BASE/(kind+'_build');src=BASE/(kind+'_source')
    with tarfile.open(build/'source.tgz') as t:
        for m in t.getmembers():
            if m.isfile():assert t.extractfile(m).read()==(src/m.name).read_bytes(),(kind,m.name)
    source_hashes[kind]={name:sha(build/name) for name in ('source.tgz','mlmq','command.json')}
checks=json.loads((BASE/'checks/results.json').read_text());assert checks['valid'] and not checks['problems']
cap_records=json.loads((BASE/'checks/capacity_records.json').read_text());assert len(cap_records)==32
caprows=[]
for r in cap_records:
    assert r['valid'] and r['rc']==0
    raw=(BASE/'checks'/(r['graph']+'_'+r['variant']+'.log')).read_text();n=1 if r['variant'].startswith('M1') else 2
    caps=fields(raw,'CAPACITY_BUCKET');assert len(caps)==16*n
    for c in caps:
        assert c['capacity']=='16776704' and c['no_wrap']=='1' and c['reads']==c['writes'] and 0<=int(c['writes'])<=16776704
        caprows.append(dict(graph=r['graph'],variant=r['variant'],**c))
with (OUT/'capacity_by_bucket.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(caprows[0]));w.writeheader();w.writerows(caprows)
reset=(BASE/'checks/sequential_reset.log').read_text();s=fields(reset,'BENCH')
assert [int(x['source']) for x in s]==[0,1030,2050,0] and all(x['gpu_count']=='2' and x['correct']=='1' for x in s)
assert len(fields(reset,'CAPACITY_QUERY'))==8 and len(fields(reset,'QUERY_REUSE'))==8
primitive=(BASE/'checks/primitives.log').read_text();trans=fields(primitive,'TRANSPORT');assert len(trans)==2
assert all(x['delayed_ack_fired']=='1' and x['pending']=='0' and int(x['backpressure_retries'])>0 and x['precommit_checks']=='256' for x in trans)
assert 'L3_SUPPLEMENT_PRIMITIVES PASS' in primitive and len(fields(primitive,'CANDIDATE'))==2
largest=max(caprows,key=lambda x:int(x['writes']))
save('audit.json',dict(base=str(BASE),formal_queries=formal,total_queries=384,processes=64,wide_oracle_lines=wide,dual_final_conservation_lines=l2,
    identical_core_file_count=len(core_files),source_provenance=source_hashes,checks=checks,max_bucket=largest,
    max_capacity_fraction=int(largest['writes'])/16776704,primitive_transport=trans,
    note='Performance and capacity-guard runs are different executions. No weak-memory proof is implied.'))
(OUT/'same_graph_summary.csv').write_bytes((BASE/'matrix/summary.csv').read_bytes())
(OUT/'same_graph_summary.json').write_bytes((BASE/'matrix/summary.json').read_bytes())
save('raw_evidence_sha256.json',{str(p.relative_to(ROOT)):sha(p) for folder in (BASE/'matrix',BASE/'checks') for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix not in ('.gr','.i32')})
print('AUDIT_PASS',json.dumps(dict(formal=formal,core_files=len(core_files),max_bucket=largest)))
