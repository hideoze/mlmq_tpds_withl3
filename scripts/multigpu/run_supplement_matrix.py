#!/usr/bin/env python3
"""Four-cell same-CSR comparison, genuine two GPUs; run inside Slurm only."""
import csv,json,math,os,shutil,socket,subprocess,sys
from pathlib import Path
from statistics import median
from run_usa_road_matrix import run_one,sha256
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/l3_supplement_20260921')
OUT=BASE/'matrix'
assert os.environ.get('SLURM_JOB_ID')
OUT.mkdir(exist_ok=False)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
for label,cmd in [('gpu',['nvidia-smi']),('topology',['nvidia-smi','topo','-m']),('apps',['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])]:
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/(label+'.log')).write_text(p.stdout);assert p.returncode==0
    if label=='apps':assert not p.stdout.strip(),'GPU already occupied'
graphs={};cuts={'NY':60,'COL':55,'W':40,'USA':60}
for name in ('NY','BAY','COL','FLA','CAL','E','W','USA'):
    directory=BASE/'graphs'/name;directory.mkdir(parents=True,exist_ok=False)
    original=(ROOT/'tmp/landmark206_matrix'/name/'landmark.gr').resolve(strict=True)
    source=json.loads(original.with_name('layout.json').read_text())['source_new_id']
    import struct
    with original.open('rb') as f:n=struct.unpack('<4Q',f.read(32))[2]
    cut_percent=cuts.get(name,50);cut=n*cut_percent//100
    augmented=directory/'augmented.gr';oracle=directory/'oracle.i32'
    cmd=[str(BASE/'export_graph'),str(original),str(source),str(cut),str(augmented),str(oracle)]
    with (directory/'export.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,(name,p.returncode)
    graphs[name]=dict(source=source,cut=cut,cut_percent=cut_percent,vertices=n,
        original=str(original),augmented=str(augmented),oracle=str(oracle),
        hashes={x:sha256(p) for x,p in [('original',original),('augmented',augmented),('oracle',oracle)]})
    save('graphs.json',graphs);print('GRAPH_READY',name,flush=True)
variants={'M1-original':('single','original',1),'M1-shortcut':('single','augmented',1),
          'M2-original':('dual','original',2),'M2-shortcut':('dual','augmented',2)}
save('manifest.json',dict(job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),graphs=graphs,
    binaries={k:dict(path=str(BASE/(k+'_build/mlmq')),sha256=sha256(BASE/(k+'_build/mlmq'))) for k in ('single','dual')},
    variants=variants,warmups=1,formal=5,rounds=2,workers=512,blocks=107,delta=200000,
    single='Independent no-L3 worker from frozen 213, shared core synchronized to current dual source',
    adapters='Host oracle, capacity disclosure, INT_MAX byte budget; host chain disabled; no GPU test hooks'))
shutil.copy2(__file__,OUT/'runner.py')
env={k:v for k,v in os.environ.items() if not k.startswith(('MLMQ_','BENCH_','L3_SUPPLEMENT_'))}
env.update(MLMQ_BENCH='1',MLMQ_WORK_BLOCKS='107',BENCH_DELTA='200000',BENCH_QUEUE='L1SLF_L2DQ',BENCH_REPEATS='5',BENCH_WARMUPS='1')
records=[]
for round_id in range(2):
    for graph,data in graphs.items():
        for variant in (list(variants) if round_id==0 else list(reversed(variants))):
            build,view,n=variants[variant];env.update(BENCH_SOURCE=str(data['source']),MLMQ_CUT_PERCENT=str(data['cut_percent']),L3_SUPPLEMENT_ORACLE=data['oracle'])
            prefix=OUT/f'r{round_id}_{graph}_{variant}'
            r=run_one(BASE/(build+'_build/mlmq'),Path(data[view]),'MLMQ',n,prefix,env,data['source'],'L1SLF_L2DQ',1,5,400,200000)
            raw=prefix.with_suffix('.log').read_text()
            # Single main has one additional final check; require every timed query's independent check.
            if raw.count('WIDE_ORACLE ') < 6 or 'L3_CHAIN_SETUP' in raw:
                r.update(valid=False,reason='missing wide oracle or unexpected augmentation')
            if n==2 and raw.count('L2_FINAL ')!=12:r.update(valid=False,reason='missing L2 conservation')
            records.append(dict(round=round_id,graph=graph,variant=variant,**r));save('records.json',records)
            print(json.dumps(dict(round=round_id,graph=graph,variant=variant,rc=r['rc'],valid=r['valid'],reason=r['reason'],solve=[s['solve_ms'] for s in r['samples']])),flush=True)
            if not r['valid']:raise RuntimeError('Retained failed run: '+str(prefix))
rows=[]
for graph in graphs:
    row={'graph':graph}
    for variant in variants:
        values=[float(s['solve_ms']) for r in records if r['graph']==graph and r['variant']==variant for s in r['samples'] if s['warmup']=='0']
        assert len(values)==10 and all(math.isfinite(v) and v>0 for v in values)
        row[variant]=median(values)
        for rnd in range(2):row[f'{variant}_r{rnd}']=median(float(s['solve_ms']) for r in records if r['graph']==graph and r['variant']==variant and r['round']==rnd for s in r['samples'] if s['warmup']=='0')
    row['same_original']=row['M1-original']/row['M2-original']
    row['same_shortcut']=row['M1-shortcut']/row['M2-shortcut']
    row['complete']=row['M1-original']/row['M2-shortcut']
    row['single_shortcut_gain']=row['M1-original']/row['M1-shortcut']
    rows.append(row)
save('summary.json',rows)
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('complete.json',dict(processes=len(records),queries=384,formal=320,valid=True,geomeans={k:math.exp(sum(math.log(r[k]) for r in rows)/8) for k in ('same_original','same_shortcut','complete','single_shortcut_gain')}))
print('MATRIX_PASS processes=64 queries=384 formal=320',flush=True)
