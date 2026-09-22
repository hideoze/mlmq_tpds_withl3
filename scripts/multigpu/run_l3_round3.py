#!/usr/bin/env python3
"""Round 3: L2/warp-shape parameter sweep on dual-GPU (production builds).

Hypotheses under test (from starvation diagnostics: 52-88% empty reads):
  H1 warp:items ratio  - fewer work warps per GPU (MLMQ_WORK_BLOCKS sweep, env)
  H2 claim granularity - larger l2_batch_size (16/32) -> fewer failed claims
  H3 bucket width      - BUCKET_MAX=2 -> denser concurrent buckets
  H4 partition cut     - per-graph cut (already measured separately)

Runs on NY, FLA, W, USA (small / mid / imbalanced-large / large), n=2,
production chain+win25 build. All samples checked for correctness.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in
          ('NY', 'FLA', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
ROAD = WS / 'tmp/base_today/l3road/mlmq'
B4 = WS / 'tmp/base_today/prod_b4/mlmq'
BM2 = WS / 'tmp/base_today/prod_bm2/mlmq'
BM8 = WS / 'tmp/base_today/prod_bm8/mlmq'
BM16 = WS / 'tmp/base_today/prod_bm16/mlmq'
BO = WS / 'tmp/base_today/prod_bo/mlmq'
BO_D = WS / 'tmp/base_today/diag_bo/mlmq'


def run_one(binary, graph, n_gpu, source, out, env_extra, repeats=3, warmups=1, timeout=400):
    env = dict(os.environ)
    env.update(MLMQ_BENCH='1', MLMQ_WORK_BLOCKS='107', BENCH_DELTA='200000',
               BENCH_SOURCE=str(source), BENCH_WARMUPS=str(warmups),
               BENCH_REPEATS=str(repeats), BENCH_QUEUE='L1SLF_L2DQ')
    env.update(env_extra)
    cmd = [str(binary), '-i', str(graph), '-n', str(n_gpu), '-d', '200000']
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc, stdout = r.returncode, r.stdout
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or b''
        if isinstance(partial, bytes):
            partial = partial.decode('utf-8', 'replace')
        rc, stdout = 124, partial
    out.with_suffix('.log').write_text(stdout)
    rows = [l for l in stdout.splitlines() if l.startswith('BENCH ')]
    parse = []
    for line in rows:
        d = {}
        for tok in line.split()[1:]:
            if '=' in tok:
                d[tok.split('=', 1)[0]] = tok.split('=', 1)[1]
        parse.append(d)
    formal = [float(r['solve_ms']) for r in parse if r.get('warmup') == '0']
    ok = len(formal) == repeats and all(r.get('correct') == '1' for r in parse if r.get('warmup') == '0')
    return dict(rc=rc, solve_ms=formal, correct=ok,
                median_ms=median(formal) if formal else None)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []

    CONFIGS = [
        ('base',    ROAD, {}),
        ('wb96',    ROAD, {'MLMQ_WORK_BLOCKS': '96'}),
        ('wb80',    ROAD, {'MLMQ_WORK_BLOCKS': '80'}),
        ('wb64',    ROAD, {'MLMQ_WORK_BLOCKS': '64'}),
        ('b4',      B4,   {}),
        ('bm2',     BM2,  {}),
        ('bm8',     BM8,  {}),
        ('bm16',    BM16, {}),
        ('bo',      BO,   {}),
    ]

    def go(g, name, binary, env_extra):
        pre = out / f'{g}_{name}'
        r = run_one(binary, GRAPHS[g], 2, LAYOUT[g]['source_new_id'], pre,
                    env_extra, repeats=a.repeats)
        records.append(dict(graph=g, variant=name, **r))
        (out / 'records.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(dict(graph=g, variant=name, rc=r['rc'],
                              correct=r['correct'],
                              med=round(r['median_ms'], 3) if r['median_ms'] else None)), flush=True)

    for turn in range(2):
        gs = ('NY', 'FLA', 'W', 'USA') if turn == 0 else ('USA', 'W', 'FLA', 'NY')
        for g in gs:
            for name, binary, env_extra in CONFIGS:
                go(g, name, binary, env_extra)

    # Starvation check with backoff on (diag build; counters only, times inflated)
    import re
    WORK_RE = re.compile(r'L3_WORK gpu=(\d+).*?edges=(\d+) calls=(\d+) '
                         r'empty_reads=(\d+) idle_iters=(\d+) busy_iters=(\d+)')
    for g in ('NY', 'W'):
        pre = out / f'{g}_bo_diag'
        r = run_one(BO_D, GRAPHS[g], 2, LAYOUT[g]['source_new_id'], pre, {}, repeats=1, warmups=0)
        logs = (pre.with_suffix('.log')).read_text()
        works = WORK_RE.findall(logs)
        stats = [dict(gpu=w[0], edges=w[1], empty=w[3], idle=w[4], busy=w[5]) for w in works[-2:]]
        records.append(dict(graph=g, variant='bo_diag', **r))
        (out / 'records.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(dict(graph=g, variant='bo_diag', rc=r['rc'],
                              correct=r['correct'], work=stats)), flush=True)

    (out / 'records.json').write_text(json.dumps(records, indent=2))
    print('ROUND3_DONE', flush=True)


if __name__ == '__main__':
    main()
