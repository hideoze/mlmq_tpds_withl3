#!/usr/bin/env python3
"""Round 2: clean cut-sweep times (production build, no diag inflation) +
single-GPU starvation baseline (fixed diag1: WORK_DIAG only, 512 threads)."""
import json, os, re, subprocess, sys, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in
          ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
BINS = {
    'diag1_1': WS / 'tmp/base_today/diag1/mlmq',
    'l3road_2': WS / 'tmp/base_today/l3road/mlmq',
    'pcut40_2': WS / 'tmp/base_today/prod_cut40/mlmq',
    'pcut55_2': WS / 'tmp/base_today/prod_cut55/mlmq',
    'pcut60_2': WS / 'tmp/base_today/prod_cut60/mlmq',
}
WORK_RE = re.compile(r'L3_WORK gpu=(\d+).*?expanded=(\d+) edges=(\d+) calls=(\d+) '
                     r'empty_reads=(\d+) idle_iters=(\d+) busy_iters=(\d+)')


def run_one(binary, graph, n_gpu, source, out, repeats=3, warmups=1, timeout=400):
    env = dict(os.environ)
    env.update(MLMQ_BENCH='1', MLMQ_WORK_BLOCKS='107', BENCH_DELTA='200000',
               BENCH_SOURCE=str(source), BENCH_WARMUPS=str(warmups),
               BENCH_REPEATS=str(repeats), BENCH_QUEUE='L1SLF_L2DQ')
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
    works = WORK_RE.findall(stdout)
    stat = None
    if works:
        n = n_gpu
        last = works[-n:]
        stat = dict(edges=sum(int(x[2]) for x in last),
                    empty_reads=sum(int(x[4]) for x in last),
                    idle_iters=sum(int(x[5]) for x in last),
                    busy_iters=sum(int(x[6]) for x in last),
                    per_gpu=[dict(gpu=x[0], edges=int(x[2]), empty=int(x[4]),
                                  idle=int(x[5]), busy=int(x[6])) for x in last])
    return dict(rc=rc, solve_ms=formal, correct=ok,
                median_ms=median(formal) if formal else None, work=stat)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []

    def go(g, v):
        n = 1 if v.endswith('_1') else 2
        pre = out / f'{g}_{v}'
        r = run_one(BINS[v], GRAPHS[g], n, LAYOUT[g]['source_new_id'], pre, repeats=a.repeats)
        records.append(dict(graph=g, variant=v, **r))
        (out / 'records.json').write_text(json.dumps(records, indent=2))
        w = r.get('work') or {}
        print(json.dumps(dict(graph=g, variant=v, rc=r['rc'], correct=r['correct'],
                              med=round(r['median_ms'], 3) if r['median_ms'] else None,
                              edges=w.get('edges'), empty=w.get('empty_reads'),
                              idle=w.get('idle_iters'), busy=w.get('busy_iters'))), flush=True)
        return r

    # 1) single-GPU starvation baseline (all 8) + dual reference for ratio
    for turn in range(2):
        gs = ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA') if turn == 0 else \
             reversed(('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA'))
        for g in gs:
            go(g, 'diag1_1')

    # 2) clean cut sweep (production builds): all graphs for completeness
    for turn in range(2):
        gs = ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA') if turn == 0 else \
             reversed(('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA'))
        for g in gs:
            go(g, 'l3road_2')
            go(g, 'pcut40_2')
            go(g, 'pcut55_2')
            go(g, 'pcut60_2')

    (out / 'records.json').write_text(json.dumps(records, indent=2))
    print('ROUND2_DONE', flush=True)


if __name__ == '__main__':
    main()
