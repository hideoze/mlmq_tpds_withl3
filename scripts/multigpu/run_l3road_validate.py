#!/usr/bin/env python3
"""Validate the l3road build (L3-gated chain shortcuts + responsive window).

n=1 must NOT build chain shortcuts (L3 feature only); n=2 builds them.
Correctness is checked against the on-the-fly CPU Dijkstra on the ORIGINAL graph.
"""
import json, os, subprocess, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in
          ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
BINS = {
    'nol3_1': WS / 'tmp/base_today/nol3/mlmq',     # original single-GPU, 512 workers
    'l3road_1': WS / 'tmp/base_today/l3road/mlmq', # L3 build at n=1 (no chain, 320 workers)
    'l3road_2': WS / 'tmp/base_today/l3road/mlmq', # L3 build at n=2 (chain + window)
}


def parse(stdout):
    rows = []
    for line in stdout.splitlines():
        if line.startswith('BENCH '):
            d = {}
            for tok in line.split()[1:]:
                if '=' in tok:
                    d[tok.split('=', 1)[0]] = tok.split('=', 1)[1]
            rows.append(d)
    return rows


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
        rc, stdout = 124, (e.stdout or '')
    out.with_suffix('.log').write_text(stdout)
    rows = parse(stdout)
    formal = [float(r['solve_ms']) for r in rows if r.get('warmup') == '0']
    ok = len(formal) == repeats and all(r.get('correct') == '1' for r in rows if r.get('warmup') == '0')
    chain = [l for l in stdout.splitlines() if l.startswith('L3_CHAIN_SETUP')]
    return dict(rc=rc, solve_ms=formal, correct=ok, median_ms=median(formal) if formal else None,
                chain_built=len(chain), chain_line=chain[0] if chain else '')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--graphs', nargs='+', default=['NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA'])
    ap.add_argument('--variants', nargs='+', default=['nol3_1', 'l3road_1', 'l3road_2'])
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []
    for turn in range(2):
        order = a.variants if turn == 0 else list(reversed(a.variants))
        for g in a.graphs:
            src = LAYOUT[g]['source_new_id']
            for v in order:
                n = 1 if v.endswith('_1') else 2
                pre = out / f'r{turn}_{g}_{v}'
                r = run_one(BINS[v], GRAPHS[g], n, src, pre, repeats=a.repeats)
                rec = dict(round=turn, graph=g, variant=v, n_gpu=n, **r)
                records.append(rec)
                (out / 'records.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(round=turn, graph=g, variant=v, rc=r['rc'],
                                      correct=r['correct'], chain=r['chain_built'],
                                      med=round(r['median_ms'], 3) if r['median_ms'] else None)),
                      flush=True)
    # checks: n=1 must not build chain; n=2 must build chain; all correct
    errs = []
    for rec in records:
        if rec['rc'] != 0 or not rec['correct']:
            errs.append(f"{rec['graph']}/{rec['variant']} rc={rec['rc']} correct={rec['correct']}")
        want = 1 if rec['n_gpu'] == 2 else 0
        if rec['chain_built'] != want:
            errs.append(f"{rec['graph']}/{rec['variant']} chain_built={rec['chain_built']} want={want}")
    # summary
    summary = []
    for g in a.graphs:
        row = dict(graph=g)
        for v in a.variants:
            ms = [r['median_ms'] for r in records if r['graph'] == g and r['variant'] == v and r['rc'] == 0]
            if ms:
                row[v] = round(median(ms), 4)
        if 'nol3_1' in row and 'l3road_2' in row and row.get('l3road_2'):
            row['speedup'] = round(row['nol3_1'] / row['l3road_2'], 4)
        summary.append(row)
    import math
    sp = [s['speedup'] for s in summary if s.get('speedup')]
    geo = math.exp(sum(math.log(x) for x in sp) / len(sp)) if sp else None
    (out / 'summary.json').write_text(json.dumps(dict(rows=summary, geomean=geo,
                                                      validation_errors=errs), indent=2))
    print('VALIDATION_DONE', flush=True)
    print(json.dumps(dict(rows=summary, geomean=geo, validation_errors=errs[:10]), indent=2), flush=True)


if __name__ == '__main__':
    main()
