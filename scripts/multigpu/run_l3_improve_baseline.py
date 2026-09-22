#!/usr/bin/env python3
"""Baseline + timing diagnostics for the L3 improvement task.

Runs on one Slurm allocation (2x A100). Measures, on the landmark USA-road graphs:
  nol3_1 : single-GPU original MLMQ (no L3)          -> the comparison baseline
  prod_2 : current production L3 dual-GPU
  chain_2: 224-style chain-shortcut + idle-probe dual-GPU
  time_2 : timing/work diagnostic dual-GPU build

All slow samples are kept; medians use formal (non-warmup) samples only.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')

GRAPHS = {
    'NY':   WS / 'tmp/landmark206_matrix/NY/landmark.gr',
    'W':    WS / 'tmp/landmark206_matrix/W/landmark.gr',
    'USA':  WS / 'tmp/landmark206_matrix/USA/landmark.gr',
}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}

BINS = {
    'nol3_1': WS / 'tmp/base_today/nol3/mlmq',
    'prod_2': WS / 'tmp/base_today/prod/mlmq',
    'chain_2': WS / 'tmp/base_today/chain224/mlmq',
    'time_2': WS / 'tmp/base_today/timing/mlmq',
}


def parse_bench(stdout):
    rows = []
    for line in stdout.splitlines():
        if line.startswith('BENCH '):
            row = {}
            for tok in line.split()[1:]:
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    row[k] = v
            rows.append(row)
    return rows


def run_one(binary, graph, n_gpu, source, out, repeats=3, warmups=1, timeout=300):
    env = dict(os.environ)
    env.update(MLMQ_BENCH='1', MLMQ_WORK_BLOCKS='107', BENCH_DELTA='200000',
               BENCH_SOURCE=str(source), BENCH_WARMUPS=str(warmups),
               BENCH_REPEATS=str(repeats), BENCH_QUEUE='L1SLF_L2DQ')
    cmd = [str(binary), '-i', str(graph), '-n', str(n_gpu), '-d', '200000']
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc, stdout = r.returncode, r.stdout
    except subprocess.TimeoutExpired as e:
        rc, stdout = 124, (e.stdout or '')
    dt = time.monotonic() - t0
    (out.with_suffix('.log')).write_text(stdout)
    rows = parse_bench(stdout)
    formal = [float(r['solve_ms']) for r in rows if r.get('warmup') == '0']
    correct = [r.get('correct') == '1' for r in rows if r.get('warmup') == '0']
    ok = len(formal) == repeats and all(correct)
    return dict(rc=rc, solve_ms=formal, correct=ok, wall=dt,
                median_ms=median(formal) if formal else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--graphs', nargs='+', default=['NY', 'W', 'USA'])
    ap.add_argument('--variants', nargs='+', default=['nol3_1', 'prod_2', 'chain_2', 'time_2'])
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(['nvidia-smi'], check=False)
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
                rec = dict(round=turn, graph=g, variant=v, n_gpu=n, source=src, **r)
                records.append(rec)
                (out / 'records.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(round=turn, graph=g, variant=v, rc=r['rc'],
                                      correct=r['correct'],
                                      solve=[round(x, 3) for x in r['solve_ms']])),
                      flush=True)
    # summary
    summary = []
    for g in a.graphs:
        row = dict(graph=g)
        for v in a.variants:
            xs = [float(s['solve_ms']) for r in records
                  if r['graph'] == g and r['variant'] == v and r['rc'] == 0
                  for s in [r] if s.get('median_ms') is not None]
            ms = [r['median_ms'] for r in records
                  if r['graph'] == g and r['variant'] == v and r['rc'] == 0]
            if ms:
                row[v] = dict(median_ms=round(median(ms), 4), samples=ms)
        if 'nol3_1' in row:
            for v in a.variants:
                if v != 'nol3_1' and v in row:
                    row[f'speedup_{v}'] = round(row['nol3_1']['median_ms'] / row[v]['median_ms'], 4)
        summary.append(row)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('BASELINE_SUMMARY_DONE', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
