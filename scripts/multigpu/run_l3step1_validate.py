#!/usr/bin/env python3
"""Step-1 validation: per-graph runtime cut (MLMQ_CUT_PERCENT) vs no-L3 baseline.

Eight graphs, 2 reversed rounds, 1 warmup + 3 formal per process. Slow samples
kept. Cut choices from the production cut sweep (round 2):
  NY 60, BAY 50(vertex), COL 55, FLA 50, CAL 50, E 50, W 40, USA 60.
"""
import json, math, os, subprocess, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in
          ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
CUT = {'NY': 60, 'COL': 55, 'W': 40, 'USA': 60}  # others: vertex cut (no env)
BINS = {
    'nol3_1': WS / 'tmp/base_today/nol3/mlmq',
    'l3roadcut_2': WS / 'tmp/base_today/prod_nogate512/mlmq',
}


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
    rows = []
    for line in stdout.splitlines():
        if line.startswith('BENCH ') and 'warmup=0' in line:
            d = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
            rows.append(d)
    formal = [float(r['solve_ms']) for r in rows]
    ok = len(formal) == repeats and all(r.get('correct') == '1' for r in rows)
    return dict(rc=rc, solve_ms=formal, correct=ok, median_ms=median(formal) if formal else None)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []
    for turn in range(2):
        gs = list(GRAPHS) if turn == 0 else list(reversed(list(GRAPHS)))
        for g in gs:
            src = LAYOUT[g]['source_new_id']
            cut_env = {'MLMQ_CUT_PERCENT': str(CUT[g])} if g in CUT else {}
            # baseline first on turn 0, last on turn 1
            order = ['nol3_1', 'l3roadcut_2'] if turn == 0 else ['l3roadcut_2', 'nol3_1']
            for v in order:
                n = 1 if v.endswith('_1') else 2
                pre = out / f'r{turn}_{g}_{v}'
                r = run_one(BINS[v], GRAPHS[g], n, src, pre,
                            cut_env if n == 2 else {}, repeats=a.repeats)
                records.append(dict(round=turn, graph=g, variant=v, n_gpu=n, **r))
                (out / 'records.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(round=turn, graph=g, variant=v, rc=r['rc'],
                                      correct=r['correct'],
                                      med=round(r['median_ms'], 3) if r['median_ms'] else None)), flush=True)
    errs = [f'{r["graph"]}/{r["variant"]} rc={r["rc"]} correct={r["correct"]}'
            for r in records if r['rc'] != 0 or not r['correct']]
    summary = []
    for g in GRAPHS:
        row = dict(graph=g)
        for v in BINS:
            ms = [r['median_ms'] for r in records if r['graph'] == g and r['variant'] == v and r['rc'] == 0]
            if ms:
                row[v] = round(median(ms), 4)
        if row.get('nol3_1') and row.get('l3roadcut_2'):
            row['speedup'] = round(row['nol3_1'] / row['l3roadcut_2'], 4)
        summary.append(row)
    sp = [s['speedup'] for s in summary if s.get('speedup')]
    geo = math.exp(sum(math.log(x) for x in sp) / len(sp)) if sp else None
    (out / 'summary.json').write_text(json.dumps(dict(rows=summary, geomean=geo, errs=errs), indent=2))
    print('STEP1_DONE', flush=True)
    print(json.dumps(dict(geomean=geo, errs=errs[:5]), indent=2), flush=True)
    for s in summary:
        print(f" {s['graph']:4} nol3={s.get('nol3_1')} l3cut={s.get('l3roadcut_2')} sp={s.get('speedup')}")


if __name__ == '__main__':
    main()
