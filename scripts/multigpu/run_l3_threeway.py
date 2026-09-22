#!/usr/bin/env python3
"""Same-batch three-way measurement: ADDS1 vs no-L3 MLMQ1 vs L3 MLMQ2.

All three configurations run interleaved on the same allocation, same graphs,
same sources, 1 warmup + 3 formal per process, 2 reversed rounds. ADDS uses
the frozen official binary (tmp/adds_official_v4/adds). Slow samples kept.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in
          ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
CUT = {'NY': 60, 'COL': 55, 'W': 40, 'USA': 60}
BINS = {
    'adds1': WS / 'tmp/adds_official_v4/adds',
    'nol3_1': WS / 'tmp/base_today/nol3/mlmq',
    'l3cut_2': WS / 'tmp/base_today/prod_nogate512/mlmq',
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


def run_one(variant, graph, source, out, repeats=3, warmups=1, timeout=400):
    env = dict(os.environ)
    env.update(MLMQ_BENCH='1', MLMQ_WORK_BLOCKS='107', BENCH_DELTA='200000',
               BENCH_SOURCE=str(source), BENCH_WARMUPS=str(warmups),
               BENCH_REPEATS=str(repeats), BENCH_QUEUE='L1SLF_L2DQ')
    if variant == 'l3cut_2' and graph.name:
        gname = graph.parent.name
        if gname in CUT:
            env['MLMQ_CUT_PERCENT'] = str(CUT[gname])
    if variant == 'adds1':
        cmd = [str(BINS[variant]), str(graph)]
    else:
        n = 1 if variant.endswith('_1') else 2
        cmd = [str(BINS[variant]), '-i', str(graph), '-n', str(n), '-d', '200000']
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
    rows = parse(stdout)
    formal = [float(r['solve_ms']) for r in rows if r.get('warmup') == '0']
    expect_alg = 'ADDS' if variant == 'adds1' else 'MLMQ'
    ok = (len(formal) == repeats
          and all(r.get('correct') == '1' and r.get('algorithm') == expect_alg
                  for r in rows if r.get('warmup') == '0'))
    return dict(rc=rc, solve_ms=formal, correct=ok, median_ms=median(formal) if formal else None)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []
    for turn in range(2):
        gs = list(GRAPHS) if turn == 0 else list(reversed(list(GRAPHS)))
        for g in gs:
            src = LAYOUT[g]['source_new_id']
            order = ['adds1', 'nol3_1', 'l3cut_2'] if turn == 0 else ['l3cut_2', 'nol3_1', 'adds1']
            for v in order:
                pre = out / f'r{turn}_{g}_{v}'
                r = run_one(v, GRAPHS[g], src, pre, repeats=a.repeats)
                records.append(dict(round=turn, graph=g, variant=v, **r))
                (out / 'records.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(round=turn, graph=g, variant=v, rc=r['rc'],
                                      correct=r['correct'],
                                      med=round(r['median_ms'], 3) if r['median_ms'] else None)), flush=True)
    errs = [f'{r["graph"]}/{r["variant"]} rc={r["rc"]} correct={r["correct"]}'
            for r in records if r['rc'] != 0 or not r['correct']]
    (out / 'records.json').write_text(json.dumps(records, indent=2))
    print('THREEWAY_DONE errors=%d' % len(errs), flush=True)
    for e in errs[:10]:
        print('ERR', e, flush=True)
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
