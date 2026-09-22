#!/usr/bin/env python3
"""Phase-2 diagnostics: starvation attribution + cut-point sweep.

Runs (all on landmark graphs, standard sources):
  diag1_1   : single-GPU no-L3 (512 workers) + starvation counters
  diag_ch_2 : dual L3 (chain+win25, vertex cut) + starvation counters
  cut40/55/60_2 : dual L3 with shifted partition cut (PARTITION_CUT_PERCENT)

Prints a compact per-graph table of solve ms, edge counts, and the
empty/idle/busy read-iteration split.
"""
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
    'diag_ch_2': WS / 'tmp/base_today/diag_ch/mlmq',
    'cut40_2': WS / 'tmp/base_today/diag_cut40/mlmq',
    'cut55_2': WS / 'tmp/base_today/diag_cut55/mlmq',
    'cut60_2': WS / 'tmp/base_today/diag_cut60/mlmq',
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
    # work metrics from the LAST formal sample (representative)
    works = WORK_RE.findall(stdout)
    stat = None
    if works:
        # aggregate across gpus of the last sample group
        n = n_gpu
        last = works[-n:]
        stat = dict(
            expanded=sum(int(x[1]) for x in last),
            edges=sum(int(x[2]) for x in last),
            calls=sum(int(x[3]) for x in last),
            empty_reads=sum(int(x[4]) for x in last),
            idle_iters=sum(int(x[5]) for x in last),
            busy_iters=sum(int(x[6]) for x in last),
            per_gpu=[dict(gpu=x[0], expanded=int(x[1]), edges=int(x[2]),
                          empty=int(x[4]), idle=int(x[5]), busy=int(x[6])) for x in last])
    return dict(rc=rc, solve_ms=formal, correct=ok,
                median_ms=median(formal) if formal else None, work=stat)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--starve-graphs', nargs='+',
                    default=['NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA'])
    ap.add_argument('--cut-graphs', nargs='+', default=['BAY', 'E', 'CAL', 'W', 'USA'])
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-smi', 'topo', '-m'], check=False)
    records = []

    def go(g, v, tag):
        n = 1 if v.endswith('_1') else 2
        pre = out / f'{g}_{v}'
        r = run_one(BINS[v], GRAPHS[g], n, LAYOUT[g]['source_new_id'], pre, repeats=a.repeats)
        records.append(dict(graph=g, variant=v, tag=tag, **r))
        (out / 'records.json').write_text(json.dumps(records, indent=2))
        w = r.get('work') or {}
        print(json.dumps(dict(graph=g, variant=v, rc=r['rc'], correct=r['correct'],
                              med=round(r['median_ms'], 3) if r['median_ms'] else None,
                              edges=w.get('edges'), empty=w.get('empty_reads'),
                              idle=w.get('idle_iters'), busy=w.get('busy_iters'),
                              per_gpu=[(x['gpu'], x['edges'], x['empty'], x['idle']) for x in w.get('per_gpu', [])])),
              flush=True)

    # Part 1: starvation, single vs dual, all graphs
    for turn in range(2):
        for g in (a.starve_graphs if turn == 0 else reversed(a.starve_graphs)):
            go(g, 'diag1_1', 'starve')
            go(g, 'diag_ch_2', 'starve')

    # Part 2: cut sweep on the imbalanced-suspect graphs
    for turn in range(2):
        for g in (a.cut_graphs if turn == 0 else reversed(a.cut_graphs)):
            for v in ('cut40_2', 'cut55_2', 'cut60_2'):
                go(g, v, 'cut')

    (out / 'records.json').write_text(json.dumps(records, indent=2))
    print('DIAG_DONE', flush=True)


if __name__ == '__main__':
    main()
