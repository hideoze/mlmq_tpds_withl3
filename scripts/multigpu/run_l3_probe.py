#!/usr/bin/env python3
"""Single-GPU vs dual-GPU per-edge throughput probe + L3 window tuning."""
import json, os, subprocess, time
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
GRAPHS = {g: WS / f'tmp/landmark206_matrix/{g}/landmark.gr' for g in ('NY', 'W', 'USA')}
LAYOUT = {g: json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())
          for g in GRAPHS}
BINS = {
    't512_1': WS / 'tmp/base_today/timing512/mlmq',
    't512_2': WS / 'tmp/base_today/timing512/mlmq',
    'win25_2': WS / 'tmp/base_today/win25/mlmq',
    'cwin25_2': WS / 'tmp/base_today/chainwin25/mlmq',
    'chain_2': WS / 'tmp/base_today/chain224/mlmq',
}


def parse(stdout, tag):
    rows = []
    for line in stdout.splitlines():
        if line.startswith('BENCH '):
            d = {}
            for tok in line.split()[1:]:
                if '=' in tok:
                    d[tok.split('=', 1)[0]] = tok.split('=', 1)[1]
            rows.append(d)
    return rows


def grab(stdout, key):
    out = []
    for line in stdout.splitlines():
        if key in line:
            out.append(line.strip())
    return out


def run_one(binary, graph, n_gpu, source, out, repeats=3, warmups=1, timeout=300):
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
    rows = parse(stdout, 'BENCH ')
    formal = [float(r['solve_ms']) for r in rows if r.get('warmup') == '0']
    ok = len(formal) == repeats and all(r.get('correct') == '1' for r in rows if r.get('warmup') == '0')
    # aggregate work counts across all formal samples
    works = grab(stdout, 'L3_WORK gpu')
    l2s = grab(stdout, 'L2_FINAL gpu')
    return dict(rc=rc, solve_ms=formal, correct=ok, median_ms=median(formal) if formal else None,
                work_lines=works[-4:], l2_lines=l2s[-4:])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--graphs', nargs='+', default=['W', 'USA'])
    ap.add_argument('--variants', nargs='+', default=['t512_1', 't512_2', 'win25_2', 'cwin25_2'])
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
                rec = dict(round=turn, graph=g, variant=v, n_gpu=n, **{k: val for k, val in r.items()})
                records.append(rec)
                (out / 'records.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(round=turn, graph=g, variant=v, rc=r['rc'],
                                      correct=r['correct'], med=round(r['median_ms'], 3) if r['median_ms'] else None)),
                      flush=True)
    summary = []
    for g in a.graphs:
        row = dict(graph=g)
        for v in a.variants:
            ms = [r['median_ms'] for r in records if r['graph'] == g and r['variant'] == v and r['rc'] == 0]
            if ms:
                row[v] = round(median(ms), 4)
                row[v + '_samples'] = [round(x, 3) for x in
                                       sum([r['solve_ms'] for r in records
                                            if r['graph'] == g and r['variant'] == v and r['rc'] == 0], [])]
        if 't512_1' in row:
            for v in a.variants:
                if v != 't512_1' and v in row and row[v]:
                    row['sp_' + v] = round(row['t512_1'] / row[v], 4)
        summary.append(row)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('PROBE_DONE'); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
