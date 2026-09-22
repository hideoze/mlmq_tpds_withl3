#!/usr/bin/env python3
"""ADDS(1 GPU) on the frozen 37327 same-graph G/G+ with the identical protocol.

Reads the frozen graphs.json exported by job 37327, runs the checksum-pinned
official ADDS binary on original and augmented views, two interleaved rounds
with reversed order, 1 warmup + 5 formal per process, slow samples kept.
Run inside Slurm only.
"""
import json, os, sys
from pathlib import Path
from run_usa_road_matrix import run_one, sha256

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'tmp/l3_adds_sameprotocol_20260922')
OUT = BASE / 'adds_matrix'
assert os.environ.get('SLURM_JOB_ID')
OUT.mkdir(parents=True, exist_ok=False)
frozen = json.loads((ROOT / 'tmp/l3_supplement_20260921_v2/matrix/graphs.json').read_text())
adds = ROOT / 'tmp/adds_official_v4/adds'
expected_adds = '511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2'
assert sha256(adds) == expected_adds, sha256(adds)
for g, data in frozen.items():
    for view in ('original', 'augmented'):
        p = Path(data[view])
        assert p.is_file(), p
        actual = sha256(p)
        assert actual == data['hashes'][view], (g, view, actual)
(OUT / 'inputs.json').write_text(json.dumps(dict(
    adds=dict(path=str(adds), sha256=expected_adds),
    frozen_graphs=str(ROOT / 'tmp/l3_supplement_20260921_v2/matrix/graphs.json'),
    protocol=dict(warmups=1, formal=5, rounds=2, reversed_round2=True,
                  source='frozen 37327 graphs and sources'),
), indent=2) + '\n')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('MLMQ_', 'BENCH_', 'L3_SUPPLEMENT_'))}
env.update(BENCH_WARMUPS='1', BENCH_REPEATS='5')
views = [('ADDS-G', 'original', 1), ('ADDS-Gp', 'augmented', 1)]
records = []
for round_id in range(2):
    names = list(frozen) if round_id == 0 else list(reversed(list(frozen)))
    for g in names:
        data = frozen[g]
        for label, view, n in (views if round_id == 0 else list(reversed(views))):
            env['BENCH_SOURCE'] = str(data['source'])
            prefix = OUT / f'r{round_id}_{g}_{label}'
            r = run_one(adds, Path(data[view]), 'ADDS', n, prefix, env,
                        data['source'], 'L1SLF_L2DQ', 1, 5, 400)
            records.append(dict(round=round_id, graph=g, variant=label, **r))
            (OUT / 'records.json').write_text(json.dumps(records, indent=2))
            formal = [float(s['solve_ms']) for s in r['samples'] if s['warmup'] == '0']
            print(json.dumps(dict(round=round_id, graph=g, variant=label, rc=r['rc'],
                                  valid=r['valid'], reason=r['reason'], solve=formal)),
                  flush=True)
            if not r['valid']:
                raise RuntimeError('Retained failed run: ' + str(prefix))
(OUT / 'complete.json').write_text(json.dumps(dict(
    processes=len(records), queries=len(records) * 6, formal=len(records) * 5,
    valid=True), indent=2) + '\n')
print('ADDS_MATRIX_PASS processes=16 queries=96 formal=80', flush=True)
