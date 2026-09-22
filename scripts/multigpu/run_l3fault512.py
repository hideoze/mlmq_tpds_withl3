#!/usr/bin/env python3
"""Fault-injection stress test for the l3road build.

Verifies the dual-GPU protocol still reaches the correct result and terminates
normally when the transport deliberately delays/retries: publish retry, claim
retry, ready delay, ack delay, and hidden RX notifications.
"""
import json, os, re, subprocess, sys
from pathlib import Path
from statistics import median

WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
FAULT_BIN = WS / 'tmp/base_today/fault512/mlmq'
ROAD = {g: (WS / f'tmp/landmark206_matrix/{g}/landmark.gr',
            json.loads((WS / f'tmp/landmark206_matrix/{g}/layout.json').read_text())['source_new_id'])
        for g in ('NY', 'W', 'USA')}
FIXTURES = {
    'late_fan': WS / 'tmp/l3_224_stage/fault/late_fan.gr',
    'zero_fan': WS / 'tmp/l3_224_stage/fault/zero_fan.gr',
}


def run(binary, graph, n_gpu, source, env_extra, timeout=300):
    env = dict(os.environ)
    env.update(MLMQ_BENCH='1', MLMQ_WORK_BLOCKS='107', BENCH_DELTA='200000',
               BENCH_SOURCE=str(source), BENCH_WARMUPS='0', BENCH_REPEATS='1',
               BENCH_QUEUE='L1SLF_L2DQ')
    env.update(env_extra)
    cmd = [str(binary), '-i', str(graph), '-n', str(n_gpu), '-d', '200000']
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or '')


def check_fault_evidence(stdout, tag):
    """The L3_FAULT_AUDIT line survives benchmark quiet mode and reports how
    often each injected transport fault fired.  Hidden-RX-notification evidence
    is the L3_ACK_FAULT recovery line.  Missing/zero counts mean the protocol
    never exercised a recovery path we intended to stress."""
    problems = []
    audits = re.findall(r'L3_FAULT_AUDIT gpu=(\d+) publish=(\d+) claim=(\d+) ready=(\d+) ack=(\d+) pending=(-?\d+)', stdout)
    if not audits:
        problems.append(f'{tag}: no L3_FAULT_AUDIT evidence line')
        return problems
    total = [0, 0, 0, 0]
    for _g, pub, claim, ready, ack, _pend in audits:
        total[0] += int(pub); total[1] += int(claim)
        total[2] += int(ready); total[3] += int(ack)
    if sum(total) == 0:
        problems.append(f'{tag}: all fault counters zero (recovery paths never exercised)')
    if 'L3_ACK_FAULT' not in stdout and 'recovered' not in stdout:
        # L3_ACK_FAULT prints only when the ACK-scan fault path is taken; the
        # ordinary delayed-ACK recovery may show recovered_scans>0.
        if 'L3_ACK_FAULT' not in stdout:
            problems.append(f'{tag}: no L3_ACK_FAULT recovery evidence')
    return problems


def main():
    out = Path('tmp/base_today/run_fault512'); out.mkdir(parents=True, exist_ok=True)
    all_problems = []
    results = []

    # 1) small fault fixtures: two sources, n=2
    for name, path in FIXTURES.items():
        for src in (0, 1025):
            tag = f'{name}_s{src}'
            rc, stdout = run(FAULT_BIN, path, 2, src, {})
            (out / f'{tag}.log').write_text(stdout)
            correct = 'mlmq sssp correct!' in stdout
            problems = [] if (rc == 0 and correct) else [f'{tag}: rc={rc} correct={correct}']
            problems += check_fault_evidence(stdout, tag)
            all_problems += problems
            results.append(dict(fixture=tag, rc=rc, correct=correct, problems=problems))
            print(json.dumps(dict(fixture=tag, rc=rc, correct=correct,
                                  problems=problems)), flush=True)

    # 2) road graphs under full fault injection, n=2, standard source
    for g, (path, src) in ROAD.items():
        tag = f'{g}'
        rc, stdout = run(FAULT_BIN, path, 2, src, {})
        (out / f'{tag}.log').write_text(stdout)
        correct = 'mlmq sssp correct!' in stdout
        problems = [] if (rc == 0 and correct) else [f'{tag}: rc={rc} correct={correct}']
        problems += check_fault_evidence(stdout, tag)
        all_problems += problems
        results.append(dict(road=g, rc=rc, correct=correct, problems=problems))
        print(json.dumps(dict(road=g, rc=rc, correct=correct, problems=problems)), flush=True)

    (out / 'results.json').write_text(json.dumps(dict(results=results,
                                                      all_problems=all_problems), indent=2))
    print('FAULT_TEST_DONE problems=%d' % len(all_problems), flush=True)
    for p in all_problems[:20]:
        print('PROBLEM:', p, flush=True)
    return 1 if all_problems else 0


if __name__ == '__main__':
    sys.exit(main())
