#!/usr/bin/env python3
"""Count chain-shortcut eligibility on Galois binary .gr graphs.

strict = degree-2 vertex, both neighbors same side of cut (current rule adds
in-degree==2); loose = drop the in-degree requirement. Original edges are
always retained, so shortcuts are additive and sound either way.
"""
import struct, sys, json
from pathlib import Path


def load(fn):
    with open(fn, 'rb') as f:
        data = f.read()
    version, sizeEdgeTy, numNodes, numEdges = struct.unpack_from('<QQQQ', data, 0)
    assert version == 1, version
    off = 32
    row = struct.unpack_from(f'<{numNodes+1}Q', data, off)
    off += 8 * numNodes  # row array has numNodes entries + trailing read
    # Galois layout: outIdx has numNodes+1 entries; then dst uint32 array
    off = 32 + 8 * (numNodes + 1)
    dst = struct.unpack_from(f'<{numEdges}I', data, off)
    return numNodes, numEdges, row, dst


def main():
    WS = Path('/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26')
    print(f"{'graph':5} {'n':>9} {'m':>9} {'strict':>8} {'loose':>8} {'gain':>6}")
    for g in ('NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA'):
        fn = WS / f'tmp/landmark206_matrix/{g}/landmark.gr'
        n, m, row, dst = load(fn)
        # row_start in memory is 0-based over nnodes+1; file outIdx matches
        cut = n // 2
        indeg = [0] * n
        for d in dst:
            if d < n:
                indeg[d] += 1
        strict = loose = 0
        for u in range(n):
            deg = row[u + 1] - row[u]
            if deg != 2:
                continue
            a = dst[row[u]]; b = dst[row[u] + 1]
            if a >= n or b >= n or a == u or b == u or a == b:
                continue
            if ((a < cut) == (u < cut)) and ((b < cut) == (u < cut)):
                loose += 1
                if indeg[u] == 2:
                    strict += 1
        print(f"{g:5} {n:9d} {m:9d} {strict:8d} {loose:8d} {loose/max(strict,1):5.2f}x")


if __name__ == '__main__':
    main()
