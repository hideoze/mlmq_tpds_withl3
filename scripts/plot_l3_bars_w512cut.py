"""Grouped-bar L3 road performance figure, current version (W512 + per-graph cut).

Style follows figures/style_reference/new_pic/road.py: light green/orange/purple/
blue fills, bar_width 0.25-0.35, bold axis labels, per-graph subpanels sharing
the x axis, PDF+SVG+PNG outputs.

Panels:
  (a) solve time: no-L3 single GPU vs L3 dual GPU, one bar pair per graph
  (b) dual-GPU speedup with 1x reference line and geometric-mean line

Data: figures/l3_evaluation/metrics_w512cut.csv (generated from step3 records;
medians of 6 formal samples per configuration, all CPU-verified).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, default=REPO / 'figures/l3_evaluation')
p.add_argument('--metrics', type=Path,
              default=REPO / 'figures/l3_evaluation/metrics_w512cut.csv')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(a.metrics.open()))
order = ['NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA']
by_graph = {r['graph']: r for r in rows}
assert set(order) == set(by_graph), 'metrics file graph set mismatch'

nol3 = [float(by_graph[g]['nol3_1_ms']) for g in order]
l3 = [float(by_graph[g]['l3roadcut_2_ms']) for g in order]
sp = [float(by_graph[g]['speedup']) for g in order]
geo = math.exp(sum(math.log(x) for x in sp) / len(sp))
for g, n, l, s in zip(order, nol3, l3, sp):
    assert abs(n / l - s) < 5e-3, f'speedup identity fails for {g}'

bar_color = ['#B8DDBC', '#F0A780', '#9392BE', '#96B6D8']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.labelweight': 'bold', 'axes.linewidth': .8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none',
                     'savefig.bbox': 'tight', 'savefig.pad_inches': .06})

def style(ax):
    ax.tick_params(direction='out', width=.8)
    ax.set_axisbelow(True)

width = 7
height = 4.6
fig, axs = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                        layout='constrained',
                        gridspec_kw={'height_ratios': [1.25, 1]})
x = np.arange(len(order))
bar_width = 0.35

# (a) absolute solve time, log scale for cross-graph magnitude range
axs[0].bar(x - bar_width / 2, nol3, width=bar_width, color=bar_color[1],
           edgecolor=bar_color[1], label='MLMQ no-L3 (1 GPU)')
axs[0].bar(x + bar_width / 2, l3, width=bar_width, color=bar_color[2],
           edgecolor=bar_color[2], label='MLMQ + L3 (2 GPUs)')
axs[0].set_yscale('log')
axs[0].set_ylabel('Solve time (ms, log scale)')
axs[0].legend(loc='upper left', fontsize=8.5, frameon=True)
axs[0].text(.015, .96, '(a)', transform=axs[0].transAxes, va='top',
            fontweight='bold')

# (b) speedup with references
axs[1].bar(x, sp, width=bar_width, color=bar_color[3], edgecolor=bar_color[3],
           label='L3 dual-GPU speedup (vs no-L3 1 GPU)')
axs[1].axhline(1, color='#404040', lw=1.4, ls='--')
axs[1].axhline(geo, color=bar_color[1], lw=1.4, ls=':',
               label=f'Geometric mean = {geo:.3f}$\\times$')
for i, s in enumerate(sp):
    axs[1].annotate(f'{s:.3f}', (i, s), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=8)
axs[1].set_ylim(0, 1.85)
axs[1].yaxis.set_major_locator(MultipleLocator(.25))
axs[1].set_ylabel('Speedup over no-L\nsingle GPU')
axs[1].set_xlabel('Road network graph name')
axs[1].set_xticks(x, order)
axs[1].legend(loc='upper left', fontsize=8.5, frameon=True)
axs[1].text(.015, .96, '(b)', transform=axs[1].transAxes, va='top',
            fontweight='bold')

for ax in axs:
    style(ax)
    ax.set_xlim(-.6, len(order) - .4)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(a.out / f'l3_road_bars_w512cut.{ext}', dpi=400)
plt.close(fig)

prov = dict(
    metrics=str(a.metrics.relative_to(REPO)),
    metrics_sha256=hashlib.sha256(a.metrics.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    version='L3-ROAD-CHAIN-WIN25-W512-CUT-20260916',
    style_reference='figures/style_reference/new_pic/road.py',
    geomean_speedup=round(geo, 4),
    note='grouped bars: (a) solve times (log axis); (b) speedups with 1x and geomean reference lines',
)
(a.out / 'provenance_l3_road_bars_w512cut.json').write_text(json.dumps(prov, indent=2) + '\n')
print(f'geomean={geo:.4f} written to {a.out}')
