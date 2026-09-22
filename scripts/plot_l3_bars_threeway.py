"""Three-way L3 road bars with ADDS baseline (same-batch measurements).

Style follows figures/style_reference/new_pic/road.py and the W512 cut bars:
light fills, bold axis labels, shared-x subpanels, PDF+SVG+PNG.

Panels:
  (a) solve time bars: ADDS (1 GPU) / MLMQ no-L3 (1 GPU) / MLMQ+L3 (2 GPUs)
  (b) speedups over ADDS for both MLMQ configurations, with geometric means

Data (no hard-coded numbers):
  ADDS1, nol3_1 : tmp/base_today/run_threeway1gpu/records.json (same batch,
                  2 reversed rounds x 1 warmup + 3 formal, all CPU-verified)
  l3cut_2       : tmp/base_today/run_step3/records.json (same protocol; the
                  cross-batch nol3_1 drift measured <1.5%, median <0.5%)
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

REPO = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, default=REPO / 'figures/l3_evaluation')
p.add_argument('--records', type=Path,
              default=REPO / 'tmp/base_today/run_threeway_samebatch/records.json')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
tw = json.loads(a.records.read_text())
s3 = tw  # same batch: all three variants from one interleaved measurement
order = ['NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA']
CUT = {'NY': 60, 'COL': 55, 'W': 40, 'USA': 60}


def med(records, graph, variant):
    xs = [r['median_ms'] for r in records
          if r['graph'] == graph and r['variant'] == variant
          and r['rc'] == 0 and r['correct']]
    assert len(xs) == 2, f'{graph}/{variant}: {len(xs)} rounds'
    # same batch: every formal sample verified upstream; medians over 2x5 samples
    return median(xs)


rows = []
for g in order:
    adds = med(tw, g, 'adds1')
    nol3 = med(tw, g, 'nol3_1')
    l3 = med(tw, g, 'l3cut_2')
    rows.append(dict(graph=g, adds1_ms=round(adds, 6),
                     nol3_1_ms=round(nol3, 6), l3cut_2_ms=round(l3, 6),
                     adds_over_nol3=round(adds / nol3, 4),
                     adds_over_l3=round(adds / l3, 4),
                     nol3_over_l3=round(nol3 / l3, 4),
                     cut_percent=CUT.get(g, 50),
                     version='L3-ROAD-CHAIN-WIN25-W512-CUT-20260916'))
geo_adds_l3 = math.exp(sum(math.log(r['adds_over_l3']) for r in rows) / len(rows))
geo_adds_nol3 = math.exp(sum(math.log(r['adds_over_nol3']) for r in rows) / len(rows))
geo_nol3_l3 = math.exp(sum(math.log(r['nol3_over_l3']) for r in rows) / len(rows))
for r in rows:
    assert abs(r['adds_over_nol3'] * r['nol3_over_l3'] - r['adds_over_l3']) < 2e-2


def sig2(v):
    """Two significant digits without trailing zeros: 136.288 -> 140, 6.876 -> 6.9."""
    return f"{float(f'{v:.2g}')}"

bar_color = ['#96B6D8', '#F0A780', '#9392BE']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.labelweight': 'bold', 'axes.linewidth': .8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none',
                     'savefig.bbox': 'tight', 'savefig.pad_inches': .06})

def style(ax):
    ax.tick_params(direction='out', width=.7, labelsize=7)
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(3.5, 3.1))
# top band (>=0.86) is reserved for the OUTSIDE framed legend;
# bottom band for tick labels + x-axis title; middle split 55/45.
axs = [fig.add_axes([0.20, 0.57, 0.775, 0.33]),
       fig.add_axes([0.20, 0.135, 0.775, 0.315])]
x = np.arange(len(order))
# bar half-spacing (center offset) and width: centers +-0.30 with width 0.26
bo = 0.30
bw = 0.26

axs[0].bar(x - bo, [r['adds1_ms'] for r in rows], width=bw,
           color=bar_color[0], edgecolor=bar_color[0], label='ADDS (1 GPU)')
axs[0].bar(x, [r['nol3_1_ms'] for r in rows], width=bw,
           color=bar_color[1], edgecolor=bar_color[1], label='MLMQ no-L3 (1 GPU)')
axs[0].bar(x + bo, [r['l3cut_2_ms'] for r in rows], width=bw,
           color=bar_color[2], edgecolor=bar_color[2], label='MLMQ + L3 (2 GPUs)')
axs[0].set_yscale('log')
axs[0].set_ylabel('Solve time\n(ms, log scale)', fontsize=7.5)
axs[0].set_xticks(x, order)


axs[1].bar(x - bo / 2, [r['adds_over_nol3'] for r in rows], width=bw,
           color=bar_color[1], edgecolor=bar_color[1],
           label='no-L3 1 GPU over ADDS')
axs[1].bar(x + bo / 2, [r['adds_over_l3'] for r in rows], width=bw,
           color=bar_color[2], edgecolor=bar_color[2],
           label='L3 2 GPUs over ADDS')
axs[1].axhline(1, color='#404040', lw=1.4, ls='--')
axs[1].axhline(geo_adds_nol3, color=bar_color[1], lw=1.3, ls=':')
axs[1].axhline(geo_adds_l3, color=bar_color[2], lw=1.3, ls=':')
axs[1].set_ylim(0, 4.6)
axs[1].yaxis.set_major_locator(MultipleLocator(.5))
axs[1].set_ylabel('Speedup\nover ADDS', fontsize=7.5)
axs[1].set_xlabel('Graph Name (Road Network)')
axs[1].set_xticks(x, order)


# Single shared legend above the top panel, one horizontal row, paper style.
handles = [plt.Rectangle((0, 0), 1, 1, fc=bar_color[0], ec=bar_color[0]),
           plt.Rectangle((0, 0), 1, 1, fc=bar_color[1], ec=bar_color[1]),
           plt.Rectangle((0, 0), 1, 1, fc=bar_color[2], ec=bar_color[2])]
labels = ['ADDS (1 GPU)', 'MLMQ no-L3 (1 GPU)', 'MLMQ + L3 (2 GPUs)']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0),
           ncols=3, frameon=True, framealpha=.95, edgecolor='#cccccc',
           fontsize=6, handlelength=.9, handletextpad=.4,
           columnspacing=.8, borderpad=.4, borderaxespad=0)

for ax in axs:
    style(ax)
    ax.set_xlim(-.6, len(order) - .4)

with (a.out / 'metrics_threeway_bars.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(a.out / f'l3_road_bars_threeway.{ext}', dpi=400)
plt.close(fig)

prov = dict(
    records_source=str(a.records.resolve().relative_to(REPO)) if a.records.is_relative_to(REPO) else str(a.records),
    records_sha256=hashlib.sha256(a.records.read_bytes()).hexdigest(),
    adds_binary='tmp/adds_official_v4/adds (511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2)',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    version='L3-ROAD-CHAIN-WIN25-W512-CUT-20260916',
    batches='single same-batch interleaved measurement (adds1/nol3_1/l3cut_2), 2 reversed rounds x 1 warmup + 5 formal',
    geomean=dict(adds_over_l3=round(geo_adds_l3, 4),
                 adds_over_nol3=round(geo_adds_nol3, 4),
                 nol3_over_l3=round(geo_nol3_l3, 4)),
    style_reference='figures/style_reference/new_pic/road.py',
    timing='solve only; preprocessing excluded',
)
(a.out / 'provenance_l3_road_bars_threeway.json').write_text(json.dumps(prov, indent=2) + '\n')
print(f"geomeans: ADDS/L3={geo_adds_l3:.4f} ADDS/nol3={geo_adds_nol3:.4f} nol3/L3={geo_nol3_l3:.4f}")
print(f"written to {a.out}")
