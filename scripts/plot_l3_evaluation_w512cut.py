"""L3 road performance figure, current production version (W512 + per-graph cut).

Data: tmp/base_today/run_step3/records.json — 8 graphs x 2 variants x 2 rounds,
1 warmup + 3 formal per process, all CPU-verified. Comparison is the project's
goal metric: original no-L3 single GPU vs L3 dual GPU (same batch).

Panels:
  (a) dual-GPU speedup (no-L3 1-GPU time / L3 2-GPU time), 1x reference line
  (b) absolute solve times, both variants

Predecessor figure (ADDS-normalized, historical stages) stays untouched as
l3_road_performance.* — this writes l3_road_performance_w512cut.*.
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
              default=REPO / 'tmp/base_today/run_step3/records.json')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
records = json.loads(a.records.read_text())
order = ['NY', 'BAY', 'COL', 'FLA', 'CAL', 'E', 'W', 'USA']
CUT = {'NY': 60, 'COL': 55, 'W': 40, 'USA': 60}

def samples(graph, variant):
    xs = [r for r in records if r['graph'] == graph and r['variant'] == variant]
    assert len(xs) == 2 and all(r['rc'] == 0 and r['correct'] for r in xs)
    # all formal samples across both rounds, warmup excluded upstream
    return [v for r in xs for v in r['solve_ms']]

metrics = []
for g in order:
    nol3 = samples(g, 'nol3_1')
    l3 = samples(g, 'l3roadcut_2')
    nm, lm = median(nol3), median(l3)
    metrics.append(dict(graph=g,
                        nol3_1_ms=round(nm, 6),
                        l3roadcut_2_ms=round(lm, 6),
                        speedup=round(nm / lm, 4),
                        nol3_samples=len(nol3), l3_samples=len(l3),
                        cut_percent=CUT.get(g, 50),
                        version='L3-ROAD-CHAIN-WIN25-W512-CUT-20260916',
                        stage='step3_same_batch'))
geo = math.exp(sum(math.log(m['speedup']) for m in metrics) / len(metrics))
assert all(m['nol3_samples'] == 6 and m['l3_samples'] == 6 for m in metrics)

purple = '#9392BE'; orange = '#F0A780'; blue = '#96B6D8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.labelweight': 'bold', 'axes.linewidth': .8,
                     'lines.linewidth': 1.6, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'savefig.bbox': 'tight', 'savefig.pad_inches': .06})

def style(ax):
    ax.tick_params(direction='out', width=.8)
    ax.set_axisbelow(True)

with (a.out / 'metrics_w512cut.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(metrics[0]))
    w.writeheader()
    w.writerows(metrics)

fig, axs = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True,
                        layout='constrained',
                        gridspec_kw={'height_ratios': [1.25, 1]})
x = np.arange(len(metrics))
axs[0].plot(x, [m['speedup'] for m in metrics], color=purple, marker='X', ms=7,
            label='L3 dual-GPU speedup (vs no-L3 1 GPU, same batch)')
axs[0].axhline(1, color=blue, lw=1.4, ls='--', label='1$\\times$ reference')
axs[0].axhline(geo, color=orange, lw=1.3, ls=':',
               label=f'Geometric mean = {geo:.3f}$\\times$')
for i, m in enumerate(metrics):
    axs[0].annotate(f"{m['speedup']:.3f}", (i, m['speedup']),
                    xytext=(0, 7), textcoords='offset points',
                    ha='center', fontsize=8)
axs[0].set_ylabel('Speedup over no-L3 single GPU')
axs[0].set_ylim(0, 1.85)
axs[0].yaxis.set_major_locator(MultipleLocator(.25))
axs[0].legend(loc='upper left', fontsize=8.5, frameon=True)
axs[0].text(.015, .96, '(a)', transform=axs[0].transAxes, va='top',
            fontweight='bold')

axs[1].plot(x, [m['nol3_1_ms'] for m in metrics], color=orange, marker='^',
            ms=6, label='MLMQ no-L3 (1 GPU)')
axs[1].plot(x, [m['l3roadcut_2_ms'] for m in metrics], color=purple, marker='X',
            ms=6, label='MLMQ + L3 (2 GPUs)')
axs[1].set_yscale('log')
axs[1].set_ylabel('Solve time (ms, log scale)')
axs[1].set_xlabel('Road network graph name')
axs[1].set_xticks(x, order)
axs[1].legend(loc='upper left', fontsize=8.5, frameon=True)
axs[1].text(.015, .96, '(b)', transform=axs[1].transAxes, va='top',
            fontweight='bold')
for ax in axs:
    style(ax)
    ax.set_xlim(-.35, 7.35)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(a.out / f'l3_road_performance_w512cut.{ext}', dpi=400)
plt.close(fig)

prov = dict(
    records=str(a.records.relative_to(REPO)),
    records_sha256=hashlib.sha256(a.records.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    version='L3-ROAD-CHAIN-WIN25-W512-CUT-20260916',
    build=('chain strict shortcuts (n_gpu>1 only), responsive window 25000 '
           '(boundary index), boundary_index, idle token probe, ACK scan, '
           '512 workers; MLMQ_CUT_PERCENT: NY60/COL55/W40/USA60, others vertex split'),
    comparison=('same-batch: original no-L3 single GPU (512 workers, original '
                'graph) vs L3 dual GPU; 2 reversed rounds x 1 warmup + 3 formal '
                'per process; all 48 dual samples CPU-verified per-vertex'),
    timing='solve only; preprocessing (chain build, reorder) excluded',
    geomean_speedup=round(geo, 4),
    note=('predecessor figure l3_road_performance.* (ADDS-normalized, '
          'historical stages) retained unchanged'),
)
(a.out / 'provenance_w512cut.json').write_text(json.dumps(prov, indent=2) + '\n')
print(f"geomean={geo:.4f} written to {a.out}")
