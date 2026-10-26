"""Reproduce a sufficient IID resolution frontier, not estimated power.

For unique, different population MSTs with both cycle gaps at least Delta,
Corollary 8.2 implies P(L_H > 0) >= 1-alpha for n >= sufficient_n.
The gap is an assumed population effect, not a data-selected tuning value.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def sufficient_n(p, gap, alpha=.05):
    if p < 3 or not 0 < gap <= 2 or not 0 < alpha < 1:
        raise ValueError('Require p >= 3, 0 < gap <= 2 and 0 < alpha < 1')
    m = p * (p - 1) // 2
    return 2 * (math.floor(32 * math.log(4 * m / alpha) / gap**2) + 1)


def radius(n, p, alpha=.05):
    m = p * (p - 1) // 2
    return min(2., math.sqrt(2 * math.log(4 * m / alpha) / (n // 2)))


def main(root):
    out = root / 'results/power_frontier'
    out.mkdir(parents=True, exist_ok=True)
    (root / 'figures').mkdir(exist_ok=True)
    (root / 'manuscript').mkdir(exist_ok=True)
    rows = []
    for p in [8, 49]:
        for gap in [.1, .2, .4, .6]:
            n = sufficient_n(p, gap)
            # Check the strict inequality and the smallest qualifying integer.
            assert 4 * radius(n, p) < gap
            assert 4 * radius(n - 1, p) >= gap
            rows.append(dict(p=p, gap=gap, alpha=.05, n_sufficient=n))
    with (out / 'thresholds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    star_gap = 2 * (math.asin(.65) - math.asin(.65**2)) / math.pi
    metadata = dict(
        target='Sufficient 95% probability of exact recovery for two unique MSTs',
        assumptions='IID within each period, unique MSTs, both cycle gaps >= Delta',
        detection='If the two trees differ, exact recovery implies L_H > 0',
        formula='2 * (floor(32 * log(4*m/alpha) / Delta**2) + 1)',
        not_claimed='Minimal required sample size, estimated power, or financial-data guarantee',
        star_gap=star_gap, star_n_sufficient=sufficient_n(8, star_gap),
        threshold_boundary_checks=len(rows))
    (out / 'design.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Nodes & $\Delta=.1$ & $\Delta=.2$ & $\Delta=.4$ & $\Delta=.6$ \\',
             r'\midrule']
    for p in [8, 49]:
        vals = [str(r['n_sufficient']) for r in rows if r['p'] == p]
        lines.append(' & '.join([str(p)] + vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (root / 'manuscript/power_frontier_table.tex').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    gaps = np.linspace(.1, .7, 250)
    for p, color in [(8, '#306d8c'), (49, '#c56e29')]:
        ax.plot(gaps, [sufficient_n(p, d) for d in gaps], color=color,
                lw=2, label=f'{p} nodes')
        for gap in [.2, .4, .6]:
            n = sufficient_n(p, gap)
            ax.plot(gap, n, 'o', color=color, ms=4)
            if p == 49:
                ax.annotate(f'{n:,}', (gap, n), xytext=(5, 6),
                            textcoords='offset points', fontsize=8, color=color)
    ax.axhline(1006, color='#59636c', lw=1, ls='--', label='Application size: 1,006')
    ax.set_yscale('log')
    ax.set_ylim(800, 90000)
    ax.set_yticks([1000, 5000, 20000, 80000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlabel(r'Assumed minimum population cycle gap $\Delta$ (Kendall units)')
    ax.set_ylabel('Sufficient observations\nper period')
    ax.set_title('IID benchmark: at least 95% probability of complete recovery', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.16)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout(pad=.8)
    fig.savefig(root / 'figures/power_frontier.pdf')
    fig.savefig(root / 'figures/power_frontier.png', dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
