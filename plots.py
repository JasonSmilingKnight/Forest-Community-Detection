"""Validate trial CSVs and export paper tables, figures, and comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import METRICS, RATES, SIZES, summarize

ROOT = Path(__file__).resolve().parent
METHODS = {'with_protected': 'SPAR', 'SCORE': 'SCORE', 'Gao-DCBM': 'GMZZ (local solver)',
           'MonteCarlo': 'SPAR + Monte Carlo'}
TABLE_METRICS = ['first_10', 'first_50', 'highdeg_10', 'highdeg_50', 'root',
                 'depth_1', 'depth_2', 'overall_classified']
ARRIVAL = ['weighted_overall', 'weighted_overall_t1_log', 'weighted_overall_t3_2', 'weighted_overall_r2']
DEGREE = ['weighted_overall_deg2', 'weighted_overall_deg2_log', 'weighted_overall_deg3', 'weighted_overall_deg4']
WEIGHT_LABELS = dict(zip(ARRIVAL + DEGREE, [
    r'$w_t=t^{-1}$', r'$w_t=(t\log t)^{-1}$', r'$w_t=t^{-3/2}$', r'$w_t=t^{-2}$',
    r'$w_v=d_v^2$', r'$w_v=d_v^2(1+\log d_v)$', r'$w_v=d_v^3$', r'$w_v=d_v^4$']))
FIGURE_METRICS = dict(first_50='first_50', highdeg_50='highdeg_50', depth_1='depth_1',
    weighted_overall='weighted_overall_t1', weighted_overall_t1_log='weighted_overall_t1_log',
    weighted_overall_t3_2='weighted_overall_t3_2', weighted_overall_r2='weighted_overall_t2',
    **{m: m for m in DEGREE})


def load_trials(source):
    """Read new records or historical CSVs; do not infer unknown fallback counts."""
    source = Path(source)
    if source.suffix == '.zip':
        with ZipFile(source) as archive:
            files = [(name, archive.read(name)) for name in sorted(archive.namelist()) if name.endswith('.csv')]
    else:
        files = [(p.name, p.read_bytes()) for p in sorted(source.glob('*.csv'))
                 if not any(s in p.name for s in ('_means', '_calibration', '.partial', '.tmp'))]
    from io import BytesIO
    frames, provenance = [], []
    for name, raw in files:
        if '_c2' in name:
            continue
        frame = pd.read_csv(BytesIO(raw))
        if not {'strategy', 'n', 'theta_rate', 'trial', 'cluster_count', 'overall_classified'} <= set(frame):
            continue
        if 'scenario' not in frame:
            match = re.search(r'experiment(-?[\d.]+)(_3)?_scaled_theta', name)
            if not match:
                raise ValueError(f'cannot identify scenario: {name}')
            frame['alpha'] = float(match[1])
            frame['scenario'] = 'unbalanced' if match[2] else 'balanced'
        frame['alpha'] = pd.to_numeric(frame['alpha']).astype(float)
        expected = frame.scenario.map({'balanced': 2, 'unbalanced': 3})
        if expected.isna().any():
            raise ValueError(f'unknown scenario: {name}')
        if 'status' not in frame:
            frame['status'] = np.where((frame.cluster_count == expected) & frame.overall_classified.notna(), 'ok', 'failed')
        if not frame.status.isin(['ok', 'failed']).all():
            raise ValueError(f'unknown trial status: {name}')
        if ((frame.status == 'ok') & ((frame.cluster_count != expected) | frame.overall_classified.isna())).any():
            raise ValueError(f'inconsistent successful trial: {name}')
        frame['fallback_known'] = 'result_source' in frame
        if 'result_source' not in frame:
            frame['result_source'] = 'unknown'
        if 'classified_fraction' not in frame:
            frame['classified_fraction'] = np.nan
        for metric in METRICS:
            if metric not in frame:
                frame[metric] = np.nan
            observed = frame[metric].dropna()
            if not observed.between(0, 1).all():
                raise ValueError(f'out-of-range {metric}: {name}')
        frames.append(frame)
        provenance.append(dict(file=name, sha256=hashlib.sha256(raw).hexdigest(), rows=len(frame)))
    if not frames:
        raise ValueError('no trial records found')
    df = pd.concat(frames, ignore_index=True)
    keys = ['scenario', 'strategy', 'alpha', 'n', 'theta_rate', 'trial']
    if df.duplicated(keys).any():
        raise ValueError('duplicate trial keys; do not mix reruns with references')
    if 'graph_hash' in df and df.graph_hash.notna().any():
        groups = df.dropna(subset=['graph_hash']).groupby(['scenario', 'alpha', 'n', 'theta_rate', 'trial'])
        if (groups.graph_hash.nunique() > 1).any():
            raise ValueError('methods were evaluated on different graphs')
    return df, provenance


def _summary(df):
    result = summarize(df)
    unknown = df.groupby(['scenario', 'strategy', 'alpha', 'n', 'theta_rate']).fallback_known.all()
    result['fallback_count_known'] = [bool(unknown.loc[tuple(row)]) for row in result[['scenario', 'strategy', 'alpha', 'n', 'theta_rate']].to_numpy()]
    result.loc[~result.fallback_count_known, 'fallback_trials'] = np.nan
    result['se'] = result['std'] / np.sqrt(result.valid_trials)
    return result


def _finish(fig, path, label):
    fig.text(.01, .005, label, fontsize=8, color='#555555')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _write_latex_table(frame, path):
    """Write a compact booktabs table without an optional template dependency."""
    lines = [r'\begin{tabular}{' + 'r' * (len(frame.columns) + 2) + '}', r'\toprule',
             ' & '.join([r'$\alpha$', '$n$'] + [str(c).replace('_', r'\_') for c in frame.columns]) + r' \\', r'\midrule']
    for (alpha, n), values in frame.iterrows():
        cells = [f'{alpha:g}', str(int(n))] + ['--' if pd.isna(v) else f'{v:.2f}' for v in values]
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def draw_grid(df, means, metric, path, label, box=False):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharey=True)
    for i, alpha in enumerate((-.5, 0, 2)):
        for j, rate in enumerate(RATES):
            ax = axes[i, j]
            cell = df[(df.alpha == alpha) & (df.theta_rate == rate) & (df.status == 'ok')]
            if box:
                cell = cell[cell.strategy == 'with_protected']
                groups = [(n, g[metric].dropna()) for n, g in cell.groupby('n') if g[metric].notna().any()]
                if groups:
                    ax.boxplot([g for n, g in groups], tick_labels=[str(n) for n, g in groups], showfliers=False)
            else:
                for strategy, name in METHODS.items():
                    data = means[(means.alpha == alpha) & (means.theta_rate == rate) &
                                 (means.metric == metric) & (means.strategy == strategy)].sort_values('n')
                    if data['mean'].notna().any():
                        ax.plot(data.n, data['mean'], marker='o', markersize=3, label=name)
            ax.set_title(rf'$\alpha={alpha:g}$, $\beta={RATES[rate]:g}$', fontsize=10)
            if not cell[metric].notna().any():
                ax.text(.5, .5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='#777777', fontsize=9)
                ax.set_xticks([])
            ax.set_ylim(-.02, .7 if (df.scenario == 'unbalanced').any() else .55)
            ax.grid(alpha=.2)
            if i == 2:
                ax.set_xlabel('Number of vertices')
            if j == 0:
                ax.set_ylabel('Misclassification rate')
    if not box:
        handles, labels = {}, {}
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.update(zip(l, h))
        fig.legend(handles.values(), handles.keys(), loc='upper center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .02, 1, .95))
    _finish(fig, path, label)


def draw_weights(means, metrics, path, label):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, metric in zip(axes, metrics):
        for strategy, name in METHODS.items():
            part = means[(means.alpha == 0) & (means.theta_rate == 'rate_1') &
                         (means.metric == metric) & (means.strategy == strategy)].sort_values('n')
            if part['mean'].notna().any():
                ax.plot(part.n, part['mean'], marker='o', markersize=3, label=name)
        ax.set_title(WEIGHT_LABELS[metric], fontsize=10)
        ax.set_xlabel('Number of vertices')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Weighted misclassification rate')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .9))
    _finish(fig, path, label)


def export(source, output, reference=None, review_threshold=.05):
    df, provenance = load_trials(source)
    output = Path(output)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    means = _summary(df)
    means.to_csv(tables / 'means.csv', index=False)
    pure_mc = df[(df.strategy == 'MonteCarlo') & (df.result_source == 'MonteCarlo')]
    if len(pure_mc):
        _summary(pure_mc).to_csv(tables / 'monte_carlo_without_fallback.csv', index=False)
    keys = ['scenario', 'strategy', 'alpha', 'n', 'theta_rate']
    means.drop_duplicates(keys)[keys + ['total_trials', 'success_rate', 'fallback_trials', 'fallback_count_known']].to_csv(tables / 'run_status.csv', index=False)
    pilot = means.total_trials.min() < 200
    label = 'Pilot: fewer than 200 trials per cell; not a reproduction claim' if pilot else '200-trial summary; failures and fallback counts in run_status.csv'
    if not df.fallback_known.all():
        label = 'Historical reference; fallback counts were not recorded'
    for scenario, part in df.groupby('scenario'):
        selected = means[means.scenario == scenario]
        for rate in RATES:
            wide = selected[(selected.theta_rate == rate) & (selected.strategy == 'with_protected') & selected.metric.isin(TABLE_METRICS)].pivot(index=['alpha', 'n'], columns='metric', values='mean').reindex(columns=TABLE_METRICS)
            wide.to_csv(tables / f'{scenario}_{rate}.csv')
            _write_latex_table(wide, tables / f'{scenario}_{rate}.tex')
        for metric in ('first_50', 'highdeg_50', 'depth_1'):
            prefix = 'boxplot_3x3_' if scenario == 'balanced' else 'boxplot_3x3_K3_'
            draw_grid(part, selected, metric, figures / f'{prefix}{metric}.pdf', label, box=True)
        if scenario == 'balanced':
            for metric, suffix in FIGURE_METRICS.items():
                if selected.loc[selected.metric == metric, 'mean'].notna().any():
                    draw_grid(part, selected, metric, figures / f'mean_3x3_{suffix}.pdf', label)
            for metrics, name in ((ARRIVAL, 'arrival_time'), (DEGREE, 'degree')):
                if selected.loc[selected.metric.isin(metrics), 'mean'].notna().any():
                    draw_weights(selected, metrics, figures / f'mean_1x4_c1_alpha0_rate1_{name}_weights.pdf', label)
    manifest = json.loads((ROOT / 'tests/fixtures/paper_manifest.json').read_text())
    reference_rows = []
    for name, table in manifest['tables'].items():
        scenario = 'unbalanced' if name.endswith(' 3') else 'balanced'
        rate = 'rate_3_4' if 'n34' in name else 'rate_1_2' if 'n12' in name else 'rate_1'
        for row in table['rows']:
            reference_rows.extend(dict(scenario=scenario, alpha=row['alpha'], n=row['n'], theta_rate=rate,
                strategy='with_protected', metric=m, paper=row[m]) for m in TABLE_METRICS)
    comparison = means.merge(pd.DataFrame(reference_rows), on=keys + ['metric'], how='inner')
    comparison['difference'] = comparison['mean'] - comparison.paper
    comparison['rounding_agreement'] = comparison.difference.abs() <= .005 + 1e-12
    comparison['review_required'] = (comparison.difference.abs() > review_threshold) | (comparison.valid_trials < 200)
    comparison.to_csv(tables / 'paper_comparison.csv', index=False)
    if reference:
        before, _ = load_trials(reference)
        paired = means.merge(_summary(before), on=keys + ['metric'], suffixes=('_after', '_before'))
        paired['difference'] = paired.mean_after - paired.mean_before
        paired.to_csv(tables / 'before_after.csv', index=False)
    report = dict(inputs=provenance, pilot=bool(pilot), review_threshold=review_threshold,
        note='Absolute-difference screening only; small differences do not certify scientific equivalence.',
        outputs={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (tables, figures) for p in sorted(folder.iterdir()) if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--review-threshold', type=float, default=.05)
    args = parser.parse_args()
    export(args.input, args.output, args.reference, args.review_threshold)
