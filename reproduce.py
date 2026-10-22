"""Rebuild paper simulation tables and figures using package-local inputs."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ['first_10', 'first_50', 'highdeg_10', 'highdeg_50',
           'root', 'depth_1', 'depth_2', 'overall_classified']


def select_input(scenario, alpha, phase, supplied=None):
    """Only finalized CSVs qualify; never merge partial and completed trials."""
    filename = f'{scenario}_alpha{alpha:g}_{phase}.csv'
    directories = ([supplied] if supplied else []) + [
        ROOT / 'results/raw', ROOT / 'results/amarel/rerun_20261015']
    for directory in directories:
        path = directory / filename
        if path.is_file():
            return path, 'completed rerun'
    raise FileNotFoundError(f'No packaged input for {scenario}, alpha={alpha}, {phase}')


def rewrite_table(text, label, means):
    def replace_table(match):
        table = match.group()
        if '\\label{' + label + '}' not in table:
            return table
        for alpha in (2, 0, -.5):
            marker = r'\multirow{5}{*}{' + f'{alpha:g}' + '}'
            start = table.index(marker) + len(marker)
            end = table.index(r'\hline', start)
            def row(match):
                n = int(match[1])
                values = means.loc[(alpha, n), METRICS]
                return '& ' + str(n) + ' & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\'
            block, count = re.subn(r'&\s*(\d+)\s*&[^\n]*?\\\\', row, table[start:end])
            if count != 5:
                raise ValueError(f'Expected five rows in {label}, alpha={alpha}')
            table = table[:start] + block + table[end:]
        return table
    return re.sub(r'\\begin\{table\}.*?\\end\{table\}', replace_table, text, flags=re.S)


def rebuild(output, supplied=None):
    sys.path.insert(0, str(ROOT / 'paper_plots'))
    import analyze_results_all_methods_mean as lines
    import analyze_results_alpha0_rate1_weights as weights
    staged = output / 'inputs'
    staged.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    tables = output / 'tables'
    tables.mkdir(exist_ok=True)
    provenance, distances = [], {}
    for scenario in ('balanced', 'unbalanced'):
        frames = []
        for alpha in (2, 0, -.5):
            phases = ('distance', 'benchmarks', 'montecarlo') if scenario == 'balanced' else ('distance',)
            for phase in phases:
                source, status = select_input(scenario, alpha, phase, supplied)
                frame = pd.read_csv(source)
                if frame.duplicated(['strategy', 'n', 'theta_rate', 'trial']).any():
                    raise ValueError(f'Duplicate trial records: {source}')
                provenance.append(dict(scenario=scenario, alpha=alpha, phase=phase,
                    source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                    status=status, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
                suffix = '_3_scaled_theta' if scenario == 'unbalanced' else '_scaled_theta'
                suffix += {'distance': '', 'benchmarks': '_benchmarks', 'montecarlo': '_montecarlo'}[phase]
                shutil.copy2(source, staged / f'distance_recovery_raw_data_experiment{alpha:g}{suffix}.csv')
                if phase == 'distance':
                    frame = frame.loc[frame.strategy == 'with_protected'].copy()
                    if 'status' in frame:
                        frame = frame.loc[frame.status == 'ok']
                    frame['alpha'] = alpha
                    frames.append(frame)
        distances[scenario] = pd.concat(frames, ignore_index=True)
        module = importlib.import_module('analyze_results_scaled_theta' if scenario == 'balanced'
                                       else 'analyze_results_3_scaled_theta')
        data = {alpha: module.load_raw_data(str(staged / (
            f'distance_recovery_raw_data_experiment{alpha:g}' +
            ('_3_scaled_theta.csv' if scenario == 'unbalanced' else '_scaled_theta.csv'))))
            for alpha in (2, 0, -.5)}
        module.generate_combined_boxplots(data, str(figures))
        for rate, short in [('rate_1', 'n1'), ('rate_3_4', 'n34'), ('rate_1_2', 'n12')]:
            means = distances[scenario].loc[lambda d: d.theta_rate == rate].groupby(['alpha', 'n'])[METRICS].mean()
            label = 'tab:mean_error_scale_' + short + (' 3' if scenario == 'unbalanced' else '')
            snapshot = ROOT / f'results/paper/tables/{scenario}_{rate}.tex'
            (tables / snapshot.name).write_text(rewrite_table(snapshot.read_text(encoding='utf-8'), label, means), encoding='utf-8')
    for metric in lines.METRICS_TO_PLOT:
        lines.plot_combined_mean(lines.load_method_means(str(staged), metric), metric, str(figures))
    means = weights.load_means(staged, 1)
    for group in weights.WEIGHT_GROUPS:
        weights.plot_weights(means, group, figures, 1)
    (output / 'input_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(list(figures.glob("*.pdf")))} figures and six tables to {output}')
    print('Input sources are identified in input_manifest.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced')
    parser.add_argument('--data-dir', type=Path, help='Prefer finalized new simulation CSVs in this folder')
    args = parser.parse_args()
    rebuild(args.output.resolve(), args.data_dir.resolve() if args.data_dir else None)


if __name__ == '__main__':
    main()
