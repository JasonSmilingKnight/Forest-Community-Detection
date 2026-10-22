"""Copy reviewed case-study PDFs into the packaged manuscript snapshot."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/case_studies')
    args = parser.parse_args()
    copied = []
    for case in ('intro', 'toy', 'coauthorship', 'hierarchical'):
        source = args.source / case
        destination = ROOT / 'manuscript' / ('case_study_iter_core' if case == 'hierarchical' else 'numerical_output')
        for path in source.glob('*.pdf'):
            if (destination / path.name).is_file():
                shutil.copy2(path, destination / path.name)
                copied.append(str((destination / path.name).relative_to(ROOT)))
    if not copied:
        raise SystemExit('No matching case-study PDFs found; check --source.')
    print('\n'.join(copied))


if __name__ == '__main__':
    main()
