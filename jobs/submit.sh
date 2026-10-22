#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p jobs/logs
# Optional argument selects a subset, e.g. 0-1 for an initial simulation check.
sbatch --array="${1:-0-11,18-23%3}" jobs/run.slurm
