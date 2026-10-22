#!/usr/bin/env bash
#SBATCH --job-name=spar-jmlr
#SBATCH --partition=mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-11,18-23%3
#SBATCH --output=jobs/logs/%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "${SPAR_ENV:-leafstrip}"
fi
PYTHON="${SPAR_PYTHON:-python}"
"$PYTHON" -c 'import numpy, pandas, scipy, networkx, igraph, matplotlib, tqdm'

index="${SLURM_ARRAY_TASK_ID:?Missing array index}"
alphas=(-0.5 0 2)
# Finish lighter sweeps before filling the array with long Monte Carlo jobs.
phases=(distance benchmarks calibration montecarlo)
phase="${phases[$((index / 6))]}"
local_index=$((index % 6))
scenario=balanced
if (( local_index >= 3 )); then scenario=unbalanced; fi
alpha="${alphas[$((local_index % 3))]}"
extra=(--resume)
if [[ "$phase" == calibration ]]; then extra=(); fi
"$PYTHON" experiments.py --scenario "$scenario" --alpha "$alpha" --phase "$phase" \
    --num-cores "${SLURM_CPUS_PER_TASK}" "${extra[@]}"
