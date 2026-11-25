#!/usr/bin/env bash
#SBATCH --job-name=hvfhv-group-b-models
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=${SLURM_SUBMIT_DIR:?submit from the repository root}
SHARED_SAMPLE_DIR=/workspace/yli2955/dscp_nyc_taxi_shared/v1/samples

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

MPLCONFIG_DIR=$(mktemp -d "/tmp/hvfhv-matplotlib-${SLURM_JOB_ID}-XXXXXX")
export MPLCONFIGDIR="$MPLCONFIG_DIR"

cleanup() {
    rm -rf "$MPLCONFIG_DIR"
}
trap cleanup EXIT

COMPARISON_OUTPUT=reports/group_b/b2/candidate_comparison_full.json
CALIBRATION_OUTPUT=reports/group_b/b2/calibration_2025_full.csv
FIGURE_DIR=reports/group_b/figures/full
RESOURCE_LOG=logs/time-group-b-b2-${SLURM_JOB_ID}.txt

mkdir -p logs reports/group_b/b2 "$FIGURE_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
python --version

/usr/bin/time -v -o "$RESOURCE_LOG" \
    python -m src.group_b.model_candidates \
    --sample-dir "$SHARED_SAMPLE_DIR" \
    --config configs/group_b/model_spec_v1.json \
    --comparison-output "$COMPARISON_OUTPUT" \
    --calibration-output "$CALIBRATION_OUTPUT" \
    --figure-dir "$FIGURE_DIR"

python scripts/group_b/audit_b2_acceptance.py \
    --comparison "$COMPARISON_OUTPUT" \
    --calibration "$CALIBRATION_OUTPUT" \
    --figure-dir "$FIGURE_DIR" \
    --expected-mode FULL

echo "comparison_sha256=$(sha256sum "$COMPARISON_OUTPUT" | cut -d ' ' -f 1)"
echo "calibration_sha256=$(sha256sum "$CALIBRATION_OUTPUT" | cut -d ' ' -f 1)"
echo "finished=$(date --iso-8601=seconds)"
