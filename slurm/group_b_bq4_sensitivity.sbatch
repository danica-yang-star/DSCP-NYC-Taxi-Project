#!/usr/bin/env bash
#SBATCH --job-name=hvfhv-group-b-bq4
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

MPLCONFIG_DIR=$(mktemp -d "/tmp/hvfhv-bq4-matplotlib-${SLURM_JOB_ID}-XXXXXX")
export MPLCONFIGDIR="$MPLCONFIG_DIR"
cleanup() { rm -rf "$MPLCONFIG_DIR"; }
trap cleanup EXIT

REPORT_OUTPUT=reports/group_b/bq4/sensitivity_full.json
SCENARIOS_OUTPUT=reports/group_b/bq4/sensitivity_scenarios_full.csv
LOO_OUTPUT=reports/group_b/bq4/leave_one_year_out_full.csv
FIGURE_DIR=reports/group_b/figures/bq4/full
RESOURCE_LOG=logs/time-group-b-bq4-${SLURM_JOB_ID}.txt

mkdir -p logs reports/group_b/bq4 "$FIGURE_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
python --version

/usr/bin/time -v -o "$RESOURCE_LOG" \
    python -m src.group_b.sensitivity \
    --sample-dir "$SHARED_SAMPLE_DIR" \
    --config configs/group_b/bq4_sensitivity_v1.json \
    --report-output "$REPORT_OUTPUT" \
    --scenarios-output "$SCENARIOS_OUTPUT" \
    --leave-one-year-out-output "$LOO_OUTPUT" \
    --figure-dir "$FIGURE_DIR"

python scripts/group_b/audit_bq4_acceptance.py \
    --report "$REPORT_OUTPUT" \
    --bq2-report reports/group_b/bq2/main_effects_full.json \
    --scenarios "$SCENARIOS_OUTPUT" \
    --leave-one-year-out "$LOO_OUTPUT" \
    --figure-dir "$FIGURE_DIR" \
    --expected-mode FULL

echo "report_sha256=$(sha256sum "$REPORT_OUTPUT" | cut -d ' ' -f 1)"
echo "scenarios_sha256=$(sha256sum "$SCENARIOS_OUTPUT" | cut -d ' ' -f 1)"
echo "loo_sha256=$(sha256sum "$LOO_OUTPUT" | cut -d ' ' -f 1)"
echo "finished=$(date --iso-8601=seconds)"
