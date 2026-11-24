#!/usr/bin/env bash
#SBATCH --job-name=hvfhv-pilot-2024-01
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=${SLURM_SUBMIT_DIR:?submit from the repository root}
YEAR=2024
MONTH=01
SEED=202401

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

JOB_TMP_DIR=$(mktemp -d "/tmp/hvfhv-${SLURM_JOB_ID}-XXXXXX")
cleanup() { rm -rf "$JOB_TMP_DIR"; }
trap cleanup EXIT

RAW_FILE="$JOB_TMP_DIR/fhvhv_tripdata_${YEAR}-${MONTH}.parquet"
ZONE_FILE="$JOB_TMP_DIR/taxi_zone_lookup.csv"
SAMPLE_OUTPUT="$PROJECT_DIR/data/interim/hpc/model_sample_${YEAR}-${MONTH}.parquet"
SUMMARY_OUTPUT="$PROJECT_DIR/reports/audits/hpc/monthly_summary_${YEAR}-${MONTH}.csv"
AUDIT_OUTPUT="$PROJECT_DIR/reports/audits/hpc/cleaning_audit_${YEAR}-${MONTH}.json"
RESOURCE_LOG="$PROJECT_DIR/logs/time-${YEAR}-${MONTH}-${SLURM_JOB_ID}.txt"
FILE_SIZE_LOG="$PROJECT_DIR/logs/file-sizes-${YEAR}-${MONTH}-${SLURM_JOB_ID}.csv"

RAW_URL=$(sed -n 2p "$PROJECT_DIR/configs/pilot_sources.csv" | cut -d, -f5)
ZONE_URL=$(sed -n 3p "$PROJECT_DIR/configs/pilot_sources.csv" | cut -d, -f5)

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "raw_source=$RAW_URL"
echo "zone_source=$ZONE_URL"
python --version
command -v curl
test -x /usr/bin/time

curl --fail --location --silent --show-error --retry 3 --output "$RAW_FILE" "$RAW_URL"
curl --fail --location --silent --show-error --retry 3 --output "$ZONE_FILE" "$ZONE_URL"

echo "file,bytes" > "$FILE_SIZE_LOG"
stat -c "%n,%s" "$RAW_FILE" "$ZONE_FILE" >> "$FILE_SIZE_LOG"

/usr/bin/time -v -o "$RESOURCE_LOG" python -m src.common.clean_month --input "$RAW_FILE" --zone-lookup "$ZONE_FILE" --year "$YEAR" --month "$MONTH" --sample-fraction 0.005 --seed "$SEED" --sample-output "$SAMPLE_OUTPUT" --summary-output "$SUMMARY_OUTPUT" --audit-output "$AUDIT_OUTPUT" --force

stat -c "%n,%s" "$SAMPLE_OUTPUT" "$SUMMARY_OUTPUT" "$AUDIT_OUTPUT" >> "$FILE_SIZE_LOG"

sha256sum "$SAMPLE_OUTPUT"
sha256sum "$SUMMARY_OUTPUT"

cleanup
trap - EXIT
test ! -e "$JOB_TMP_DIR"
echo "temporary_raw_cleanup=PASS"
echo "finished=$(date --iso-8601=seconds)"
