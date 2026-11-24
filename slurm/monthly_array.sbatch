#!/usr/bin/env bash
#SBATCH --job-name=hvfhv-clean
#SBATCH --partition=short
#SBATCH --array=1-48%4
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/array-%A_%a.out
#SBATCH --error=logs/array-%A_%a.err

set -euo pipefail

PROJECT_DIR=${SLURM_SUBMIT_DIR:?submit from the repository root}
MANIFEST="$PROJECT_DIR/configs/months_2022_2025.csv"

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
test -n "$ROW"

IFS=, read -r TASK_ID YEAR MONTH RAW_URL <<< "$ROW"

if [[ "$TASK_ID" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Manifest task ID does not match Slurm task ID" >&2
    exit 2
fi

ZONE_URL=$(sed -n 3p "$PROJECT_DIR/configs/pilot_sources.csv" | cut -d, -f5)
SEED=$((YEAR * 100 + 10#$MONTH))

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p data/interim/hpc/array
mkdir -p reports/audits/hpc/array
mkdir -p logs

JOB_TMP_DIR=$(mktemp -d "/tmp/hvfhv-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-XXXXXX")
cleanup() { rm -rf "$JOB_TMP_DIR"; }
trap cleanup EXIT

RAW_FILE="$JOB_TMP_DIR/fhvhv_tripdata_${YEAR}-${MONTH}.parquet"
ZONE_FILE="$JOB_TMP_DIR/taxi_zone_lookup.csv"
SAMPLE_OUTPUT="$PROJECT_DIR/data/interim/hpc/array/model_sample_${YEAR}-${MONTH}.parquet"
SUMMARY_OUTPUT="$PROJECT_DIR/reports/audits/hpc/array/monthly_summary_${YEAR}-${MONTH}.csv"
AUDIT_OUTPUT="$PROJECT_DIR/reports/audits/hpc/array/cleaning_audit_${YEAR}-${MONTH}.json"
RESOURCE_LOG="$PROJECT_DIR/logs/time-${YEAR}-${MONTH}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
FILE_SIZE_LOG="$PROJECT_DIR/logs/file-sizes-${YEAR}-${MONTH}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "task_id=$SLURM_ARRAY_TASK_ID"
echo "year_month=${YEAR}-${MONTH}"
echo "hostname=$(hostname)"
echo "started=$(date --iso-8601=seconds)"

curl --fail --location --silent --show-error --retry 3 --output "$RAW_FILE" "$RAW_URL"
curl --fail --location --silent --show-error --retry 3 --output "$ZONE_FILE" "$ZONE_URL"

echo "file,bytes" > "$FILE_SIZE_LOG"
echo "raw_parquet,$(stat -c %s "$RAW_FILE")" >> "$FILE_SIZE_LOG"
echo "zone_lookup,$(stat -c %s "$ZONE_FILE")" >> "$FILE_SIZE_LOG"

/usr/bin/time -v -o "$RESOURCE_LOG" python -m src.common.clean_month --input "$RAW_FILE" --zone-lookup "$ZONE_FILE" --year "$YEAR" --month "$MONTH" --sample-fraction 0.005 --seed "$SEED" --sample-output "$SAMPLE_OUTPUT" --summary-output "$SUMMARY_OUTPUT" --audit-output "$AUDIT_OUTPUT" --force

echo "model_sample,$(stat -c %s "$SAMPLE_OUTPUT")" >> "$FILE_SIZE_LOG"
echo "monthly_summary,$(stat -c %s "$SUMMARY_OUTPUT")" >> "$FILE_SIZE_LOG"
echo "cleaning_audit,$(stat -c %s "$AUDIT_OUTPUT")" >> "$FILE_SIZE_LOG"

sha256sum "$SAMPLE_OUTPUT"
sha256sum "$SUMMARY_OUTPUT"

cleanup
trap - EXIT
test ! -e "$JOB_TMP_DIR"

echo "temporary_raw_cleanup=PASS"
echo "finished=$(date --iso-8601=seconds)"
