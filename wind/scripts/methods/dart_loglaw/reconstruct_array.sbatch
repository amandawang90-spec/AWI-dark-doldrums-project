#!/bin/bash
#SBATCH --job-name=dart_100m
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --array=0-99%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270321/AWI-dark-doldrums-project/wind/logs/dart_reconstruct_%A_%a.log

# 100 tasks = all Oct/Nov/Dec/Jan/Feb calendar-month files across DART years 1950-1969
# (5 months x 20 years). Each task is fully independent -- no winter-season grouping,
# just the plain calendar year/month embedded in the DART file's own name.
set -uo pipefail
cd /work/ab0995/a270321/AWI-dark-doldrums-project/wind
module load analysis-toolbox/python-04.2026

MONTHS=(10 11 12 1 2)
YEAR_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
MONTH_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
YEAR=$(( 1950 + YEAR_IDX ))
MONTH=${MONTHS[$MONTH_IDX]}

OUTFILE="data/results/dart_1950c_100m/u100v100_${YEAR}$(printf '%02d' $MONTH).nc"
if [ -s "$OUTFILE" ]; then
  echo "task $SLURM_ARRAY_TASK_ID: $OUTFILE already exists, skipping"
  exit 0
fi

echo "task $SLURM_ARRAY_TASK_ID -> YEAR=$YEAR MONTH=$MONTH"
python3 scripts/methods/dart_loglaw/reconstruct_month.py $YEAR $MONTH --chunk 7
