#!/bin/bash
#SBATCH --job-name=dart_100m_2080c
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --array=0-64%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270321/AWI-dark-doldrums-project/wind/logs/dart_reconstruct_2080c_%A_%a.log

# 65 tasks = Oct/Nov/Dec/Jan/Feb calendar-month files across DART years 2080-2092
# (5 months x 13 years). Same method/tuning as the 1950C array job (27619193):
# chunk=7, %20 throttle matching the account's MaxJobs limit.
set -uo pipefail
cd /work/ab0995/a270321/AWI-dark-doldrums-project/wind
module load analysis-toolbox/python-04.2026

MONTHS=(10 11 12 1 2)
YEAR_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
MONTH_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
YEAR=$(( 2080 + YEAR_IDX ))
MONTH=${MONTHS[$MONTH_IDX]}

OUTFILE="data/results/dart_2080c_100m/u100v100_${YEAR}$(printf '%02d' $MONTH).nc"
if [ -s "$OUTFILE" ]; then
  echo "task $SLURM_ARRAY_TASK_ID: $OUTFILE already exists, skipping"
  exit 0
fi

echo "task $SLURM_ARRAY_TASK_ID -> YEAR=$YEAR MONTH=$MONTH"
python3 scripts/methods/dart_loglaw/reconstruct_month.py $YEAR $MONTH --run 2080c --chunk 7
