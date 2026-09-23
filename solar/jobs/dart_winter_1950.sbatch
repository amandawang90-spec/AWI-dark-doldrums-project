#!/bin/bash
#SBATCH --job-name=dart_winter_1950
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/work/ab0995/a270321/AWI-dark-doldrums-project/solar/logs/dart_winter_1950_%j.log
set -euo pipefail
export PATH=/sw/spack-levante/mambaforge-4.11.0-0-Linux-x86_64-sobz6z/bin:$PATH
export OMP_NUM_THREADS=64
cd /work/ab0995/a270321/AWI-dark-doldrums-project/solar
python3 -u scripts/evaluation/dart_winter_eval.py /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C \
    195010 195011 195012 195101 195102 --out data/dart_eval_1950_winter1
