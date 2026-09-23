#!/bin/bash
# Verify the ERA5 download: completeness, structure, and unit conventions.
# Exits non-zero if anything is wrong.
#
#   ./check_era5.sh                 # everything in data/
#   ./check_era5.sh 202508          # one month

set -euo pipefail
cd "$(dirname "$0")"

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1 || true
command -v python3 >/dev/null || export PATH=/sw/spack-levante/mambaforge-4.11.0-0-Linux-x86_64-sobz6z/bin:$PATH

python3 check_era5.py "$@"
