#!/bin/bash
# 3-hourly ERA5 cloud cover (tcc/hcc/mcc/lcc) + top net solar radiation (tsr).
# Usage: ./download_era5_clouds_3h.sh YYYYMM [YYYYMM ...]

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
mkdir -p data

python3 download_era5_clouds_3h.py "$@"
