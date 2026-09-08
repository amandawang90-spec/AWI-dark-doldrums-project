#!/bin/bash
# ERA5 monthly means: ssrd, ssrc, tsrc (radiation) and tcwv (column water vapour).
# Usage: ./download_era5_monthly.sh YYYYMM [YYYYMM ...]

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
mkdir -p data

python3 download_era5_monthly.py "$@"
