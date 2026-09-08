#!/bin/bash
# Verify the 3-hourly ERA5 cloud/tsr downloads in solar/data.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
python3 check_downloads.py
