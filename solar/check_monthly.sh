#!/bin/bash
# Verify the ERA5 monthly-mean downloads in solar/data.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
python3 check_monthly.py
