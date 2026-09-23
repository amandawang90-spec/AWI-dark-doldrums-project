#!/bin/bash
# List the real time coverage and variables of every ERA5 file in wind/data.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
python3 check_time_coverage.py
