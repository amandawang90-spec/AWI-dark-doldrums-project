#!/bin/bash
# Determine whether monthly-mean accumulated radiation is per hour or per day.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

cd "$(dirname "$0")"
python3 check_monthly_units.py
