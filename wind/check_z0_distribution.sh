#!/bin/bash
# Sanity-check the ERA5 land z0 (fsr) distribution: area-weighted percentiles and
# a breakdown into Davenport roughness classes.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

python3 check_z0_distribution.py
