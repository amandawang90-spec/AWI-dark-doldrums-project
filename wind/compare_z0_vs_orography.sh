#!/bin/bash
# Compare ERA5 surface roughness (fsr / z0) against TCO1279 subgrid-orography
# fields (sdor, sdfor) to show they are different physical quantities despite
# both being reported in metres.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

python3 compare_z0_vs_orography.py
