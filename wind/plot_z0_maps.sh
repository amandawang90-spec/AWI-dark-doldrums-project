#!/bin/bash
# Map TCo1279 'sr' (surface roughness) next to a populated control field (sdor) and
# ERA5's real z0 (fsr), to show what roughness data we do and do not have.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

python3 plot_z0_maps.py
