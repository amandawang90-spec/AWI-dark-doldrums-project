#!/bin/bash
# Inspect orography-related fields (sdor, sdfor, isor, anor, slor, z) in an IFS
# GRIB init file, to see which are properly populated on the TCO1279 grid.
#
# Usage: ./check_orography.sh [gribfile]

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

GRIBFILE="${1:-/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT}"

python3 check_orography.py "$GRIBFILE"
