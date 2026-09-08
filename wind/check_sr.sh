#!/bin/bash
# Check whether the surface roughness (sr) field in the TCO1279L137 init files is
# properly populated (spatially varying) or just a flat placeholder constant.
#
# Usage: ./check_sr.sh [gribfile]

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

GRIBFILE="${1:-/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT}"

python3 check_sr.py "$GRIBFILE"
