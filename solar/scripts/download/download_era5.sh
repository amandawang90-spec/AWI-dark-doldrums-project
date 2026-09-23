#!/bin/bash
# Download every ERA5 field needed for the solar reconstruction.
# Skips files that already exist; pass --force to re-fetch, --list to preview.
#
#   ./download_era5.sh 202508 202603 202608
#
# Needs cdsapi >= 0.7.2 (the 0.5.x legacy client fails with "'tuple' object is not
# callable") and a token in ~/.cdsapirc -- https://cds.climate.copernicus.eu/how-to-api
# Run on a LOGIN node: compute nodes have no outbound internet.

set -euo pipefail
cd "$(dirname "$0")"

# AWI albedo provides python via a module; on DKRZ levante use the spack python,
# which carries a user-installed cdsapi in ~/.local. Neither is fatal if absent.
module load analysis-toolbox/python-04.2026 >/dev/null 2>&1 || true
command -v python3 >/dev/null || export PATH=/sw/spack-levante/mambaforge-4.11.0-0-Linux-x86_64-sobz6z/bin:$PATH

python3 -u download_era5.py "$@"
