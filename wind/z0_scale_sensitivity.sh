#!/bin/bash
# Quantify (a) the log-law's sensitivity to z0 and (b) how much error is introduced
# by supplying a coarse-resolution z0 field in place of a fine-resolution one --
# i.e. whether ERA5's ~25 km z0 can be applied to ~9 km TCo1279 winds.

set -euo pipefail

module load analysis-toolbox/python-04.2026 >/dev/null 2>&1

python3 z0_scale_sensitivity.py
