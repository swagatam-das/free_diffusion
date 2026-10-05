#!/usr/bin/env bash
# Runs the self-contained checks that experiments/ does not cover; logs go to logs/ (overwritten).
# Usage:  bash reproduce/run_all.sh                  (about a minute)
#         bash reproduce/run_all.sh --with-dynamics  (also the S&P 500 matrix-diffusion run, ~1-2 min more)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 exp_gap_closing.py | tee logs/gap_closing.txt
if [ "${1:-}" = "--with-dynamics" ]; then
  python3 exp_realdata_sp500.py --dynamics | tee logs/realdata_sp500_with_dynamics.txt
else
  python3 exp_realdata_sp500.py            | tee logs/realdata_sp500.txt
fi
