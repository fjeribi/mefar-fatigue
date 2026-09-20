#!/usr/bin/env bash
# Reproduce every analysis in the paper.
# Usage: bash run_all.sh /path/to/MEFAR_raw_root /path/to/MEFAR_DOWN.csv [results_dir]
set -euo pipefail
RAW_ROOT="$1"; RELEASE="$2"; OUT="${3:-results}"

python scripts/01_protocol1_sample_level.py --release "$RELEASE" --out-dir "$OUT"
python scripts/02_reconstruct_subjectwise.py --raw-root "$RAW_ROOT" --out-dir "$OUT"
python scripts/03_protocol2_conventional.py --out-dir "$OUT"
python scripts/04_protocol2_deep.py        --out-dir "$OUT"
python scripts/05_session_proxy.py         --out-dir "$OUT"
python scripts/06_ablation_statistics.py   --out-dir "$OUT"
python scripts/07_gating_and_masking.py    --out-dir "$OUT"
python scripts/08_partitioning_control.py  --out-dir "$OUT"
python scripts/09_descriptives.py          --out-dir "$OUT" --release "$RELEASE" --raw-root "$RAW_ROOT"
python scripts/10_uncertainty.py           --out-dir "$OUT"
python scripts/11_windowed_features.py     --raw-root "$RAW_ROOT" --out-dir "$OUT"
echo "All analyses complete. Results in $OUT/"
