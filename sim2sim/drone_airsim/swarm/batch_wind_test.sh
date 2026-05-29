#!/usr/bin/env bash
set -euo pipefail

runs="${1:-10}"
shift || true

for seed in $(seq 0 $((runs - 1))); do
  python eval.py --resume swarm.pth --target_speed 2.5 --seed "${seed}" \
    --trace_policy --trace_stride 1 \
    --use_wind --wind_mode mixed \
    --wind_mean_range -2 2 \
    --wind_gust_range -3 3 \
    --wind_vertical_range 0 1.5 \
    --wind_side_range 0 12 \
    --wind_update_interval 8 \
    --wind_randomize_prob 0.85 \
    "$@"
done
