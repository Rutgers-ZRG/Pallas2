#!/bin/bash
# Si + NaCl benchmark matrix: 3 seeds each, MatterSim, real pressure.
set -uo pipefail
cd /Users/li/dev/Pallas2
mkdir -p runs

run_one () {
  local sys=$1 press=$2 seed=$3
  local wd=runs/${sys}-s${seed}
  echo "=== $sys seed=$seed P=${press} GPa ==="
  conda run -n reform pallas run \
    benchmarks/$sys/POSCAR_A benchmarks/$sys/POSCAR_B \
    --calc mattersim --pressure "$press" --probes 5 --max-gen 30 \
    --patience 5 --seed "$seed" --natx 100 --n-workers 4 \
    --workdir "$wd" 2>&1 | tail -2
  cp "$wd/summary.json" benchmarks/$sys/result_s${seed}.json 2>/dev/null || true
}

for seed in 42 43 44; do run_one si 12 "$seed"; done
for seed in 42 43 44; do run_one nacl 30 "$seed"; done
echo "MATRIX DONE"
