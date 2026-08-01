#!/bin/bash
# D2-fix re-benchmark: Si + NaCl matrix, 3 seeds each, MatterSim, real pressure.
# Same protocol as run_local_matrix.sh but writes to *-d2-* workdirs and does
# NOT touch the v1.0.1 workdirs or benchmarks/*/result_s*.json.
set -uo pipefail
cd /Users/li/dev/Pallas2
mkdir -p runs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

run_one () {
  local sys=$1 press=$2 seed=$3
  local wd=runs/${sys}-d2-s${seed}
  rm -rf "$wd"   # stale pallas.db rows would pollute init_run
  echo "=== $sys seed=$seed P=${press} GPa ==="
  conda run --live-stream -n reform pallas run \
    benchmarks/$sys/POSCAR_A benchmarks/$sys/POSCAR_B \
    --calc mattersim --pressure "$press" --probes 5 --max-gen 15 \
    --patience 4 --seed "$seed" --natx 100 --n-workers 2 \
    --workdir "$wd" 2>&1 | grep -E 'Gen |barrier|Barrier|No path|Error|error|WARNING'
  echo "--- $sys s$seed summary:"
  python3 -c "import json;s=json.load(open('$wd/summary.json'));print('    barrier_ev=%s path=%s unconv=%s' % (s['barrier_ev'], s['path_names'], s.get('path_saddles_unconverged')))" 2>/dev/null || echo "    (no summary)"
}

for seed in 42 43 44; do run_one si 12 "$seed"; done
for seed in 42 43 44; do run_one nacl 30 "$seed"; done
echo "MATRIX-D2 DONE"
