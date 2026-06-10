# PALLAS

**P**h**A**se Transition **L**andscape Exp**L**oration with **A**utomated **S**addle Search

PALLAS is an automated method for discovering transition pathways between crystal phases. Given two endpoint structures (e.g., graphite and diamond), it builds a graph of minima and saddle points on the potential energy surface, then extracts the lowest-barrier pathway via minimax optimization.

![image](https://github.com/user-attachments/assets/259c32c9-34e4-40aa-9f62-297d10d6d92d)

## Key Features

- **Fingerprint-guided search**: Uses GOM structural fingerprints ([torch_fplib](https://github.com/Rutgers-ZRG/torch-fplib)) to measure structural similarity and compute analytical gradients (forces + stress) via PyTorch autograd
- **Solid-state dimer method**: Finds transition states without computing the full Hessian, with variable cell shape optimization for periodic systems
- **QuickMin optimizer**: Momentum-based translation that maintains velocity through low-force regions, outperforming FIRE for dimer searches
- **Graph-based pathway construction**: NetworkX graph with minima as nodes and saddle points as edges; minimax (bottleneck) algorithm finds the minimum-barrier path
- **Iterative barrier refinement**: Attacks the highest saddle on the current best path with targeted multi-probe searches
- **Saddle validation**: Connectivity test (push along +/- dimer mode, verify distinct minima on both sides)
- **Calculator-agnostic**: Works with any ASE-compatible calculator (MatterSim, MACE, NequIP/Allegro, VASP, etc.)

## Installation

PALLAS requires Python 3.9+ and the following dependencies:

```
numpy
torch
torch_fplib
ase
networkx
joblib
scipy
```

Optional (for specific calculators):
- `mattersim` — MatterSim universal potential (default calculator)
- `nequip` — NequIP/Allegro models

Clone and install:
```bash
git clone https://github.com/Rutgers-ZRG/Pallas2.git
cd Pallas2
pip install -e .
```

`torch_fplib` is not yet on PyPI — install it from the
[torch-fplib](https://github.com/Rutgers-ZRG/torch-fplib) repository first
(`pip install /path/to/torch-fplib`).

## Command line

```bash
# full search with the default MatterSim calculator
pallas run POSCAR_graphite POSCAR_diamond --pressure 15 --probes 5 --workdir out/

# results: out/summary.json, out/profile.png, out/pathway.extxyz, out/config.json
pallas plot out/      # regenerate the enthalpy profile
pallas export out/    # write pathway POSCARs
```

`--calc emt` (testing), `--calc nequip --model-ef ... --model-stress ...`
(custom MLIPs), `--n-workers N` (parallel probes on CPU calculators),
`--k-paths K` (report the K best bottleneck-diverse pathways).

## Quick Start

```python
from pallas import Pallas, PallasConfig

# Configure
config = PallasConfig(
    znucl=[6],          # carbon
    pressure_gpa=10.0,  # external pressure in GPa
    n_probes=5,         # probes per generation
    max_gen=50,         # max generations
    patience=5,         # stop after N gens without improvement
)

# Initialize with endpoint structures (VASP POSCAR format)
pallas = Pallas(config)
pallas.init_run(['POSCAR_graphite', 'POSCAR_diamond'])

# Run unified search (connect + refine + converge)
path, barrier = pallas.run()
```

## How `run()` Works

The unified search loop automatically switches between two modes:

```
for each generation:
    if no A→B path exists in graph:
        CONNECT — launch probes from frontier tips toward the other side
    else:
        REFINE  — launch probes to attack the bottleneck saddle

    register all new minima/saddles in graph
    re-evaluate minimax path

    if no improvement for `patience` generations:
        stop
```

**CONNECT mode**: Probes grow chains from both sides (A→ and ←B).
Multiple frontier tips are ranked by FP distance to the other side,
and probes are distributed round-robin across the best tips.
This naturally handles paths with many intermediates (A → I1 → I2 → ... → B).

**REFINE mode**: Identifies the highest-energy saddle (bottleneck) on
the current best path, finds its flanking minima, and launches probes
from both sides of the bottleneck toward each other.  All new structures
enter the graph; minimax re-evaluation may find a lower-barrier route.

## Optional Post-Processing

After `run()` completes, you can optionally validate saddle points:

```python
# Validate saddle connectivity (push ±mode → distinct minima)
stats = pallas.validate_graph()

# Re-extract path after validation
path, barrier = pallas.find_best_path()
```

## Search Methods

### `run()` — Recommended

Unified generational search that combines connection and refinement in one loop. See "How `run()` Works" above.

Each probe within a generation does:

1. **FP-gradient mode**: Computes the fingerprint-distance gradient toward the target (mixed with a random component by the probe's alpha) and perturbs the minimum along it
2. **Dimer search**: Runs the solid-state dimer from the perturbed structure with the FP-gradient initial mode to find the transition state
3. **Saddle escape**: Pushes along the dimer's unstable mode toward the target, then relaxes to the next minimum

Probes use varying FP-gradient / random mixing ratios (alpha schedule) for diversity within each generation.

### `run_fp_guided(n_probes=1)` — Advanced

Single-pass FP-gradient-guided chain growing without generational convergence. Useful for quick exploration or debugging.

### `validate_graph()`

Post-processing: tests each saddle by pushing along +/- dimer mode and relaxing, verifying the saddle connects two distinct minima. Prunes invalid saddles.

## Using a Custom Calculator

By default, PALLAS uses MatterSim (lazily loaded on first use). To use a different ASE calculator:

```python
from pallas import set_calculator

# Example: NequIP dual-model calculator
from pallas import NequIPDualCalc
calc = NequIPDualCalc(
    ef_model_path='model_energy_force.pth',
    stress_model_path='model_stress.pth',
)
set_calculator(calc)

# Or any ASE calculator
from mace.calculators import mace_mp
set_calculator(mace_mp(default_dtype='float64'))
```

## Configuration

`PallasConfig` is a dataclass with all tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **System** | | |
| `znucl` | `[]` | Atomic numbers in type order (e.g., `[6]` for C, `[11, 17]` for NaCl) |
| `pressure_gpa` | `None` | External pressure (GPa) — preferred input |
| `press` | `0.0` | External pressure (eV/A^3, internal unit) |
| **Fingerprint** | | |
| `fpcutoff` | `5.5` | Fingerprint cutoff radius (A) |
| `natx` | `200` | Max neighbors per atom (fingerprint dimension) |
| **Search (`run()`)** | | |
| `n_probes` | `5` | Probes per generation |
| `n_workers` | `1` | Parallel probe workers (CPU calculators only) |
| `probe_alloc` | `'adaptive'` | Frontier-tip allocation: `adaptive` or `round_robin` |
| `max_gen` | `50` | Max generations |
| `patience` | `5` | Stop after N gens without barrier improvement |
| `min_barrier_change` | `0.001` | Minimum improvement to reset patience (eV) |
| **Optimization** | | |
| `opt_steps` | `2000` | Max FIRE steps for local optimization |
| `opt_fmax` | `0.001` | Force convergence for optimization (eV/A) |
| `saddle_steps` | `2000` | Max steps for dimer saddle search |
| `saddle_fmax` | `0.01` | Force convergence for saddle (eV/A) |
| **FP-guided parameters** | | |
| `fp_step_scale` | `0.05` | Perturbation scale along FP gradient |
| `fp_push_scale` | `0.05` | Post-saddle push scale toward target |
| **Convergence** | | |
| `ediff` | `0.001` | Energy threshold for structure deduplication (eV) |
| `dist_threshold` | `0.01` | FP distance threshold for connection |

## Project Structure

```
Pallas2/
├── pallas/                # core package
│   ├── __init__.py        # public API
│   ├── config.py          # PallasConfig (GPa-aware pressure input)
│   ├── search.py          # Pallas engine: generational connect/refine loop
│   ├── probes.py          # FP-guided probe steps (parallel-safe compute)
│   ├── analysis.py        # reporting, trajectories, listpath
│   ├── structure.py       # PallasAtom, fp_distance, spacegroup_label, enthalpy
│   ├── dimer.py           # SolidStateDimer, QuickMin optimizer
│   ├── xcal.py            # XCalculator (FP distance with autograd forces/stress)
│   ├── optimize.py        # local_optimization, cal_saddle (pressure-aware)
│   ├── graph.py           # minimax, kinetic minimax, k-best paths
│   ├── cli.py             # pallas run/plot/export command line
│   ├── plotting.py        # enthalpy profile figures
│   ├── nequip_calc.py     # NequIPDualCalc (dual-model E+F / stress)
│   ├── rcovdata.py        # covalent radii data
│   ├── core.py            # back-compat shim
│   └── utils.py           # asedb2vasp helper
├── tests/
│   └── cdse/              # CdSe RS->WZ benchmark scripts and NequIP models
├── LICENSE
└── README.md
```

## How It Works

### Structural Fingerprints

PALLAS uses GOM (Gaussian Overlap Matrix) fingerprints from `torch_fplib` as a structural distance metric. Each atom gets a fingerprint vector encoding its local chemical environment. The distance between two structures is computed via Hungarian matching (optimal atom assignment) of per-atom fingerprints.

The `XCalculator` computes analytical forces and stress of the FP distance using PyTorch autograd with a strain parametrization, enabling gradient-driven structural steering in fingerprint space.

### Solid-State Dimer Method

The dimer method finds transition states by following the lowest curvature mode on the PES without computing the Hessian. PALLAS extends this to periodic systems with variable cell shape:

- **Rotation**: Conjugate gradient optimization of the dimer orientation to align with the minimum curvature mode
- **Translation**: QuickMin (velocity Verlet with momentum reset) moves the dimer uphill along the unstable mode and downhill along all other modes
- **Cell DOFs**: Lattice vectors are included as additional degrees of freedom, weighted by a Jacobian factor

### Graph Construction

All discovered minima and saddle points are stored in an ASE database and registered as nodes in a NetworkX graph. Edges connect saddles to their flanking minima, weighted by the maximum energy along the edge. The minimax algorithm (Kruskal's MST with early stopping) finds the path that minimizes the maximum energy barrier.

## Analysis

After a search completes, analyze the results:

```python
from pallas.core import listpath

# Load saved graph and print the best path with POSCAR output
listpath('graph.pkl', 'pallas.db', start=1, end=2)
# Writes POSCARs to path_output/
```

## License

MIT License. See [LICENSE](LICENSE) for details.
