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
pip install -e .   # when setup.py/pyproject.toml is added
```

For now, add the project root to your Python path:
```bash
export PYTHONPATH=/path/to/Pallas2:$PYTHONPATH
```

## Quick Start

```python
from ase.io import read
from pallas import Pallas, PallasConfig

# Configure
config = PallasConfig(
    znucl=[6],          # carbon
    press=10.0,         # external pressure in eV/A^3
    maxstep=20,         # max search iterations
    opt_fmax=0.001,     # force convergence for local optimization
    saddle_fmax=0.01,   # force convergence for dimer
)

# Initialize with endpoint structures (VASP POSCAR format)
pallas = Pallas(config)
pallas.init_run(['POSCAR_graphite', 'POSCAR_diamond'])

# Run search
graph = pallas.run_fp_guided(n_probes=3)

# Find the minimum-barrier path
path, barrier = pallas.find_best_path()
```

## Recommended Pipeline

```python
# 1. Multi-probe FP-guided search (primary discovery)
graph = pallas.run_fp_guided(n_probes=3)

# 2. Iterative barrier refinement (attack the bottleneck)
best_path, best_barrier = pallas.refine_barrier(n_rounds=3, n_probes=5)

# 3. Validate saddle points (connectivity check)
stats = pallas.validate_graph()

# 4. Extract optimal pathway
path, barrier = pallas.find_best_path()
```

## Search Methods

### `run_fp_guided(n_probes=1)` — Recommended

Bidirectional FP-gradient-guided chain-growing search. At each step:

1. **FP-drag**: Walks from the current chain tip toward the target in fingerprint space, with perpendicular PES relaxation, to locate the approximate saddle region (energy maximum along the path)
2. **Dimer refinement**: Runs the solid-state dimer method at the approximate saddle to find the true transition state
3. **Saddle escape**: Pushes along the dimer's unstable mode toward the target, then relaxes to the next minimum

Multiple probes per step use varying FP/random mixing ratios to explore diverse pathways. All discovered structures feed into a single graph.

### `run_pso()` — Legacy

Bidirectional particle swarm optimization. Particles on both sides are driven toward each other, with saddle searches at each step. Less directed than `run_fp_guided` but can discover unexpected intermediates.

### `refine_barrier(n_rounds, n_probes)`

Iteratively improves the current best path by:
1. Finding the highest-energy saddle (bottleneck) on the minimax path
2. Launching targeted searches between the flanking minima
3. Re-evaluating the best path after each round

### `validate_graph()`

Tests each saddle point by pushing along +/- dimer mode and relaxing to verify that the saddle connects two distinct minima. Invalid saddles are pruned from the graph.

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
| `znucl` | `[]` | Atomic numbers in type order (e.g., `[6]` for C, `[11, 17]` for NaCl) |
| `press` | `0.0` | External pressure (eV/A^3) |
| `fpcutoff` | `5.5` | Fingerprint cutoff radius (A) |
| `natx` | `200` | Max neighbors per atom (fingerprint dimension) |
| `maxstep` | `50` | Max search iterations |
| `popsize` | `10` | PSO population size |
| `opt_steps` | `2000` | Max FIRE steps for local optimization |
| `opt_fmax` | `0.001` | Force convergence for optimization (eV/A) |
| `saddle_steps` | `2000` | Max steps for dimer saddle search |
| `saddle_fmax` | `0.01` | Force convergence for saddle (eV/A) |
| `bias_steps` | `60` | Max steps for FP bias relaxation |
| `fp_step_scale` | `0.05` | Perturbation scale along FP gradient |
| `fp_push_scale` | `0.05` | Post-saddle push scale toward target |
| `refine_rounds` | `3` | Barrier refinement iterations |
| `refine_probes` | `5` | Saddle searches per refinement round |
| `ediff` | `0.001` | Energy threshold for structure deduplication (eV) |
| `dist_threshold` | `0.01` | FP distance threshold for connection |

## Project Structure

```
Pallas2/
├── pallas/                # core package
│   ├── __init__.py        # public API
│   ├── core.py            # Pallas, PallasConfig, PallasAtom
│   ├── dimer.py           # SolidStateDimer, QuickMin optimizer
│   ├── xcal.py            # XCalculator (FP distance with autograd forces/stress)
│   ├── optimize.py        # local_optimization, cal_saddle, set_calculator
│   ├── graph.py           # minimax_path, minimax_barrier (Kruskal + BFS)
│   ├── nequip_calc.py     # NequIPDualCalc (dual-model E+F / stress)
│   ├── rcovdata.py        # covalent radii data
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
