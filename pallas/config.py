"""PALLAS configuration."""

from dataclasses import dataclass, field

from ase.units import GPa

# Self-audit guards (D6/D7, 2026-08-01 zero-barrier lessons).
# A candidate saddle whose fingerprint AND energy match a known MINIMUM is
# that minimum, not a saddle. Run dist_threshold can be coarse (0.1 on
# CdSe) — the identity test is capped at a tight absolute scale so real
# saddles near minima are not over-pruned.
SADDLE_MIN_IDENTITY_DFP = 5e-3
SADDLE_MIN_IDENTITY_EDIFF = 5e-3   # eV
# Refinement may lower a saddle below a flanking minimum; such an edge is
# no longer a valid adjacency (kinetic minimax would read it as barrier 0).
# Sub-tolerance violations are noise-flat connections and are kept.
EDGE_INVARIANT_TOL = 1e-3          # eV


@dataclass
class PallasConfig:
    """Configuration for PALLAS pathway search."""
    # Fingerprint parameters
    fpcutoff: float = 5.5
    natx: int = 200
    lmax: int = 0       # 0 = s-only, 1 = s+p

    # System
    znucl: list = field(default_factory=list)   # atomic numbers in type order
    press: float = 0.0                          # external pressure (eV/A^3)
    pressure_gpa: float = None                  # external pressure (GPa); converted to press

    # Search step cap (run_fp_guided chain growing)
    maxstep: int = 50

    # Optimization step limits
    opt_steps: int = 2000           # max FIRE steps for local optimization
    opt_fmax: float = 0.001         # force convergence for optimization
    saddle_steps: int = 2000        # max FIRE steps for dimer saddle search
    saddle_fmax: float = 0.01       # force convergence for saddle

    # FP-guided search parameters
    fp_step_scale: float = 0.05     # perturbation scale along FP gradient (small to stay near basin)
    fp_push_scale: float = 0.05     # post-saddle push scale toward target
    max_retries: int = 2            # retries with smaller step on saddle failure
    max_saddle_rise: float = 20.0   # reject saddles more than this above the source (eV/cell)

    # Barrier refinement parameters
    refine_rounds: int = 3          # number of refinement iterations
    refine_probes: int = 5          # saddle searches per refinement round

    # Convergence
    ediff: float = 0.005            # energy diff threshold, whole cell (eV)
    dist_threshold: float = 0.005   # FP distance threshold for same structure

    # Generational search parameters (used by run())
    n_probes: int = 5               # probes per generation
    n_workers: int = 1              # parallel probe workers (1 = serial;
                                    # >1 needs a picklable CPU calculator)
    probe_alloc: str = 'adaptive'   # frontier-tip allocation: 'adaptive' | 'round_robin'
    max_gen: int = 50               # max generations
    patience: int = 5               # stop after N gens without barrier improvement
    min_barrier_change: float = 0.001  # minimum improvement to reset patience (eV)

    def __post_init__(self):
        if self.pressure_gpa is not None:
            if self.press != 0.0:
                raise ValueError("Give press (eV/A^3) or pressure_gpa, not both")
            self.press = self.pressure_gpa * GPa  # ase.units.GPa = eV/A^3 per GPa
