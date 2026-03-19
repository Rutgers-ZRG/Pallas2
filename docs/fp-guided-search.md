# FP-Guided Transition Pathway Search — Design Document

## Date: 2026-03-18

## Problem with Current Method

The current PALLAS uses **random perturbations + random dimer modes** in a bidirectional PSO framework. This leads to:

1. **Incorrect saddle points**: Dimer finds saddles unrelated to the A→B pathway. In the CdSe RS→WZ test, saddle S6 had E = -0.414 eV — *lower* than the starting minimum RS (0.000 eV). A true transition state must be higher than adjacent minima.

2. **Blind exploration**: Random perturbations in (3N+9)-dimensional space are extremely unlikely to point toward the target structure. Most dimer searches are wasted.

3. **No directional information**: The PSO velocities try to encode direction, but they operate in configuration space without structural awareness.

## Proposed Method: FP-Gradient-Guided Search

### Key Insight

The fingerprint distance gradient ∇_r d_fp(A, B) answers: *"which direction should atoms move to get closer to B in structural space?"* This is exactly the missing information.

**Proof of concept**: `fp_drive_xdatcar.py` (ReformPy) demonstrates that FP-gradient forces successfully drive CdSe hex→sq, including both atomic positions and cell shape changes.

### Algorithm

```
path = [A]
current = A
while d_fp(current, B) > threshold:
    1. Compute F_fp = -∇_r d_fp(current, B)     [XCalculator forces]
       This gives direction in (positions + cell) space toward B

    2. Perturb current along F_fp:
       A' = current + λ · F_fp/|F_fp|            [controlled step size]

    3. Run solid-state dimer with mode₀ = F_fp direction
       → Finds saddle S on the current→B pathway  [not random saddle]

    4. Validate S:
       - E(S) > E(current)                        [energy check]
       - Curvature along dimer mode < 0            [saddle check]

    5. Small FP-guided push on S toward B:
       F_fp' = -∇_r d_fp(S, B)
       S' = S + μ · F_fp'/|F_fp'|

    6. Optimize S' on real PES (MatterSim) → new minimum M

    7. path.extend([S, M])
       current = M
```

### Advantages over NEB/String Methods

- **No interpolation needed**: FP gradient naturally guides through configuration space. Linear interpolation between crystal structures often fails (atom collisions, unphysical geometries).
- **Includes cell degrees of freedom**: XCalculator strain parametrization handles cell shape changes automatically. Variable-cell NEB is notoriously difficult.
- **One saddle at a time**: Dimer method scales as O(1) per saddle, not O(N_images). Much cheaper than NEB for long pathways.
- **Discovers intermediates naturally**: Each step finds the next minimum, then re-aims toward B. Intermediate phases (e.g., 5-coordinated states) are found automatically.

### Advantages over Random PSO (Current PALLAS)

1. **Directed**: FP gradient points toward B. Orders of magnitude more efficient than random in high-dimensional space.
2. **Correct saddles**: Dimer mode initialized along A→B direction finds the *relevant* transition state.
3. **Cell-aware**: Strain gradient deforms cell toward target shape.
4. **Iterative refinement**: Re-computes FP gradient at each minimum, naturally follows curved pathways.

### Parameters

- **λ (perturbation step size)**: Controls how far to push from the minimum before dimer search. Could be adaptive: `λ ∝ d_fp(current, B)` — larger when far, smaller when close.
- **μ (post-saddle push)**: Small fixed value, just enough to escape the saddle region and fall into the next basin.
- **Saddle validation**: Reject saddles with E(S) < E(current) or positive curvature.

### Role of PSO

With FP-guided search, PSO may be unnecessary for the primary algorithm. The greedy chain-growing approach is simpler and more directed. PSO could be kept as:
- Fallback for cases where greedy search gets stuck
- Exploration of multiple parallel pathways
- Diversity mechanism for complex energy landscapes

### Supporting Evidence

- **FP-driven optimization works**: `fp_drive_xdatcar.py` drives CdSe hex→sq to completion
- **FP distance correlates with energy**: Lipschitz analysis from CRISP shows Spearman ρ(d_fp, ΔH) = 0.71, max |ΔH| = 9.8 meV for d_fp < 0.01
- **XCalculator autograd forces are exact**: Validated to 3.88e-06 eV/A vs numerical finite difference
- **torch_fplib includes cell derivatives**: Strain parametrization gives exact Voigt stress

### Implementation Notes

All infrastructure is already in place:
- `XCalculator` (xcal.py): computes FP-distance energy, forces, stress via torch_fplib autograd
- `SolidStateDimer` (dimer.py): accepts initial mode, handles solid-state cell optimization
- `local_optimization` (zfunc.py): MatterSim FIRE relaxation
- `minimax_path` (barrier.py): efficient bottleneck path finding on the resulting graph

The main change is replacing the random PSO loop in `pallas.py` with the FP-guided chain-growing algorithm.
