# TS-core numerics review — deferred findings (2026-07-31)

Full-code review of the transition-state core (`dimer.py`, `probes.py`,
`optimize.py`, `search.py` validation/refinement). Verified correct against
TSASE and the v1.0.1 protocol: dimer rotation algebra (curvature, phi_1,
Fourier fit, force extrapolation, CG restart), pressure threading
(−V(σ+P·I) generalized cell force, `scalar_pressure`, H=E+PV in eV),
escalating ±push validation, guarded stored-mode refinement.

**Fixed on main (numerics-neutral):** `refine_path_saddles` now persists the
graph (`_save_state()`) so `graph.pkl` matches `summary.json`; `__getattr__`
recursion guard; FIRE-path convergence criterion aligned with quickmin;
QuickMin log energy now pre-step (consistent with logged fmax); single-pass
DB dedup/dij; docstring + shadowing cleanups.

**Deferred below: each changes search numerics → needs PI sign-off and a
re-benchmark of the four validated systems before merging. v1.0.1 numbers
frozen for the paper.**

## D1 — QuickMin converges on pre-step forces, returns post-step geometry

`dimer.py` `search()`: order is evaluate forces → take step → check
convergence with the *pre-step* forces → break. The returned saddle sits one
momentum step (≤ `max_step` = 0.2 Å) past the point that satisfied fmax, and
its energy/mode/curvature are quoted there — at a geometry whose forces were
never checked. Error O(few meV) at fmax=0.01; `refine_path_saddles` re-runs
the same loop, so the off-by-one is in final numbers too.

**Fix:** check convergence before stepping (evaluate → check → break → step).
Same bucket as the QuickMin trust-region item.

## D2 — Re-kick breaks the lower-triangular gauge → stored modes rotate out of frame

`dimer.py` stuck-at-minimum re-kick uses a fully random vector: cell block
not LT-constrained, atom 0 not zeroed (unlike `gen_random_mode` /
`cal_saddle` random init). The Voigt→3×3 stress mapping fills only the lower
triangle (assumes LT cell), so a kick injects upper-triangular strain the
dynamics never remove. The final `lower_triangular_cell()` in `cal_saddle`
then applies a real rotation — but `dimer_mode` is captured pre-rotation, so
the stored mode is misaligned with the stored structure (~1°/kick,
accumulating). That mode drives `_validate_saddle` ±pushes, `saddle_escape`,
and stored-mode refinement.

**Fix:** constrain the kick (`kick[0]=0`, `kick[-3,1:]=0`, `kick[-2,2]=0`),
or rotate the stored mode with the cell (positions block @ R, strain block
Rᵀ M R).

## D3 — FP-seeding cell weight is jacob² off the dimer metric (decision needed)

`fp_gradient_mode` sets the cell block to `−jacob·V·σ_fp` — steepest descent
in *strain* space expressed in generalized coordinates. The dimer's own
generalized force convention (`dimer.py` `_calculate_general_forces`) is
`−V·σ/jacob` — descent in the generalized metric the jacobian construction
exists to balance. Relative cell-vs-positions weight differs by jacob²
(≈74 for a 16-atom carbon cell): FP seeding is strongly cell-biased relative
to dimer dynamics, and the same vector sets the escape sign in
`saddle_escape`. Heuristic only — converged energies unaffected — but either
an intentional bias toward cell-driven transitions (then document it) or an
oversight (then divide by jacob² and re-benchmark search efficiency).

## D4 — Unconverged/positive-curvature dimer results enter the graph as saddles (open, minimal fix pending)

`fp_guided_saddle` falls back to the best-curvature attempt without requiring
κ<0 or convergence; `probe_compute` gates only on finite energy, volume
ratio, and rise cap; the `converged` flag is not persisted to the DB.
±push connectivity can pass a non-first-order ridge point, and when
refinement rejects, the loose energy stays on the path with only a flag.
Minimal numerics-neutral fix (not yet applied): persist `converged` in
`_update_saddle`, report unconverged-saddle counts in the summary.
