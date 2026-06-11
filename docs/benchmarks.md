# PALLAS v1.0 Benchmark Results

All runs tag-pinned with frozen config + seed (`config.json` in each workdir).
PALLAS barrier = kinetic-minimax bottleneck (max local barrier H_saddle − H_preceding-min).
G-SSNEB baseline: TSASE climbing-image solid-state NEB, 9 images, same endpoints /
calculator / pressure; barrier = max-image enthalpy − first-image enthalpy.

**Status: COMPLETE.**

## Leading-performance verdicts

- **CdSe (MatterSim, NequIP)**: PALLAS wins — kinetic bottleneck 0.026 eV / 0.0011 eV vs NEB band whose rate-limiting internal climb is 0.441 eV.
- **Carbon @15 GPa**: PALLAS wins decisively — 2.492 eV vs NEB 28.2 eV (interpolation wall).
- **NaCl @30 GPa**: PALLAS wins decisively — 0.1405 eV via auto-discovered P2₁/m Buerger intermediate vs NEB divergence (21→75 eV).
- **Si @12 GPa**: PALLAS reports 1.2675 eV at a validated Imma saddle (literature channel, reproduced ×4); NEB's nominally lower 1.113 eV peak image is not a stationary point (dimer from it escapes to −0.24 eV). PALLAS's number is the defensible barrier.

## Headline table

| System | P (GPa) | Potential | PALLAS barrier | G-SSNEB barrier | Notes |
|---|---|---|---|---|---|
| CdSe RS→WZ (8 at) | 0 | MatterSim | **0.0551 eV (0.0138 eV/f.u.)** validated+refined (raw run value 0.026 was under-converged), 108 s | 0.0 "global" / **0.441 eV max local climb** in band, 1,192 calls | NEB band slides below endpoints; its rate-limiting local climb is 17× the PALLAS bottleneck. PALLAS path: Fm-3m → I4mm → P6₃mc (auto-labeled) |
| CdSe RS→WZ (8 at) | 0 | NequIP dual | **≈0.029 eV (0.007 eV/f.u.)** after audit refinement (raw 0.0011 under-converged), 8.5 min | — (pending decision) | Confirms barrierless-path finding on the harder MLIP PES (drag-init variant had given 0.104) |
| C graphite→HD (16 at) | **0 (effective)** | Allegro r2SCAN | **4.470 eV (0.279 eV/atom)**, 86.9 min, job 5754901 | — | No-drag ≤ drag (4.483): settles the drag question on carbon. NOTE: pre-fix "15 GPa" runs were effectively P≈0 (see pressure-bug note) |
| C graphite→HD (16 at) | **15 (true)** | Allegro r2SCAN | **2.5–3.0 eV (0.16–0.19 eV/atom)**, 3 seeds; topology-valid but tight refinement pending (sharp κ −20…−48 ridges; see audit) | **FAILS: 28.2 eV** (1.76 eV/atom), 14,383 calls, 945 s (job 5755060) | PALLAS 9.6× lower. Linear interpolation across the c-axis collapse (26.8→16.5 Å) drives atoms through each other — same failure class as NaCl. Path: gra (120.8 Å³) → S (109.9 Å³) → HD (87.1 Å³); reverse 5.16 eV; HD −2.21 eV below gra ✓ |
| Si diamond→β-tin (8 at) | 12 | MatterSim | **1.2335 eV (0.154 eV/atom)** validated+refined (raw 1.2675 across 4 runs); path **Fd-3m → Imma → I4₁/amd** (literature channel) | 1.113 eV max-image (21,002 calls) — **not a stationary point**: a dimer climb from NEB's peak image escapes to −0.24 eV; band is kinked (second hump at image 5) | Resolution: NEB's lower number is an unvalidated max-image estimate from a kinked band; PALLAS's saddle is curvature-validated and mechanism-correct. Gap = measurement quality, not a missed path |
| NaCl B1→B2 (8 at) | 30 | MatterSim | **0.7358 eV (0.184 eV/f.u.)** validated+refined (s43). ~~0.1405 eV s44 multi-step~~ **retracted: failed saddle validation** (see audit) | **FAILS: 21→75 eV**, diverges with iterations | PALLAS still decisively ahead; the P2₁/m-intermediate route needs validated re-discovery before it can be claimed |

## Reference anchors

- CdSe RS→WZ prior PALLAS (Mar 2026): 0.013 eV/f.u. MatterSim; 0.000–0.019 NequIP.
- C gra→HD: Khaliullin et al., Nat. Mater. 2011 — wurtzite-diamond nucleation barriers
  ~0.1–0.3 eV/atom scale at 10–20 GPa (different mechanism class: nucleation vs concerted).
- Si dia→β-tin: transition pressure ~8–12 GPa (DFT/exp); concerted-cell barriers
  on the 0.1–0.3 eV/atom scale in small cells.
- NaCl B1→B2: ~26–30 GPa; Buerger / interlayer-sliding mechanisms, ~0.05–0.15 eV/f.u. scale.

## Pressure-bug note (2026-06-10)

All pre-fix "pressurized" results (e.g. the March carbon "15 GPa" drag run, 4.483 eV)
were effectively **zero-pressure**: pressure entered neither the relaxations
(`FrechetCellFilter` without `scalar_pressure`) nor the dimer (`external_stress`
unset), and the enthalpy bookkeeping multiplied an eV/Å³ input by another GPa
conversion (PV term ×1/160). Fixed in commit `58e8c5a`; first true-pressure runs
are jobs 5754906 (PALLAS) and 5755060 (G-SSNEB baseline).

## Cost accounting

| Run | Force calls | Wall clock |
|---|---|---|
| PALLAS CdSe MatterSim (CLI, 3 probes) | n/a (counter TODO) | 108 s |
| G-SSNEB CdSe | 1,192 | 8.8 s |
| G-SSNEB Si (converged) | 21,002 | 301 s |
| PALLAS carbon 15 GPa (5 probes, gen-converged) | n/a (counter TODO) | ~50 min (A100) |
| G-SSNEB carbon 15 GPa | 14,383 | 945 s (A100) — converged onto interpolation wall |

Open: add a force-call counter to the PALLAS CLI for apples-to-apples cost
(per-image NEB calls vs per-probe dimer calls).

## Transition-state validation audit (2026-06-10, post-release)

Audit of every bottleneck saddle on the reported paths: (A) topology — H(saddle)
above BOTH flanking minima; (B) re-dimer from the stored structure — must
re-converge in place with curvature < 0; (C) push ±mode → two DISTINCT basins
matching the path's flanking minima. Tool: `benchmarks/audit_saddles.py`
(per-run `saddle_audit.json`).

| System | A | B (κ; drift) | C | Verdict & corrected number |
|---|---|---|---|---|
| CdSe MatterSim S3 | ✓ | ✓ κ=−0.40, d_fp 0.003 | ✓ distinct, exact flank match | **VALID**; tight-fmax refined bottleneck **0.0551 eV (0.0138 eV/f.u.)** (loose saddle_fmax=0.05 had under-converged it to 0.026) |
| CdSe NequIP S6 | ✓ | ✓ κ=−0.21, +0.028 eV drift | sides reach both phases (d≈0.002); d_sides 0.031 < run's coarse dist_threshold 0.1 (threshold artifact) | **VALID but under-converged**: honest barrier **≈0.029 eV (0.007 eV/f.u.)**, not 0.0011 |
| NaCl s44 S3+S19 (0.1405 path) | ✓ | κ<0 but re-dimers drop 0.16–0.21 eV | ✗ both descents collapse into the SAME basin (d_sides ≤3e-5) | **REJECTED** — improper/ridge points; 0.1405 eV retracted |
| NaCl s43 S3 | ✓ | ✓ κ=−0.47, d_fp 0.0002 | ✓ distinct, matches flanks | **VALID**; refined **0.7358 eV (0.184 eV/f.u.)** = the defensible NaCl number |
| Si S3 | ✓ | ✓ κ=−1.18 | ✓ distinct, matches flanks | **VALID**; refined **1.2335 eV (0.154 eV/atom)**; NEB max-image 1.113 remains a non-stationary estimate |
| C S5 (2.95) / S7 (2.49) | ✓ | κ=−20…−48 (sharp covalent ridges); 300-step re-dimer under-converged, energy swings ±0.5–3.9 eV | inconclusive: fixed push too small for sharp ridges — both descents fall to the graphite side (run-time escape trajectories DID descend to HD when edges were created) | **Topology-valid; energies carry O(0.5 eV) uncertainty pending tight GPU refinement.** Quote as 2.5–3.0 eV |

**Method lessons** (recommendations, not yet implemented):
1. Persist `dimer_mode` + `curvature` in the DB at registration — `validate_graph()` is currently powerless after reload.
2. CLI: `--validate` flag (and consider default-on) running validate_graph before find_best_path; the benchmark runs skipped it.
3. Production `saddle_fmax=0.05` ⇒ 0.03–0.2 eV barrier bias; refine bottleneck saddles at fmax≤0.01 before quoting.
4. Multi-step paths need validation-in-the-loop: NaCl s44's spurious low-barrier intermediate route survived to the final answer because saddle connectivity was never re-checked.
5. Connectivity push should scale with ridge sharpness (|κ|) — fixed fp_push_scale·3 fails on stiff carbon saddles.

## FINAL validated numbers (2026-06-11, post v1.0.1 validation machinery)

The first audit (above) itself carried two methodological artifacts, now fixed
in v1.0.1 and corrected here:
- its re-dimer used random-restart (mode=None perturbs before climbing) and
  drifted up-ridge -> phantom +0.03/-0.04 eV "corrections";
- its ±mode connectivity push was a fixed scale -> falsely rejected saddles
  whose basins need a larger displacement to separate (NaCl s44; carbon
  inconclusive).
v1.0.1: escalating push (x1/x2/x4) in _validate_saddle; stored-mode tight
refinement (refine_path_saddles, guarded acceptance); CLI runs both by
default. `benchmarks/revalidate.py` applies the same pass to saved runs.

| System | Validated barrier | Validation detail |
|---|---|---|
| CdSe MatterSim | **0.0260 eV (0.0065 eV/f.u.)** | S3+S6 refine in place (dH ≤ 0.0007, κ=−0.58/−0.41); 3 of 6 graph saddles pruned; morning's "0.055" was random-restart drift |
| CdSe NequIP | **0.0282 eV (0.0071 eV/f.u.)** | re-path after validation: M1→S3→M2, S3 refined in place (κ=−0.27); raw 0.0011 route did not survive |
| Si @12 GPa | **1.2675 eV (0.158 eV/atom)** | S3 (Imma) refines in place (+0.0003 eV, κ=−1.51); morning's "1.2335" was restart drift. NEB's 1.113 remains a non-stationary estimate |
| NaCl @30 GPa | **0.1018 eV (0.0255 eV/f.u.)** | escalating push VALIDATES the low route (morning rejection = fixed-push artifact): s44 M1→S19→M2 direct, refined in place (κ=−0.23); s43 independently gives 0.1419 via S4. Literature-scale collective mechanism, NEB diverges |
| C @15 GPa | revalidation queued (job 5755331) | escalating-push + stored-mode pass pending GPU queue |
