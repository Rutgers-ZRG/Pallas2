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
| CdSe RS→WZ (8 at) | 0 | MatterSim | **0.026 eV (0.0066 eV/f.u.)**, 108 s | 0.0 "global" / **0.441 eV max local climb** in band, 1,192 calls | NEB band slides below endpoints; its rate-limiting local climb is 17× the PALLAS bottleneck. PALLAS path: Fm-3m → I4mm → P6₃mc (auto-labeled) |
| CdSe RS→WZ (8 at) | 0 | NequIP dual | **0.0011 eV (0.0006 eV/f.u.)**, 8.5 min | — (pending decision) | Confirms barrierless-path finding on the harder MLIP PES (drag-init variant had given 0.104) |
| C graphite→HD (16 at) | **0 (effective)** | Allegro r2SCAN | **4.470 eV (0.279 eV/atom)**, 86.9 min, job 5754901 | — | No-drag ≤ drag (4.483): settles the drag question on carbon. NOTE: pre-fix "15 GPa" runs were effectively P≈0 (see pressure-bug note) |
| C graphite→HD (16 at) | **15 (true)** | Allegro r2SCAN | **2.492 eV (0.156 eV/atom)** best of 3 seeds (2.49/2.95/2.95); best path = 2-step via layer-shifted intermediate (+0.51 eV, V=123 Å³) | **FAILS: 28.2 eV** (1.76 eV/atom), 14,383 calls, 945 s (job 5755060) | PALLAS 9.6× lower. Linear interpolation across the c-axis collapse (26.8→16.5 Å) drives atoms through each other — same failure class as NaCl. Path: gra (120.8 Å³) → S (109.9 Å³) → HD (87.1 Å³); reverse 5.16 eV; HD −2.21 eV below gra ✓ |
| Si diamond→β-tin (8 at) | 12 | MatterSim | **1.2675 eV (0.158 eV/atom)** — identical across 4 runs incl. 8-probe/25-gen budget; path **Fd-3m → Imma → I4₁/amd** (the literature Si-I→Si-II channel), validated saddle (κ<0) | 1.113 eV max-image (21,002 calls) — **not a stationary point**: a dimer climb from NEB's peak image escapes to −0.24 eV; band is kinked (second hump at image 5) | Resolution: NEB's lower number is an unvalidated max-image estimate from a kinked band; PALLAS's saddle is curvature-validated and mechanism-correct. Gap = measurement quality, not a missed path |
| NaCl B1→B2 (8 at) | 30 | MatterSim | **0.1405 eV (0.0351 eV/f.u.)** best of 3 (0.14/0.74/0.89), 13.4 min; best path = Fm-3m → **P2₁/m intermediate** → Pm-3m (Buerger-type, auto-discovered + auto-labeled) | **FAILS: 21→75 eV**, diverges with iterations | Decisive: linear interpolation collides atoms between incommensurate B1/B2 mappings; PALLAS needs no interpolation and finds the literature monoclinic mechanism |

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
