"""Probe mechanics: FP-guided saddle search steps (mixin for Pallas)."""


from copy import deepcopy as cp

import numpy as np

from pallas.config import SADDLE_MIN_IDENTITY_DFP, SADDLE_MIN_IDENTITY_EDIFF
from pallas.optimize import cal_saddle, local_optimization, vrand, vunit
from pallas.structure import enthalpy, fp_distance, spacegroup_label
from pallas.xcal import XCalculator

# ── Main PALLAS class ────────────────────────────────────────────────


def gen_random_mode(structure):
    """Random normalized generalized mode (atoms + cell rows)."""
    natom = len(structure)
    mode = vrand(np.zeros((natom + 3, 3)))
    mode[0] *= 0
    mode[-3, 1:] *= 0
    mode[-2, 2] *= 0
    return vunit(mode)


def fp_gradient_mode(structure, target_fp, cfg):
    """Compute normalized mode pointing toward target in FP space.

    Uses XCalculator to get forces (position gradient) and stress
    (cell gradient) of the FP distance, then assembles them into
    a (natom+3, 3) dimer-compatible mode vector.

    Parameters
    ----------
    structure : PallasAtom
    target_fp : np.ndarray, shape (nat, fp_dim)

    Returns
    -------
    np.ndarray, shape (natom+3, 3) — normalized mode.
    """
    atoms = structure.copy()
    calc = XCalculator(
        fp0=target_fp, znucl=cfg.znucl,
        cutoff=cfg.fpcutoff, natx=cfg.natx)
    atoms.calc = calc

    forces = atoms.get_forces()      # (natom, 3)
    stress = atoms.get_stress()      # (6,) Voigt

    natom = len(atoms)
    vol = atoms.get_volume()
    jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

    # Voigt → 3x3 strain gradient
    # stress_voigt = (1/V) * dE/d(epsilon)
    # Cell descent direction: -dE/d(epsilon) = -V * stress_3x3
    stress_3x3 = np.zeros((3, 3))
    stress_3x3[0, 0] = stress[0]
    stress_3x3[1, 1] = stress[1]
    stress_3x3[2, 2] = stress[2]
    stress_3x3[1, 2] = stress_3x3[2, 1] = stress[3]
    stress_3x3[0, 2] = stress_3x3[2, 0] = stress[4]
    stress_3x3[0, 1] = stress_3x3[1, 0] = stress[5]

    mode = np.zeros((natom + 3, 3))
    mode[:natom] = forces                           # position direction
    mode[-3:] = -jacob * vol * stress_3x3           # cell direction

    # Constrain redundant freedoms
    mode[0] *= 0
    mode[-3, 1:] *= 0
    mode[-2, 2] *= 0

    return vunit(mode)


def fp_guided_saddle(structure, mode, step_scale, cfg,
                     n_mode_tries=3, traj_frames=None):
    """Perturb along FP gradient, then find saddle with aligned dimer.

    Tries multiple initial mode directions from the same perturbed
    structure: the FP gradient mode first, then mixed FP+random variants.
    Returns the first converged saddle (negative curvature).

    Parameters
    ----------
    structure : PallasAtom — starting minimum.
    mode : np.ndarray — FP gradient mode (natom+3, 3).
    step_scale : float — perturbation magnitude.
    n_mode_tries : int — number of different initial modes to try.
    traj_frames : list, optional — collect dimer trajectory frames.

    Returns
    -------
    PallasAtom — saddle point (best attempt).
    """
    natom = len(structure)
    vol = structure.get_volume()
    jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

    # Perturb structure along FP gradient
    perturbed = cp(structure)
    scaled = mode * step_scale
    cellt = perturbed.get_cell() + np.dot(perturbed.get_cell(),
                                           scaled[-3:] / jacob)
    perturbed.set_cell(cellt, scale_atoms=True)
    perturbed.set_positions(perturbed.get_positions() + scaled[:-3])
    if hasattr(perturbed, 'invalidate_fp'):
        perturbed.invalidate_fp()

    # Try multiple initial modes from this perturbed structure
    best_saddle = None
    best_curvature = float('inf')
    best_frames = None

    for attempt in range(n_mode_tries):
        if attempt == 0:
            trial_mode = mode  # pure FP gradient
        else:
            # Mix FP gradient with random (decreasing FP weight)
            rand = vrand(np.zeros_like(mode))
            mix = max(0.7 - 0.3 * attempt, 0.1)
            trial_mode = vunit(mix * mode + (1.0 - mix) * rand)

        attempt_frames = [] if traj_frames is not None else None
        saddle = cal_saddle(cp(perturbed), fmax=cfg.saddle_fmax,
                            steps=cfg.saddle_steps, press=cfg.press, mode=trial_mode,
                            traj_frames=attempt_frames)
        curv = getattr(saddle, 'dimer_curvature', None)

        if curv is not None and curv < best_curvature:
            best_curvature = curv
            best_saddle = saddle
            best_frames = attempt_frames

        # Stop early if we found a true saddle
        if curv is not None and curv < 0 and saddle.converged:
            if traj_frames is not None and attempt_frames:
                traj_frames.extend(attempt_frames)
            return saddle

    if traj_frames is not None and best_frames:
        traj_frames.extend(best_frames)
    return best_saddle if best_saddle is not None else saddle


def saddle_escape(saddle, target_fp, cfg):
    """Escape saddle toward target using dimer mode push only.

    Pushes along the dimer's unstable mode in the direction toward
    the target (determined by projecting the FP gradient onto the
    dimer mode).  No non-physical FP bias is applied — after the push,
    the caller should relax on the real PES via local_optimization,
    which naturally flows to the nearest minimum without skipping
    intermediate barriers.

    Parameters
    ----------
    saddle : PallasAtom — saddle point with .dimer_mode attribute.
    target_fp : np.ndarray — target fingerprints.

    Returns
    -------
    PallasAtom — structure pushed to the target side of the saddle.
    """
    atoms = cp(saddle)
    natom = len(atoms)
    vol = atoms.get_volume()
    jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

    # Get dimer's unstable mode
    dimer_mode = getattr(saddle, 'dimer_mode', None)
    if dimer_mode is None:
        # Fallback: use FP gradient
        dimer_mode = fp_gradient_mode(saddle, target_fp, cfg)

    # Determine which direction along the dimer mode leads toward target:
    # project FP gradient onto dimer mode
    fp_mode = fp_gradient_mode(saddle, target_fp, cfg)
    dot = np.vdot(dimer_mode, fp_mode)
    push_dir = vunit(dimer_mode) * np.sign(dot) if abs(dot) > 1e-12 \
        else vunit(fp_mode)

    # Push along dimer mode to cross the saddle ridge
    push_scale = cfg.fp_push_scale * 3.0
    scaled = push_dir * push_scale
    cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
    atoms.set_cell(cellt, scale_atoms=True)
    atoms.set_positions(atoms.get_positions() + scaled[:-3])

    if hasattr(atoms, 'invalidate_fp'):
        atoms.invalidate_fp()
    return atoms


def probe_compute(current, target_fp, types, cfg, step_scale=None,
                  alpha=1.0, side="", seed=None, calc=None):
    """Pure compute stage of one probe: perturb -> dimer -> escape -> optimize.

    Safe to run in a worker process: touches no shared state. ``calc`` is an
    optional picklable calculator to install in the worker (None = lazy
    default in pallas.optimize). Returned Atoms carry SinglePointCalculators
    so energies are readable without a live calculator.

    Returns
    -------
    dict with keys: ok, error, saddle, e_sad, v_sad, new_min, e_min, v_min,
    traj_frames, target_fp.
    """
    import random as _random

    from ase.calculators.singlepoint import SinglePointCalculator

    if seed is not None:
        _random.seed(seed)
        np.random.seed(seed)
    from pallas.optimize import GeometryGuard, _get_calculator, set_calculator
    base = calc if calc is not None else _get_calculator()
    if not isinstance(base, GeometryGuard):
        base = GeometryGuard(base)
    set_calculator(base)

    try:
        fp_mode = fp_gradient_mode(current, target_fp, cfg)
    except Exception as e:
        return {'ok': False, 'error': f'fp gradient failed: {e}',
                'target_fp': target_fp}
    rand_mode = gen_random_mode(current)
    mode = vunit(alpha * fp_mode + (1.0 - alpha) * rand_mode)

    try:
        e_src = float(current.get_potential_energy())
    except Exception:
        e_src = None

    traj_frames = [current.copy()]
    saddle = None
    err = None
    for attempt in range(cfg.max_retries + 1):
        try:
            base_scale = cfg.fp_step_scale if step_scale is None else step_scale
            scale = base_scale * (0.5 ** attempt)
            cand = fp_guided_saddle(current, mode, scale, cfg,
                                    traj_frames=traj_frames)
            e_cand = float(cand.get_potential_energy())
            if not np.isfinite(e_cand):
                raise ValueError('non-finite saddle energy (runaway PES)')
            v_ratio = cand.get_volume() / max(current.get_volume(), 1e-9)
            if not 0.3 <= v_ratio <= 3.0:
                raise ValueError(
                    f'saddle cell volume ratio {v_ratio:.2f} outside [0.3, 3] '
                    f'— collapsed/exploded cell')
            if e_src is not None and e_cand - e_src > cfg.max_saddle_rise:
                raise ValueError(
                    f'saddle {e_cand - e_src:.1f} eV above source '
                    f'(> max_saddle_rise={cfg.max_saddle_rise}) — runaway dimer')
            saddle = cand
            break
        except Exception as e:
            err = e
            if attempt < cfg.max_retries:
                print(f"  [{side}] Saddle attempt {attempt+1} failed "
                      f"(scale={scale:.3f}): {e}, retrying...")
    if saddle is None:
        return {'ok': False, 'error': str(err), 'target_fp': target_fp}

    e_sad = saddle.get_potential_energy()
    v_sad = saddle.get_volume()

    try:
        escaped = saddle_escape(saddle, target_fp, cfg)
        traj_frames.append(escaped.copy())
        new_min = local_optimization(escaped, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps, press=cfg.press,
                                     traj_frames=traj_frames)
        e_min = float(new_min.get_potential_energy())
        if not np.isfinite(e_min):
            raise ValueError('non-finite minimum energy after escape')
        v_ratio = new_min.get_volume() / max(current.get_volume(), 1e-9)
        if not 0.3 <= v_ratio <= 3.0:
            raise ValueError(
                f'minimum cell volume ratio {v_ratio:.2f} outside [0.3, 3]')
    except Exception as e:
        return {'ok': False, 'error': f'escape/optimize failed: {e}',
                'target_fp': target_fp}
    v_min = new_min.get_volume()

    for at, e in ((saddle, e_sad), (new_min, e_min)):
        at.calc = SinglePointCalculator(at, energy=e)

    return {'ok': True, 'error': None,
            'saddle': saddle, 'e_sad': float(e_sad), 'v_sad': float(v_sad),
            'new_min': new_min, 'e_min': float(e_min), 'v_min': float(v_min),
            'traj_frames': traj_frames, 'target_fp': target_fp}


class ProbeMixin:
    """FP-guided probe steps: perturb, dimer, validate, escape."""

    def _gen_random_velocity(self, structure):
        return gen_random_mode(structure)

    def _fp_gradient_mode(self, structure, target_fp):
        return fp_gradient_mode(structure, target_fp, self.config)

    def _fp_guided_saddle(self, structure, mode, step_scale, traj_frames=None):
        return fp_guided_saddle(structure, mode, step_scale, self.config,
                                traj_frames=traj_frames)

    def _saddle_escape(self, saddle, target_fp):
        return saddle_escape(saddle, target_fp, self.config)

    def _validate_saddle(self, saddle, types):
        """Validate a saddle point: curvature check + connectivity test.

        1. Curvature check: dimer curvature must be negative.
        2. Connectivity: push along ±dimer_mode, relax on real PES,
           verify the two endpoints are distinct minima.

        Parameters
        ----------
        saddle : PallasAtom — saddle with .dimer_mode and .dimer_curvature.
        types : np.ndarray — atom type array.

        Returns
        -------
        dict with keys:
            valid : bool — True if saddle passes both checks.
            curvature : float — dimer curvature value.
            min_plus : PallasAtom or None — minimum from +mode descent.
            min_minus : PallasAtom or None — minimum from -mode descent.
            reason : str — failure reason if not valid.
        """
        cfg = self.config
        result = {'valid': False, 'curvature': None,
                  'min_plus': None, 'min_minus': None, 'reason': ''}

        # Curvature is advisory, not a hard reject — the dimer mode
        # may not align with the true unstable direction.
        curvature = getattr(saddle, 'dimer_curvature', None)
        result['curvature'] = curvature

        # D7 guard: a "saddle" identical to a known minimum is invalid
        # outright (coarse run thresholds can't be trusted for identity —
        # capped at the tight absolute scale)
        if self.db is not None:
            sid = getattr(saddle, 'id', None)
            hs = None
            if sid is not None:
                try:
                    # enthalpy comparison — raw E shifts by P·dV at pressure
                    hs = (self.db.get(id=sid).data['energy']
                          + cfg.press * saddle.get_volume())
                except Exception:
                    hs = None
            dfp_id = min(cfg.dist_threshold, SADDLE_MIN_IDENTITY_DFP)
            de_id = min(cfg.ediff, SADDLE_MIN_IDENTITY_EDIFF)
            for x in self.db.select(ctyp='minima'):
                d = fp_distance(saddle.get_fp(), np.array(x.data['fp']),
                                types)
                hx = x.data['energy'] + cfg.press * x.volume
                if d < dfp_id and (hs is None or abs(hs - hx) < de_id):
                    result['reason'] = (f'identical to minimum M{x.id} '
                                        f'(d={d:.5f}) — not a saddle')
                    return result

        # Primary check: connectivity (push ±mode, relax, verify distinct minima)
        dimer_mode = getattr(saddle, 'dimer_mode', None)
        if dimer_mode is None:
            result['reason'] = 'no dimer_mode stored'
            return result

        mode = vunit(dimer_mode)
        natom = len(saddle)
        vol = saddle.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5
        base_push = cfg.fp_push_scale * 3.0  # same scale as _saddle_escape

        # Sharp ridges (large |curvature|) may need a larger displacement to
        # commit to a basin: escalate the push until the two sides separate.
        last_reason = ''
        for mult in (1.0, 2.0, 4.0):
            push = base_push * mult
            min_plus = self._push_and_relax(saddle, mode, push, jacob)
            min_minus = self._push_and_relax(saddle, -mode, push, jacob)

            result['min_plus'] = min_plus
            result['min_minus'] = min_minus

            if min_plus is None or min_minus is None:
                last_reason = 'relaxation failed'
                continue

            try:
                d = fp_distance(min_plus.get_fp(), min_minus.get_fp(), types)
                ediff = abs(min_plus.get_potential_energy()
                            - min_minus.get_potential_energy())
            except Exception as exc:  # e.g. GeometryError on a degenerate basin
                last_reason = f'side evaluation failed: {exc}'
                continue

            if d < cfg.dist_threshold and ediff < cfg.ediff:
                last_reason = (f'same minimum on both sides '
                               f'(d={d:.5f}, dE={ediff:.5f}, push={push:.2f})')
                continue

            result['valid'] = True
            break
        else:
            result['reason'] = last_reason
            return result
        return result

    def _push_and_relax(self, saddle, mode, push_scale, jacob):
        """Push saddle along mode and relax on real PES.

        Returns optimized PallasAtom or None on failure.
        """
        try:
            atoms = cp(saddle)
            scaled = mode * push_scale
            cellt = atoms.get_cell() + np.dot(atoms.get_cell(),
                                              scaled[-3:] / jacob)
            atoms.set_cell(cellt, scale_atoms=True)
            atoms.set_positions(atoms.get_positions() + scaled[:-3])
            if hasattr(atoms, 'invalidate_fp'):
                atoms.invalidate_fp()

            cfg = self.config
            opt = local_optimization(atoms, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps, press=cfg.press)
            return opt
        except Exception:
            return None

    def _fp_guided_step(self, current, target_fp, types,
                        step_scale, alpha=1.0, side=""):
        """One step of FP-guided chain growing (dimer from minimum).

        Compute (perturb -> dimer -> escape -> optimize) happens in
        ``probe_compute``; registration (DB, graph, trajectories) in
        ``_register_probe``. Serial path = compute + register in-process.

        Returns
        -------
        PallasAtom — new minimum (or ``current`` if the step failed).
        """
        result = probe_compute(current, target_fp, types, self.config,
                               step_scale=step_scale, alpha=alpha, side=side)
        return self._register_probe(current, result, types, side=side)

    def _register_probe(self, current, result, types, side=""):
        """Register one probe result: DB rows, graph nodes/edges, traj files.

        Must run in the main process (mutates shared state).
        """
        cfg = self.config
        if not result['ok']:
            print(f"  [{side}] All saddle attempts failed: {result['error']}")
            return current

        saddle, new_min = result['saddle'], result['new_min']
        traj_frames = result['traj_frames']

        sad_id, _ = self._update_saddle(saddle)
        if sad_id is None:
            # D7: fake saddle (identical to a known minimum) — keep the
            # discovered endpoint minimum, skip the saddle and its edges
            print(f"  [{side}] saddle rejected as minimum-duplicate; "
                  f"registering the discovered minimum only")
            min_id, _ = self._update_minima(new_min)
            new_min.id = min_id
            h_min = (enthalpy(result['e_min'], result['v_min'], cfg.press)
                     - self.baseenergy)
            if min_id not in self.G:
                self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                                volume=result['v_min'],
                                spg=spacegroup_label(new_min)[0])
                self._save_to_traj(new_min, min_id, 'minima', h_min)
            return new_min
        saddle.id = sad_id
        h_sad = (enthalpy(result['e_sad'], result['v_sad'], cfg.press)
                 - self.baseenergy)
        if sad_id not in self.G:
            self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                            volume=result['v_sad'],
                            spg=spacegroup_label(saddle)[0])
            self._save_to_traj(saddle, sad_id, 'saddle', h_sad)
        else:
            h_sad = self.G.nodes[sad_id]['e']

        fp_c = current.get_fp()
        fp_s = saddle.get_fp()
        d_cs = fp_distance(fp_c, fp_s, types)
        h_cur = self.G.nodes[current.id]['e']
        if h_sad > h_cur:
            self.G.add_edge(current.id, sad_id,
                            weight=max(h_cur, h_sad), dist=d_cs)
        else:
            print(f"  [{side}] Skipping edge M{current.id}->S{sad_id}: "
                  f"saddle ({h_sad:.4f}) <= minimum ({h_cur:.4f})")

        curv = getattr(saddle, 'dimer_curvature', None)
        curv_str = f", k={curv:.3f}" if curv is not None else ""

        min_id, _ = self._update_minima(new_min)
        new_min.id = min_id
        h_min = (enthalpy(result['e_min'], result['v_min'], cfg.press)
                 - self.baseenergy)
        if min_id not in self.G:
            self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                            volume=result['v_min'],
                            spg=spacegroup_label(new_min)[0])
            self._save_to_traj(new_min, min_id, 'minima', h_min)
        else:
            h_min = self.G.nodes[min_id]['e']

        fp_m = new_min.get_fp()
        d_sm = fp_distance(fp_s, fp_m, types)
        traj_file = f'traj_M{current.id}_S{sad_id}_M{min_id}.extxyz'
        if h_sad > h_min:
            self.G.add_edge(sad_id, min_id,
                            weight=max(h_sad, h_min), dist=d_sm,
                            traj_file=traj_file)
        else:
            print(f"  [{side}] Skipping edge S{sad_id}->M{min_id}: "
                  f"saddle ({h_sad:.4f}) <= minimum ({h_min:.4f})")

        if self.G.has_edge(current.id, sad_id):
            self.G.edges[current.id, sad_id]['traj_file'] = traj_file

        if traj_frames:
            from ase.io import write as ase_write
            ase_write(traj_file, traj_frames)

        d_target = fp_distance(fp_m, result['target_fp'], types)
        print(f"  [{side}] S{sad_id} ({h_sad:.3f}{curv_str}) "
              f"-> M{min_id} ({h_min:.3f}) | d(->target)={d_target:.5f}")
        return new_min

    def _fp_guided_push(self, saddle, target_fp):
        """Small FP-guided push on saddle toward target.

        Parameters
        ----------
        saddle : PallasAtom — saddle point structure.
        target_fp : np.ndarray — target fingerprint.

        Returns
        -------
        PallasAtom — slightly displaced structure.
        """
        mode = self._fp_gradient_mode(saddle, target_fp)
        atoms = cp(saddle)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        scaled = mode * self.config.fp_push_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])
        atoms.invalidate_fp()
        return atoms

    # ── Database / deduplication ──────────────────────────────────────

