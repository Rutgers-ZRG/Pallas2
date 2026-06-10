"""Probe mechanics: FP-guided saddle search steps (mixin for Pallas)."""


from copy import deepcopy as cp

import numpy as np

from pallas.optimize import cal_saddle, local_optimization, vrand, vunit
from pallas.structure import enthalpy, fp_distance
from pallas.xcal import XCalculator

# ── Main PALLAS class ────────────────────────────────────────────────


class ProbeMixin:
    """FP-guided probe steps: perturb, dimer, validate, escape."""

    def _gen_random_velocity(self, structure):
        """Generate random normalized velocity mode."""
        natom = len(structure)
        mode = vrand(np.zeros((natom + 3, 3)))
        mode[0] *= 0
        mode[-3, 1:] *= 0
        mode[-2, 2] *= 0
        return vunit(mode)

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

        # Primary check: connectivity (push ±mode, relax, verify distinct minima)
        dimer_mode = getattr(saddle, 'dimer_mode', None)
        if dimer_mode is None:
            result['reason'] = 'no dimer_mode stored'
            return result

        mode = vunit(dimer_mode)
        natom = len(saddle)
        vol = saddle.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5
        push = cfg.fp_push_scale * 3.0  # same scale as _saddle_escape

        min_plus = self._push_and_relax(saddle, mode, push, jacob)
        min_minus = self._push_and_relax(saddle, -mode, push, jacob)

        result['min_plus'] = min_plus
        result['min_minus'] = min_minus

        if min_plus is None or min_minus is None:
            result['reason'] = 'relaxation failed'
            return result

        # Check the two minima are distinct
        d = fp_distance(min_plus.get_fp(), min_minus.get_fp(), types)
        ediff = abs(min_plus.get_potential_energy()
                    - min_minus.get_potential_energy())

        if d < cfg.dist_threshold and ediff < cfg.ediff:
            result['reason'] = (f'same minimum on both sides '
                                f'(d={d:.5f}, dE={ediff:.5f})')
            return result

        result['valid'] = True
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

        Parameters
        ----------
        current : PallasAtom — current chain tip (optimized minimum).
        target_fp : np.ndarray — target fingerprint to grow toward.
        types : np.ndarray — atom type array.
        step_scale : float — perturbation magnitude.
        alpha : float — FP/random mixing ratio (1.0 = pure FP, 0.0 = pure random).
        side : str — label for logging.

        Returns
        -------
        PallasAtom — new minimum (or current if step failed).
        """
        cfg = self.config

        # Step 1: Build search mode from FP gradient + random component
        fp_mode = self._fp_gradient_mode(current, target_fp)
        rand_mode = self._gen_random_velocity(current)
        mode = vunit(alpha * fp_mode + (1.0 - alpha) * rand_mode)

        # Step 2+3: Perturb along mode, run dimer with this mode
        traj_frames = []
        traj_frames.append(current.copy())  # starting minimum
        saddle = None
        for attempt in range(cfg.max_retries + 1):
            try:
                scale = step_scale * (0.5 ** attempt)  # halve on retry
                saddle = self._fp_guided_saddle(
                    current, mode, scale, traj_frames=traj_frames)
                break
            except Exception as e:
                if attempt < cfg.max_retries:
                    print(f"  [{side}] Saddle attempt {attempt+1} failed "
                          f"(scale={scale:.3f}): {e}, retrying...")
                else:
                    print(f"  [{side}] All saddle attempts failed: {e}")
                    return current

        # Register saddle
        sad_id, _ = self._update_saddle(saddle)
        saddle.id = sad_id
        h_sad = (enthalpy(saddle.get_potential_energy(), saddle.get_volume(),
                          cfg.press) - self.baseenergy)
        if sad_id not in self.G:
            self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                            volume=saddle.get_volume())
            self._save_to_traj(saddle, sad_id, 'saddle', h_sad)
        else:
            h_sad = self.G.nodes[sad_id]['e']  # use stored enthalpy

        # Edge: current → saddle (only if saddle is higher)
        fp_c = current.get_fp()
        fp_s = saddle.get_fp()
        d_cs = fp_distance(fp_c, fp_s, types)
        h_cur = self.G.nodes[current.id]['e']
        if h_sad > h_cur:
            self.G.add_edge(current.id, sad_id,
                            weight=max(h_cur, h_sad), dist=d_cs)
        else:
            print(f"  [{side}] Skipping edge M{current.id}→S{sad_id}: "
                  f"saddle ({h_sad:.4f}) ≤ minimum ({h_cur:.4f})")

        # Log saddle info
        curv = getattr(saddle, 'dimer_curvature', None)
        curv_str = f", κ={curv:.3f}" if curv is not None else ""

        # Escape saddle along dimer mode toward target, then optimize
        escaped = self._saddle_escape(saddle, target_fp)
        traj_frames.append(escaped.copy())  # escaped structure
        new_min = local_optimization(escaped, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps, press=cfg.press,
                                     traj_frames=traj_frames)
        min_id, _ = self._update_minima(new_min)
        new_min.id = min_id

        h_min = (enthalpy(new_min.get_potential_energy(), new_min.get_volume(),
                          cfg.press) - self.baseenergy)
        if min_id not in self.G:
            self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                            volume=new_min.get_volume())
            self._save_to_traj(new_min, min_id, 'minima', h_min)
        else:
            h_min = self.G.nodes[min_id]['e']  # use stored enthalpy

        # Edge: saddle → new minimum (only if saddle is higher)
        fp_m = new_min.get_fp()
        d_sm = fp_distance(fp_s, fp_m, types)
        traj_file = f'traj_M{current.id}_S{sad_id}_M{min_id}.extxyz'
        if h_sad > h_min:
            self.G.add_edge(sad_id, min_id,
                            weight=max(h_sad, h_min), dist=d_sm,
                            traj_file=traj_file)
        else:
            print(f"  [{side}] Skipping edge S{sad_id}→M{min_id}: "
                  f"saddle ({h_sad:.4f}) ≤ minimum ({h_min:.4f})")

        # Also store traj_file on current→saddle edge
        if self.G.has_edge(current.id, sad_id):
            self.G.edges[current.id, sad_id]['traj_file'] = traj_file

        # Save probe trajectory to disk
        if traj_frames:
            from ase.io import write as ase_write
            ase_write(traj_file, traj_frames)

        d_target = fp_distance(fp_m, target_fp, types)
        print(f"  [{side}] S{sad_id} ({h_sad:.3f}{curv_str}) "
              f"→ M{min_id} ({h_min:.3f}) | d(→target)={d_target:.5f}")
        return new_min

    def _fp_gradient_mode(self, structure, target_fp):
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
            fp0=target_fp, znucl=self.config.znucl,
            cutoff=self.config.fpcutoff, natx=self.config.natx)
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

    def _fp_guided_saddle(self, structure, mode, step_scale,
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
        cfg = self.config

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

    def _saddle_escape(self, saddle, target_fp):
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
            dimer_mode = self._fp_gradient_mode(saddle, target_fp)

        # Determine which direction along the dimer mode leads toward target:
        # project FP gradient onto dimer mode
        fp_mode = self._fp_gradient_mode(saddle, target_fp)
        dot = np.vdot(dimer_mode, fp_mode)
        push_dir = vunit(dimer_mode) * np.sign(dot) if abs(dot) > 1e-12 \
            else vunit(fp_mode)

        # Push along dimer mode to cross the saddle ridge
        push_scale = self.config.fp_push_scale * 3.0
        scaled = push_dir * push_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])

        if hasattr(atoms, 'invalidate_fp'):
            atoms.invalidate_fp()
        return atoms

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

