"""Probe failures (NaN forces, runaway saddles) must fail the probe, not the run."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from pallas import PallasAtom, PallasConfig, local_optimization
from pallas.probes import probe_compute

pytestmark = pytest.mark.unit


class NaNCalc(Calculator):
    """Returns finite results for n_good calls, then NaNs (runaway PES)."""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, n_good=0):
        super().__init__()
        self.n_good = n_good
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        bad = self.calls > self.n_good
        f = np.full((len(self.atoms), 3), np.nan) if bad else np.zeros((len(self.atoms), 3))
        e = np.nan if bad else -1.0
        self.results = {'energy': e, 'forces': f, 'stress': np.zeros(6)}


def test_local_optimization_survives_nan_forces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    r = local_optimization(atoms, fmax=0.01, steps=50, calc=NaNCalc(n_good=0))
    assert r.converged is False  # must not raise


def test_probe_compute_rejects_nan_pes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = PallasAtom(bulk('Cu', 'fcc', a=3.6, cubic=True))
    src.znucl = [29]
    src.natx = 60
    tgt = PallasAtom(bulk('Cu', 'fcc', a=3.9, cubic=True))
    tgt.znucl = [29]
    tgt.natx = 60
    cfg = PallasConfig(znucl=[29], natx=60, saddle_steps=50, opt_steps=50,
                       max_retries=0)
    result = probe_compute(src, tgt.get_fp(), np.ones(4, dtype=int), cfg,
                           calc=NaNCalc(n_good=3))
    assert result['ok'] is False
