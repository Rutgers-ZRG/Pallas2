"""D6/D7 self-audit guards: fake-saddle rejection + edge invariant.

Lessons from the zero-barrier CdSe/Si audit (2026-08-01): a "saddle" can be
a duplicate of a known minimum (D7), and refinement can push a saddle below
a flanking minimum, breaking the edge invariant registration enforced (D6).
"""
import numpy as np
import pytest

import ase.db
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from pallas import Pallas, PallasConfig
from pallas.structure import PallasAtom

pytestmark = pytest.mark.unit


def _pa(e, dx=0.0):
    a = PallasAtom(Atoms('Cu2', positions=[[0, 0, 0], [2.0 + dx, 0, 0]],
                         cell=[8, 8, 8], pbc=True))
    a.znucl = [29]
    a.natx = 30
    a.calc = SinglePointCalculator(a, energy=e)
    return a


def _pallas(tmp_path, monkeypatch, **kw):
    monkeypatch.chdir(tmp_path)
    p = Pallas(PallasConfig(znucl=[29], natx=30, **kw))
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    return p


def test_saddle_identical_to_minimum_rejected(tmp_path, monkeypatch):
    p = _pallas(tmp_path, monkeypatch)
    m = _pa(e=1.0)
    p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                       'energy': 1.0})
    fake = _pa(e=1.0)  # same structure, same energy -> it IS that minimum
    sid, isnew = p._update_saddle(fake)
    assert sid is None and isnew is False
    # same structure but clearly different energy -> genuine saddle-like row
    real = _pa(e=2.0)
    sid2, isnew2 = p._update_saddle(real)
    assert sid2 is not None and isnew2


def test_identity_check_capped_below_coarse_threshold(tmp_path, monkeypatch):
    """CdSe-style coarse run thresholds (0.1) must not reject a genuinely
    distinct saddle — the identity test is capped at 5e-3."""
    p = _pallas(tmp_path, monkeypatch, dist_threshold=0.1, ediff=0.01)
    m = _pa(e=1.0)
    p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                       'energy': 1.0})
    near = _pa(e=1.002, dx=0.25)  # structurally distinct (d_fp >> 5e-3)
    sid, isnew = p._update_saddle(near)
    assert sid is not None and isnew


def test_identity_check_compares_enthalpy_at_pressure(tmp_path, monkeypatch):
    """At P != 0 the duplicate test must compare H = E + PV: a volume
    difference shifts raw E by P*dV even for the same structure (Si D1
    S13 escaped the raw-E gate exactly this way)."""
    press = 0.01  # eV/A^3
    p = _pallas(tmp_path, monkeypatch, press=press)
    fake = _pa(e=0.992)  # raw dE = 0.008 > 5e-3 ...
    fake.set_cell(np.eye(3) * 8.00521, scale_atoms=True)  # V ~ +1.0 A^3
    fake.calc = SinglePointCalculator(fake, energy=0.992)  # cell change
    # invalidated the cached single-point result
    m = _pa(e=1.0)
    # fake fp identical by construction (stored fp is what the guard reads)
    p.db.write(m, ctyp='minima', data={'fp': fake.get_fp().tolist(),
                                       'energy': 1.0})
    # ... but dH = dE + P*dV = -0.008 + 0.01*1.0 = +0.002 < 5e-3 -> reject
    sid, isnew = p._update_saddle(fake)
    assert sid is None and isnew is False


def test_validate_saddle_flags_minimum_duplicate(tmp_path, monkeypatch):
    p = _pallas(tmp_path, monkeypatch)
    m = _pa(e=1.0)
    p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                       'energy': 1.0})
    fake = _pa(e=1.0)
    sid = p.db.write(fake, ctyp='saddle',
                     data={'fp': fake.get_fp().tolist(), 'energy': 1.0})
    fake.id = sid
    fake.dimer_mode = np.zeros((5, 3))
    fake.dimer_mode[0, 0] = 1.0
    fake.dimer_curvature = -0.1
    res = p._validate_saddle(fake, types=np.array([1, 1]))
    assert res['valid'] is False
    assert 'identical to minimum' in res['reason']


def test_enforce_edge_invariant():
    p = Pallas(PallasConfig(znucl=[29], natx=30))
    p.G.add_node(1, xname='M1', e=0.0)
    p.G.add_node(2, xname='M2', e=-0.5)
    p.G.add_node(3, xname='S3', e=-0.1)  # 0.1 eV below M1, above M2
    p.G.add_edge(1, 3, weight=0.0)
    p.G.add_edge(3, 2, weight=-0.1)
    removed = p._enforce_edge_invariant()
    assert ('S3', 'M1') in removed
    assert not p.G.has_edge(1, 3)
    assert p.G.has_edge(3, 2)
    # sub-tolerance (flat connection) is physical — kept
    p.G.add_node(4, xname='S4', e=-0.5004)
    p.G.add_edge(2, 4, weight=-0.5)
    assert p._enforce_edge_invariant() == []
    assert p.G.has_edge(2, 4)
