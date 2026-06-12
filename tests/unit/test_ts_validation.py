"""TS-validation machinery: DB persistence, escalating push, guarded refine."""
import numpy as np
import pytest
from ase import Atoms

from pallas import Pallas, PallasConfig
from pallas.structure import PallasAtom

pytestmark = pytest.mark.unit


def _fake_saddle(e=1.0):
    a = PallasAtom(Atoms('Cu2', positions=[[0, 0, 0], [2.0, 0, 0]],
                         cell=[8, 8, 8], pbc=True))
    a.znucl = [29]
    a.natx = 30
    from ase.calculators.singlepoint import SinglePointCalculator
    a.calc = SinglePointCalculator(a, energy=e)
    a.dimer_mode = np.zeros((5, 3))
    a.dimer_mode[0, 0] = 1.0
    a.dimer_curvature = -0.5
    return a


def test_saddle_registration_persists_mode_and_curvature(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = Pallas(PallasConfig(znucl=[29], natx=30))
    import ase.db
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    sid, isnew = p._update_saddle(_fake_saddle())
    assert isnew
    row = p.db.get(id=sid)
    assert 'dimer_mode' in row.data and 'curvature' in row.data
    assert row.data['curvature'] == pytest.approx(-0.5)
    assert np.asarray(row.data['dimer_mode']).shape == (5, 3)


def test_validate_saddle_escalates_push(monkeypatch):
    """If ±push lands in the same basin, the validator must retry with a
    larger push before giving up."""
    p = Pallas(PallasConfig(znucl=[29], natx=30, dist_threshold=0.05,
                            ediff=0.01))
    saddle = _fake_saddle()

    pushes = []
    same = _fake_saddle(e=0.0)
    distinct = _fake_saddle(e=5.0)

    def fake_push_and_relax(s, mode, push, jacob):
        pushes.append(round(push, 4))
        # First two calls (push x1, both signs): same minimum both sides.
        # After escalation: distinct minima.
        if len(pushes) <= 2:
            return same
        return distinct if mode[0, 0] > 0 else same

    monkeypatch.setattr(p, '_push_and_relax', fake_push_and_relax)
    monkeypatch.setattr('pallas.probes.fp_distance', lambda *a: 0.0)

    result = p._validate_saddle(saddle, types=np.array([1, 1]))
    # escalation happened: more than one push magnitude tried
    assert len(set(pushes)) >= 2, f"no escalation: pushes={pushes}"
    assert max(pushes) > min(pushes) * 1.5


def test_validate_saddle_survives_side_evaluation_error(monkeypatch):
    """A pathological pushed structure (GeometryError on energy read) must
    fail that escalation round, not crash validate_graph."""
    from pallas.optimize import GeometryError

    p = Pallas(PallasConfig(znucl=[29], natx=30))
    saddle = _fake_saddle()

    class Boom:
        def get_fp(self):
            raise GeometryError('degenerate basin')

        def get_potential_energy(self):
            raise GeometryError('degenerate basin')

    monkeypatch.setattr(p, '_push_and_relax', lambda *a: Boom())
    result = p._validate_saddle(saddle, types=np.array([1, 1]))
    assert result['valid'] is False
    assert 'failed' in result['reason']
