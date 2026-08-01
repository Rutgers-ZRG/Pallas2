"""refine_path_saddles: tight re-convergence with guarded acceptance."""
import numpy as np
import pytest

import ase.db
from ase import Atoms

from pallas import Pallas, PallasConfig, cal_saddle
from pallas.structure import PallasAtom

from test_dimer_mb import MB_SADDLE_XY, OFFSET, MuellerBrown

pytestmark = pytest.mark.unit


def _pa(atoms, e=None):
    p = PallasAtom(atoms)
    p.znucl = [1]
    p.natx = 20
    if e is not None:
        from ase.calculators.singlepoint import SinglePointCalculator
        p.calc = SinglePointCalculator(p, energy=e)
    return p


def test_refine_accepts_in_place_and_updates_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Loose saddle: dimer at fmax=0.05 from a start near the MB saddle.
    start = Atoms('H', positions=[[OFFSET - 0.70, OFFSET + 1.00, 0.0]],
                  cell=[50.0] * 3, pbc=True)
    mode = np.zeros((4, 3))
    d = np.array([MB_SADDLE_XY[0] + 0.70, MB_SADDLE_XY[1] - 1.00, 0.0])
    mode[0] = d / np.linalg.norm(d)
    loose = cal_saddle(start, fmax=0.08, steps=2000, calc=MuellerBrown(),
                       mode=mode)
    assert loose.converged

    p = Pallas(PallasConfig(znucl=[1], natx=20))
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    from pallas.optimize import set_calculator
    set_calculator(MuellerBrown())

    for em in (0.0, -0.1):  # endpoints occupy DB ids 1,2 as in real runs
        m = _pa(Atoms('H', positions=[[OFFSET, OFFSET, 0]], cell=[50.0] * 3,
                      pbc=True), e=em)
        m.fp = None
        p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                           'energy': em})
    sad = _pa(loose)
    sad.dimer_mode = loose.dimer_mode
    sad.dimer_curvature = loose.dimer_curvature
    sad.calc = MuellerBrown()
    sid, _ = p._update_saddle(sad)

    p.G.add_node(1, xname='M1', e=0.0, volume=1.0)
    p.G.add_node(sid, xname=f'S{sid}', e=0.5, volume=1.0)
    p.G.add_node(2, xname='M2', e=-0.1, volume=1.0)
    p.G.add_edge(1, sid, weight=0.5)
    p.G.add_edge(sid, 2, weight=0.5)

    report = p.refine_path_saddles(path=[1, sid, 2], fmax=0.01, steps=2000)
    assert len(report) == 1
    r = report[0]
    assert r['accepted'], r
    assert r['curvature'] < 0
    assert abs(r['dfp']) < 0.05
    # graph enthalpy updated by the refinement delta
    assert p.G.nodes[sid]['e'] == pytest.approx(0.5 + r['dH'], abs=1e-8)
    assert p.G.edges[1, sid]['weight'] == pytest.approx(
        max(0.0, p.G.nodes[sid]['e']))
    # refinement must persist the updated graph to disk (summary.json and
    # graph.pkl otherwise disagree by the refinement dH)
    import joblib
    g_disk = joblib.load('graph.pkl')
    assert g_disk.nodes[sid]['e'] == pytest.approx(p.G.nodes[sid]['e'])


def test_refine_second_pass_is_idempotent(tmp_path, monkeypatch):
    """D5: accepted refinement must persist to the DB row so a re-run
    re-derives dH~0 instead of adding the same delta again."""
    monkeypatch.chdir(tmp_path)
    start = Atoms('H', positions=[[OFFSET - 0.70, OFFSET + 1.00, 0.0]],
                  cell=[50.0] * 3, pbc=True)
    mode = np.zeros((4, 3))
    d = np.array([MB_SADDLE_XY[0] + 0.70, MB_SADDLE_XY[1] - 1.00, 0.0])
    mode[0] = d / np.linalg.norm(d)
    loose = cal_saddle(start, fmax=0.08, steps=2000, calc=MuellerBrown(),
                       mode=mode)

    p = Pallas(PallasConfig(znucl=[1], natx=20))
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    from pallas.optimize import set_calculator
    set_calculator(MuellerBrown())

    for em in (0.0, -0.1):
        m = _pa(Atoms('H', positions=[[OFFSET, OFFSET, 0]], cell=[50.0] * 3,
                      pbc=True), e=em)
        p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                           'energy': em})
    sad = _pa(loose)
    sad.dimer_mode = loose.dimer_mode
    sad.dimer_curvature = loose.dimer_curvature
    sad.calc = MuellerBrown()
    sid, _ = p._update_saddle(sad)
    p.G.add_node(1, xname='M1', e=0.0, volume=1.0)
    p.G.add_node(sid, xname=f'S{sid}', e=0.5, volume=1.0)
    p.G.add_node(2, xname='M2', e=-0.1, volume=1.0)
    p.G.add_edge(1, sid, weight=0.5)
    p.G.add_edge(sid, 2, weight=0.5)

    r1 = p.refine_path_saddles(path=[1, sid, 2], fmax=0.01, steps=2000)
    assert r1[0]['accepted']
    e_after_first = p.G.nodes[sid]['e']

    # DB row now carries the refined saddle
    row = p.db.get(id=sid)
    assert row.data.get('refined') is True
    assert row.data['converged'] is True

    # second pass: dH ~ 0, graph energy stable
    r2 = p.refine_path_saddles(path=[1, sid, 2], fmax=0.01, steps=2000)
    assert r2[0]['accepted'], r2
    assert abs(r2[0]['dH']) < 2e-3, f"second pass not idempotent: {r2}"
    assert p.G.nodes[sid]['e'] == pytest.approx(e_after_first, abs=2e-3)


def test_refine_rejects_drift(tmp_path, monkeypatch):
    """If the re-dimer drifts beyond max_drift, keep the original energy."""
    monkeypatch.chdir(tmp_path)
    start = Atoms('H', positions=[[OFFSET - 0.70, OFFSET + 1.00, 0.0]],
                  cell=[50.0] * 3, pbc=True)
    mode = np.zeros((4, 3))
    d = np.array([MB_SADDLE_XY[0] + 0.70, MB_SADDLE_XY[1] - 1.00, 0.0])
    mode[0] = d / np.linalg.norm(d)
    loose = cal_saddle(start, fmax=0.08, steps=2000, calc=MuellerBrown(),
                       mode=mode)

    p = Pallas(PallasConfig(znucl=[1], natx=20))
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    from pallas.optimize import set_calculator
    set_calculator(MuellerBrown())

    for em in (0.0, -0.1):
        m = _pa(Atoms('H', positions=[[OFFSET, OFFSET, 0]], cell=[50.0] * 3,
                      pbc=True), e=em)
        p.db.write(m, ctyp='minima', data={'fp': m.get_fp().tolist(),
                                           'energy': em})
    sad = _pa(loose)
    sad.dimer_mode = loose.dimer_mode
    sad.dimer_curvature = loose.dimer_curvature
    sad.calc = MuellerBrown()
    sid, _ = p._update_saddle(sad)
    p.G.add_node(1, xname='M1', e=0.0, volume=1.0)
    p.G.add_node(sid, xname=f'S{sid}', e=0.5, volume=1.0)
    p.G.add_node(2, xname='M2', e=-0.1, volume=1.0)
    p.G.add_edge(1, sid, weight=0.5)
    p.G.add_edge(sid, 2, weight=0.5)

    monkeypatch.setattr('pallas.search.fp_distance', lambda *a: 0.99)
    report = p.refine_path_saddles(path=[1, sid, 2], fmax=0.01, steps=2000,
                                   max_drift=0.05)  # forced drift -> reject
    assert report[0]['accepted'] is False
    assert p.G.nodes[sid]['e'] == pytest.approx(0.5)  # unchanged
