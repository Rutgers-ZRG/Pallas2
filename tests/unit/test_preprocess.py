import pytest
from ase.build import bulk
from ase.io import write

from pallas import Pallas, PallasConfig

pytestmark = pytest.mark.unit


def _write(tmp_path, name, atoms):
    p = tmp_path / name
    write(str(p), atoms, format='vasp', direct=True, sort=True)
    return str(p)


def test_composition_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _write(tmp_path, 'A.vasp', bulk('Si', 'diamond', a=5.43, cubic=True))
    b = _write(tmp_path, 'B.vasp', bulk('NaCl', 'rocksalt', a=5.64, cubic=True))
    p = Pallas(PallasConfig(znucl=[14]))
    with pytest.raises(ValueError, match='[Cc]omposition'):
        p.init_run([a, b])


def test_atom_count_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    si8 = bulk('Si', 'diamond', a=5.43, cubic=True)
    a = _write(tmp_path, 'A.vasp', si8)
    b = _write(tmp_path, 'B.vasp', si8.repeat((2, 1, 1)))
    p = Pallas(PallasConfig(znucl=[14]))
    with pytest.raises(ValueError, match='atom count|repeat'):
        p.init_run([a, b])


def test_matched_endpoints_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _write(tmp_path, 'A.vasp', bulk('Si', 'diamond', a=5.43, cubic=True))
    b = _write(tmp_path, 'B.vasp',
               bulk('Si', 'fcc', a=3.9, cubic=True).repeat((2, 1, 1)))
    p = Pallas(PallasConfig(znucl=[14]))
    p.init_run([a, b])
    assert len(p.reactant) == len(p.product) == 8
