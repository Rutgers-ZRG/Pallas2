"""Golden regression: CdSe RS->WZ with MatterSim (8 atoms, seed 42).

Prior reference results (eV/f.u.): 0.0127 (no-drag, Mar 22), 0.0084
(drag-init, Mar 22). The assertion band [0.005, 0.020] covers seed-to-seed
spread of the no-drag method; a value outside it after a code change means
search behavior changed — investigate before merging.

Local/nightly only (downloads MatterSim weights, ~1 min run): excluded
from CI by the golden marker.
"""
import random

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from pallas import Pallas, PallasConfig

pytestmark = pytest.mark.golden


def test_cdse_rs_wz_barrier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    random.seed(42)
    np.random.seed(42)

    rs = bulk('CdSe', 'rocksalt', a=6.05).repeat((2, 1, 1))
    wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)
    write('POSCAR1', rs, format='vasp', direct=True, sort=True)
    write('POSCAR2', wz, format='vasp', direct=True, sort=True)

    config = PallasConfig(
        znucl=sorted(set(rs.get_atomic_numbers().tolist())),
        fpcutoff=5.5, natx=100, press=0.0,
        n_probes=3, max_gen=10, patience=3,
        opt_steps=500, opt_fmax=0.005,
        saddle_steps=500, saddle_fmax=0.05,
        fp_step_scale=0.5, fp_push_scale=0.1,
        ediff=0.01, dist_threshold=0.1,
    )

    pallas = Pallas(config)
    pallas.init_run(['POSCAR1', 'POSCAR2'])
    path, barrier = pallas.run()

    assert path is not None, "no RS->WZ path found"
    per_fu = barrier / 4  # 8 atoms = 4 formula units
    assert 0.005 <= per_fu <= 0.020, f"barrier {per_fu:.4f} eV/f.u. outside golden band"
