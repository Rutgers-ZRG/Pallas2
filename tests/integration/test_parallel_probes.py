"""M1: parallel probe execution must complete and reproduce with a fixed seed."""
import random

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from pallas import Pallas, PallasConfig, set_calculator

pytestmark = pytest.mark.integration


_COUNTER = [0]


def _run(tmp_path, n_workers, seed=11):
    _COUNTER[0] += 1
    wd = tmp_path / f'w{n_workers}_{seed}_{_COUNTER[0]}'
    wd.mkdir()
    import os
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        random.seed(seed)
        np.random.seed(seed)
        fcc = bulk('Cu', 'fcc', a=3.61, cubic=True)
        hcp = bulk('Cu', 'hcp', a=2.55, c=4.17).repeat((1, 1, 2))
        write('POSCAR1', fcc, format='vasp', direct=True, sort=True)
        write('POSCAR2', hcp, format='vasp', direct=True, sort=True)
        set_calculator(EMT())
        cfg = PallasConfig(znucl=[29], natx=60, n_probes=4, max_gen=2,
                           patience=2, opt_steps=300, opt_fmax=0.01,
                           saddle_steps=200, saddle_fmax=0.1,
                           fp_step_scale=0.4, ediff=0.01, dist_threshold=0.05,
                           n_workers=n_workers)
        p = Pallas(cfg)
        p.init_run(['POSCAR1', 'POSCAR2'])
        p.run()
        energies = sorted(round(d['e'], 5) for _, d in p.G.nodes(data=True))
        return energies
    finally:
        os.chdir(cwd)


def test_parallel_completes_and_registers(tmp_path):
    energies = _run(tmp_path, n_workers=2)
    assert len(energies) >= 3  # 2 endpoints + at least one new node


def test_parallel_reproducible_same_seed(tmp_path):
    e1 = _run(tmp_path, n_workers=2, seed=23)
    e2 = _run(tmp_path, n_workers=2, seed=23)
    assert e1 == pytest.approx(e2, abs=1e-4)
