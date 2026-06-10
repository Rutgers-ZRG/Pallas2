import json
import subprocess
import sys

import pytest
from ase.build import bulk
from ase.io import write

pytestmark = pytest.mark.integration


@pytest.fixture
def emt_pair(tmp_path):
    fcc = bulk('Cu', 'fcc', a=3.61, cubic=True)            # 4 atoms
    hcp = bulk('Cu', 'hcp', a=2.55, c=4.17).repeat((1, 1, 2))  # 4 atoms
    write(str(tmp_path / 'A.vasp'), fcc, format='vasp', direct=True, sort=True)
    write(str(tmp_path / 'B.vasp'), hcp, format='vasp', direct=True, sort=True)
    return tmp_path


def test_cli_run_emt_end_to_end(emt_pair):
    wd = emt_pair / 'out'
    r = subprocess.run(
        [sys.executable, '-m', 'pallas.cli', 'run',
         str(emt_pair / 'A.vasp'), str(emt_pair / 'B.vasp'),
         '--calc', 'emt', '--pressure', '0', '--probes', '2',
         '--max-gen', '3', '--seed', '7', '--workdir', str(wd)],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]

    summary = json.loads((wd / 'summary.json').read_text())
    for key in ('barrier_ev', 'path', 'spacegroups', 'n_minima', 'n_saddles',
                'runtime_s', 'commit'):
        assert key in summary
    assert (wd / 'config.json').exists()


def test_cli_plot(emt_pair):
    wd = emt_pair / 'out2'
    subprocess.run(
        [sys.executable, '-m', 'pallas.cli', 'run',
         str(emt_pair / 'A.vasp'), str(emt_pair / 'B.vasp'),
         '--calc', 'emt', '--probes', '2', '--max-gen', '2',
         '--seed', '7', '--workdir', str(wd)],
        capture_output=True, text=True, timeout=600, check=True)
    r = subprocess.run([sys.executable, '-m', 'pallas.cli', 'plot', str(wd)],
                       capture_output=True, text=True, timeout=120)
    summary = json.loads((wd / 'summary.json').read_text())
    if summary['path']:
        assert r.returncode == 0, r.stderr[-2000:]
        assert (wd / 'profile.png').exists()
    else:  # no path found in 2 gens -> plot must fail gracefully
        assert r.returncode != 0
        assert 'no path' in (r.stderr + r.stdout).lower()
