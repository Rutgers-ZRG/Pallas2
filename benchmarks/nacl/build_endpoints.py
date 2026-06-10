#!/usr/bin/env python3
"""NaCl B1 (rocksalt) vs B2 (CsCl): endpoints + MatterSim dH(P) sanity gate.

Literature: B1 -> B2 near 27-30 GPa (exp ~26-30). Gate: B2 favorable
somewhere in 20-40 GPa. Writes POSCAR_A (B1, 8 at) and POSCAR_B
(B2 2x2x1, 8 at) relaxed at the target pressure.
"""
import sys

from ase.build import bulk
from ase.io import write
from ase.spacegroup import crystal
from ase.units import GPa

from pallas import local_optimization
from pallas.structure import enthalpy

P_GPA = 30.0


def make_b1():
    return bulk('NaCl', 'rocksalt', a=5.64, cubic=True)  # 8 atoms


def make_b2():
    b2 = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=221,
                 cellpar=[3.06, 3.06, 3.06, 90, 90, 90])  # 2 atoms
    return b2.repeat((2, 2, 1))  # 8 atoms


def h_per_fu(atoms, p_gpa):
    press = p_gpa * GPa
    r = local_optimization(atoms.copy(), fmax=0.005, steps=1000, press=press)
    nfu = len(r) // 2
    return enthalpy(r.get_potential_energy(), r.get_volume(), press) / nfu, r


def main():
    print(f"{'P(GPa)':>7} {'H_B1(eV/fu)':>12} {'H_B2(eV/fu)':>12} {'dH(B2-B1)':>10}")
    favorable = None
    for p in (0.0, 20.0, 30.0, 40.0):
        h1, _ = h_per_fu(make_b1(), p)
        h2, _ = h_per_fu(make_b2(), p)
        print(f"{p:7.1f} {h1:12.4f} {h2:12.4f} {h2 - h1:10.4f}")
        if h2 < h1 and favorable is None:
            favorable = p
    if favorable is None:
        print("GATE FAILED: B2 never favorable <= 40 GPa on MatterSim")
        sys.exit(2)
    print(f"GATE OK: B2 favorable from ~{favorable} GPa")

    _, b1 = h_per_fu(make_b1(), P_GPA)
    _, b2 = h_per_fu(make_b2(), P_GPA)
    write('POSCAR_A', b1, format='vasp', direct=True, sort=True)
    write('POSCAR_B', b2, format='vasp', direct=True, sort=True)
    print(f"endpoints written (relaxed at {P_GPA} GPa)")


if __name__ == '__main__':
    main()
