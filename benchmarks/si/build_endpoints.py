#!/usr/bin/env python3
"""Si diamond vs beta-tin: build endpoints + MatterSim dH(P) sanity gate.

Literature: diamond -> beta-tin transition near 10-12 GPa (static DFT ~8-12).
Gate: beta-tin must become enthalpically favorable somewhere in 8-20 GPa.
Writes POSCAR_A (diamond, 8 at) and POSCAR_B (beta-tin 1x1x2, 8 at) relaxed
at the target pressure.
"""
import sys

from ase.build import bulk
from ase.io import write
from ase.spacegroup import crystal
from ase.units import GPa

from pallas import local_optimization
from pallas.structure import enthalpy

P_GPA = 12.0


def make_diamond():
    return bulk('Si', 'diamond', a=5.43, cubic=True)  # 8 atoms


def make_betatin():
    bt = crystal('Si', [(0.0, 0.0, 0.0)], spacegroup=141,
                 cellpar=[4.81, 4.81, 2.65, 90, 90, 90])  # 4 atoms
    return bt.repeat((1, 1, 2))  # 8 atoms


def h_per_atom(atoms, p_gpa):
    press = p_gpa * GPa
    r = local_optimization(atoms.copy(), fmax=0.005, steps=1000, press=press)
    return enthalpy(r.get_potential_energy(), r.get_volume(), press) / len(r), r


def main():
    print(f"{'P(GPa)':>7} {'H_dia(eV/at)':>13} {'H_bt(eV/at)':>13} {'dH(bt-dia)':>11}")
    favorable = None
    for p in (0.0, 5.0, 12.0, 20.0):
        hd, _ = h_per_atom(make_diamond(), p)
        hb, _ = h_per_atom(make_betatin(), p)
        print(f"{p:7.1f} {hd:13.4f} {hb:13.4f} {hb - hd:11.4f}")
        if hb < hd and favorable is None:
            favorable = p
    if favorable is None or favorable > 20.0:
        print("GATE FAILED: beta-tin never favorable <= 20 GPa on MatterSim")
        sys.exit(2)
    print(f"GATE OK: beta-tin favorable from ~{favorable} GPa")

    _, dia = h_per_atom(make_diamond(), P_GPA)
    _, bt = h_per_atom(make_betatin(), P_GPA)
    write('POSCAR_A', dia, format='vasp', direct=True, sort=True)
    write('POSCAR_B', bt, format='vasp', direct=True, sort=True)
    print(f"endpoints written (relaxed at {P_GPA} GPa)")


if __name__ == '__main__':
    main()
