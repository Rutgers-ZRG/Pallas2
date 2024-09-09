"""This module defines an ASE interface to SpkEnergyOnly calculator
   that can compute forces and stresses through finite differences
"""

import re
from pathlib import Path
from subprocess import check_output
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.vibrations import Vibrations
import logging
from copy import deepcopy
from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from ase.constraints import ExpCellFilter
from ase.io import read, write
import numpy as np
from nequip.ase.nequip_calculator import nequip_calculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
from ase.visualize import view


def numeric_stress(atoms, pressure, d=1e-6):
    stress = np.zeros((3, 3), dtype=float)

    cell = atoms.cell.copy()
    V = atoms.get_volume()
    for i in range(3):
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = calc_energy(atoms)

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = calc_energy(atoms)

        stress[i, i] = (eplus - eminus) / (2 * d * V)
        x[i, i] += d

        j = i - 2
        x[i, j] = d
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = calc_energy(atoms)

        x[i, j] = -d
        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = calc_energy(atoms)

        stress[i, j] = (eplus - eminus) / (4 * d * V)
        stress[j, i] = stress[i, j]
    for i in range(3):
        for j in range(3):
            if i == j:
                stress[i,j] += pressure
            # else:
            #   stress[i,j] = 0.0

    atoms.set_cell(cell, scale_atoms=True)
    return stress

def get_nn_quick(atom):
    # SUBROUTINE GET_ALL_MINDISTS(POSITIONS, CELL, TYPES, MINDISTS, N)
    pos = get_true_positions(atom)
    cell = atom.get_cell()[:]
    syms = atom.get_chemical_symbols()
    types = []
    N = len(atom)
    for s in syms:
        if s == 'Cd':
            types.append(1)
        else:
            types.append(2)
    mindists = np.asfortranarray(np.zeros(len(atom)))
    
    potential.potmod.get_all_mindists(pos, cell, types, mindists, N)
    return mindists


def get_true_positions(atom):
  pos = atom.positions
  cell = atom.get_cell()[:]
  cellinv = np.linalg.inv(cell)
  #scaled_pos = np.zeros(pos.shape)
  newpos = np.zeros(pos.shape)
  for i in range(len(pos)):
    #scaled_pos[i] = cellinv@pos[i]
    tpos = pos[i]@cellinv
    for j in range(3):
      if tpos[j] < 0 :
        tpos[j] = int(np.abs(tpos[j])) + tpos[j] + 1.0
      elif tpos[j] > 1.0:
        tpos[j] = tpos[j] - int(tpos[j])
    #scaled_pos[i] = tpos
    newpos[i] = tpos@cell
  return newpos



class Custom_Nequip_Calc(Calculator):
    energy = "energy"
    forces = "forces"
    stress = "stress"
    implemented_properties=[energy, forces, stress]

    def __init__(
        self,
        nequip_ef_calc_path,
        nequip_s_calc_path
        ):
        Calculator.__init__(self)
        self.neq_ef_calc = nequip_calculator(nequip_ef_calc_path)
        self.neq_s_calc = nequip_calculator(nequip_s_calc_path)

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ):
        if self.calculation_required(atoms, properties):
            results = {}
            newatom = atoms.copy()
            #newatom.set_positions(get_true_positions(newatom))
            newatom.calc = self.neq_ef_calc
            for p in properties:
                if p == "energy":
                    results[p] = newatom.get_potential_energy()
                elif p=="forces":
                    results[p] = newatom.get_forces()
                elif p=="stress":
                    newatom2 = atoms.copy()
                    newatom2.calc = self.neq_s_calc
                    results[p] = -1*newatom2.get_stress()/1602.1766208
            self.results = results
































#
