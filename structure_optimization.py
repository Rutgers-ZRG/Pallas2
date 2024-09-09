import pickle
import numpy as np
from nequip.ase.nequip_calculator import nequip_calculator
import matplotlib.pyplot as plt
from ase.io import read, write, Trajectory
from ase.build.supercells import make_supercell
from ase.optimize import LBFGS, FIRE
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.constraints import ExpCellFilter
from custom_calc_mixed import Custom_Nequip_Calc
from ase.build.tools import niggli_reduce, sort
from scipy.optimize import linear_sum_assignment

def print_spacegroup(atom):
  print(SpacegroupAnalyzer(AseAtomsAdaptor().get_structure(atom), symprec=1e-5).get_space_group_symbol())

nequip_ef_path = "model/cdse_energy_force_model.pth"
nequip_s_path = "model/cdse_stress_model.pth"


calc  = Custom_Nequip_Calc(nequip_ef_calc_path=nequip_ef_path, nequip_s_calc_path=nequip_s_path)


atom = read("POSCAR2", format="vasp")
atom.calc = calc

print(atom.get_potential_energy())
print(atom.get_forces())
print(atom.get_stress())
print_spacegroup(atom)

#
#ecf = ExpCellFilter(atom)
ecf = atom
dyn = FIRE(ecf, trajectory="optim.traj")
dyn.run(fmax=0.01)

print_spacegroup(atom)

write("POSCAR_OPTIMIZED", atom, format='vasp', direct=True)
