import os
import torch
from loguru import logger
from ase import io
from ase.optimize import FIRE, BFGS
from ase.constraints import ExpCellFilter
from mattersim.forcefield import MatterSimCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

device = "cpu"

#use mattersim instead

def load_calculator():
    return MatterSimCalculator(device=device)

def print_spacegroup(atom):
    try:
        spg_analyzer = SpacegroupAnalyzer(AseAtomsAdaptor().get_structure(atom), symprec=1e-3)
        print("Space group:", spg_analyzer.get_space_group_symbol())
    except Exception as e:
        print("Failed to determine space group:", e)

filename = "POSCAR2"
structure = io.read(filename, format="vasp")

calc = load_calculator()
structure.calc = calc

try:
    energy = structure.get_potential_energy()
    forces = structure.get_forces()
    stress = structure.get_stress()
    print(f"Energy: {energy:.6f} eV")
    print("Forces:\n", forces)
    print("Stress:\n", stress)
except Exception as e:
    print("Error in calculating properties:", e)
    exit()

print_spacegroup(structure)

ecf = ExpCellFilter(structure)  
dyn = BFGS(ecf, trajectory="optim.traj")
success = dyn.run(fmax=0.01, steps=500)

if not success:
    logger.warning("Optimization exceeded 500 steps and was skipped.")
else:
    print_spacegroup(structure)
    io.write("POSCAR_OPTIMIZED", structure, format="vasp")
    logger.info("Optimized structure saved as POSCAR_OPTIMIZED")

