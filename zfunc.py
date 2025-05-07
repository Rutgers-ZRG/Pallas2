import ssdimer
from util import vunit, vrand
from ase.io import read, write
import os
import sys
import numpy as np

import pickle
import numpy as np
#from nequip.ase.nequip_calculator import nequip_calculator
from ase.io import read, write, Trajectory
from ase.optimize import FIRE, BFGS
# from ase.constraints import ExpCellFilter
from ase.filters import ExpCellFilter
#from custom_calc_mixed import Custom_Nequip_Calc
from copy import deepcopy as cp


#nequip_ef_path = "model/cdse_energy_force_model.pth"
#nequip_s_path = "model/cdse_stress_model.pth"
#
#calc  = Custom_Nequip_Calc(nequip_ef_calc_path=nequip_ef_path, nequip_s_calc_path=nequip_s_path)

from mattersim.forcefield import MatterSimCalculator

device = "cpu"
calc = MatterSimCalculator(device=device)


def local_optimization(patoms):
    atoms = cp(patoms)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile='opt.log')
    opt.run(fmax=0.001, steps=1000)
    fmax = np.max(np.abs(ecf.get_forces()))
    if fmax > 0.001:
        print("Warning: fmax = ", fmax)
        atoms.converged = False
    else:
        atoms.converged = True
    
    new_cell = lower_triangular_cell(atoms)
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.cal_fp()
    # print ('cell', atoms.get_cell())
    # print ('fp', atoms.get_fp())

    # print ('calc', atoms.calc)
    return atoms


def cal_saddle(patoms):
    atoms = cp(patoms)
    atoms.calc = calc
    # calculate the jacobian for the tangent
    natom = len(atoms)
    vol   = atoms.get_volume()
    jacob = (vol/natom)**(1.0/3.0) * natom**0.5

    #######################################
    # set the initial mode randomly
    mode = np.zeros((len(atoms)+3,3))
    mode = vrand(mode)
    ##constrain 3 redundant freedoms
    mode[0]    *=0
    mode[-3,1:]*=0
    mode[-2,2] *=0
    ########################################
    #
    ## displace along the initial mode direction
    mode = vunit(mode)
    cellt = atoms.get_cell()+np.dot(atoms.get_cell(), mode[-3:]/jacob)
    atoms.set_cell(cellt, scale_atoms=True)
    atoms.set_positions(atoms.get_positions() + mode[:-3])

    # set a ssdimer_atoms object
    d = ssdimer.SSDimer_atoms(atoms, mode = mode, ss=True, rotationMax = 4, phi_tol=15)

    # use FIRE optimizer in ase
    dyn = FIRE(d, logfile='ssdimer.log')
    dyn.run(fmax=0.01, steps=1000)
    fmax = np.max(np.abs(atoms.get_forces()))
    if fmax > 0.01:
        print("Warning: fmax = ", fmax)
        atoms.converged = False
    else:
        atoms.converged = True
    new_cell = lower_triangular_cell(atoms)
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.cal_fp()
    return atoms


def getx(cell1, cell2):
    mode = np.zeros((itin.nat + 3, 3))
    mode[-3:] = cell2.get_lattice() - cell1.get_lattice()
    ilat = np.linalg.inv(cell1.get_lattice())
    vol = cell1.get_volume()
    jacob = (vol / itin.nat)**(1.0 / 3.0) * itin.nat**0.5
    mode[-3:] = np.dot(ilat, mode[-3:]) * jacob
    pos1 = cell1.get_cart_positions()
    pos2 = cell2.get_cart_positions()
    for i in range(itin.nat):
        mode[i] = pos2[i] - pos1[i]
    try:
        mode = vunit(mode)
    except:
        mode = np.zeros((itin.nat + 3, 3))
    return mode

def lower_triangular_cell(atoms):
    """
    Return a copy of `atoms` whose cell has zeros above the diagonal
    and write it to `out_file` (VASP / POSCAR) if a name is given.
    """
    old_cell = atoms.cell.array          # 3×3 matrix, row vectors a1,a2,a3
    a1 = old_cell[0]

    # --- orthonormal basis tied to a1 & a2 -------------------------------
    u1 = a1 / np.linalg.norm(a1)

    # remove any a1-component from a2 so new a2 lies in the x-y plane
    v2  = old_cell[1] - np.dot(old_cell[1], u1) * u1
    u2  = v2 / np.linalg.norm(v2)

    # u3 completes the right-handed set
    u3  = np.cross(u1, u2)

    # --- build the new lower-triangular lattice --------------------------
    # Dot each original lattice vector onto the orthonormal set
    a  = np.linalg.norm(a1)                # |a1|  → first row length
    bx = np.dot(old_cell[1], u1)
    by = np.dot(old_cell[1], u2)
    cx = np.dot(old_cell[2], u1)
    cy = np.dot(old_cell[2], u2)
    cz = np.dot(old_cell[2], u3)

    new_cell = np.array([[a,  0.0, 0.0],
                         [bx, by,  0.0],
                         [cx, cy,  cz ]])

    # --- transform atomic positions --------------------------------------
    # cart = atoms.get_positions()                         # N×3 Cartesian
    # frac = np.linalg.solve(new_cell.T, cart.T).T         # → fractional in new cell

    # new_atoms = atoms.copy()
    # new_atoms.set_cell(new_cell, scale_atoms=True)

    return new_cell