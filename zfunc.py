from dimer import SolidStateDimer
# Removing util import and adding vector utility functions directly
import os
import sys
import numpy as np

import pickle
from ase.io import read, write, Trajectory
from ase.optimize import FIRE, BFGS
from ase.filters import FrechetCellFilter
from copy import deepcopy as cp


def vunit(v):
    mag = np.sqrt(np.vdot(v, v))
    if mag == 0:
        return v
    return v / mag

def vrand(v):
    vtemp = np.random.randn(v.size)
    return vtemp.reshape(v.shape)

from mattersim.forcefield import MatterSimCalculator

device = "cpu"
calc = MatterSimCalculator(device=device)


def local_optimization(patoms):
    # atoms = cp(patoms)
    atoms=patoms
    atoms.calc = calc
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf, maxstep=0.1, logfile='opt.log')
    opt.run(fmax=0.001, steps=2000)
    fmax = np.max(np.abs(ecf.get_forces()))
    if fmax > 0.001:
        print("Warning: fmax = ", fmax)
        atoms.converged = False
    else:
        atoms.converged = True
    
    new_cell = lower_triangular_cell(atoms)
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.cal_fp()
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
    d = SolidStateDimer(atoms, mode = mode)

    # use FIRE optimizer in ase
    dyn = FIRE(d, maxstep=0.1, logfile='ssdimer.log')
    dyn.run(fmax=0.01, steps=2000)
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


    return new_cell