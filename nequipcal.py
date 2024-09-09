from tsase.dimer import ssdimer
from tsase.neb.util import vunit, vrand
from ase.io import read, write
import os
import sys
import numpy as np


import pickle
import numpy as np
from nequip.ase.nequip_calculator import nequip_calculator
from ase.io import read, write, Trajectory
from ase.optimize import FIRE, BFGS
from ase.constraints import ExpCellFilter
# from ase.filters import UnitCellFilter
from custom_calc_mixed import Custom_Nequip_Calc
from copy import deepcopy as cp


nequip_ef_path = "model/cdse_energy_force_model.pth"
nequip_s_path = "model/cdse_stress_model.pth"

calc  = Custom_Nequip_Calc(nequip_ef_calc_path=nequip_ef_path, nequip_s_calc_path=nequip_s_path)


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
    atoms.cal_fp()
    return atoms