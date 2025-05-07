#!/usr/bin/env python

import time
import copy
import numpy as np
import sys
import os

from math import sqrt, atan, cos, sin, tan, pi
from ase  import atoms, units, io
from util import vunit, vmag, vrand, sPBC

class SSDimer_atoms:

    def __init__(self, R0 = None, mode = None, maxStep = 0.2, dT = 0.1, dR = 0.001, 
                 phi_tol = 5, rotationMax = 4, ss = False, express=np.zeros((3,3)), 
                 rotationOpt = 'cg', weight = 1, noZeroModes = True):
        """
        Parameters:
        R0      - an atoms object, which gives the starting point
        mode    - initial mode (will be randomized if one is not provided)
        maxStep - longest distance dimer can move in a single iteration
        dT      - quickmin timestep
        dR      - separation between the two images for rotation
        phi_tol - rotation converging tolerence, degree
        rotationMax - max rotations per translational step
        ss      - boolean, solid-state dimer or not
        express - 3*3 matrix, external stress tensor. Columns are the stress vectors. Needs to be in lower triangular form to avoid rigid rotation. 
        rotation_opt - the optimization method for the rotation part: choose from "sd" (steepest descent), "cg" (conjugate gradient), and "bfgs".
        noZeroModes  - boolean, project out the six zero modes or not. For some 2D analytical potential, it needs to be False.
        weight  - extra weight to put on the cell degrees of freedom.
                 
        """
        self.steps = 0
        self.dT = dT
        self.dR = dR
        self.phi_tol = phi_tol /180.0 * pi
        self.R0    = R0
        self.N = mode
        self.natom  = len(self.R0)
        if self.N is None:
            print("radomly initialize the lowest eigenvector")
            self.N = vrand(np.zeros((self.natom+3,3)))
            self.N[-3:] *= 0.0
        elif len(self.N) == self.natom: 
            self.N = np.vstack(( mode, np.zeros((3,3)) ))
        self.N = vunit(self.N)
        self.maxStep = maxStep
        self.Ftrans = None
        self.forceCalls = 0
        self.R1 = self.R0.copy()
        self.R1_prime = self.R0.copy()
        calc = self.R0.calc
        self.R1.calc = calc
        self.R1_prime.calc = calc
        self.rotationMax = rotationMax
        self.rotationOpt = rotationOpt
        self.noZeroModes = noZeroModes # Set to False for 2D model potentials
        self.ss       = ss
        self.express  = express
   
        vol           = self.R0.get_volume()
        avglen        = (vol/self.natom)**(1.0/3.0)
        self.weight   = weight
        self.jacobian = avglen * self.natom**0.5 * self.weight

        if self.rotationOpt == 'bfgs':
            ## BFGS for rotation: initial (inverse) Hessian
            ndim = (self.natom + 3)*3
            self.Binv0= np.eye(ndim) / 60
            self.Binv = self.Binv0
            self.B0   = np.eye(ndim) * 60
            self.B    = self.B0


    # Pipe all the stuff from Atoms that is not overwritten.
    # Pipe all requests for get_original_* to self.atoms0.
    def __getattr__(self, attr):
        """Return any value of the Atoms object"""
        return getattr(self.R0, attr)
        # print(attr)

    def __len__(self):
        if self.ss:
            return self.natom+3
        else:
            return self.natom

    def get_positions(self):
        r    = self.R0.get_positions()
        if self.ss:
            # return zeros, so the vector passed to set_positions is just dr in the generalized space. 
            # otherwise, "+" and "-" operations for position vectors need to be redefined in the optimizer
            # this trick only works for first order optimzers for sure, where no operation applied to position vectors until the last update.
            Rc   = np.vstack((r*0.0, self.R0.get_cell()*0.0))
            return Rc
        else:
            return r

    def set_positions(self,dr):
        if self.ss:
            rcell  = self.R0.get_cell()
            rcell += np.dot(rcell, dr[-3:]) / self.jacobian
            self.R0.set_cell(rcell, scale_atoms=True)
            ratom  = self.R0.get_positions() + dr[:-3]
            self.R0.set_positions(ratom)
        else:
            # get_positions() returns non-zero values
            # thus this dr is the final positions, not just dr
            ratom  = dr
            self.R0.set_positions(ratom)
    
    def update_general_forces(self, Ri):
        #update the generalized forces (f, st)
        self.forceCalls += 1
        f    = Ri.get_forces()
        if self.ss: stt  = Ri.get_stress()
        vol  = Ri.get_volume()*(-1)
        st   = np.zeros((3,3))
        #following the order of get_stress in vasp.py
        #(the order of stress in ase are the same for all calculators)
        if self.ss:
            st[0][0] = stt[0] * vol  
            st[1][1] = stt[1] * vol
            st[2][2] = stt[2] * vol
            st[2][1] = stt[3] * vol
            st[2][0] = stt[4] * vol
            st[1][0] = stt[5] * vol
            st  -= self.express * (-1)*vol
        Fc   = np.vstack((f, st/self.jacobian))
        return Fc
  
    def get_curvature(self):
        return self.curvature

    def get_mode(self):
        if self.ss:
            return self.N
        else:
            return self.N[:-3]

    def get_forces(self):
        F0        = self.minmodesearch()
        self.F0   = F0
        Fparallel = np.vdot(F0, self.N) * self.N
        Fperp     = F0-Fparallel
        alpha     = vmag(Fperp)/vmag(F0)
        print("alpha=vmag(Fperp)/vmag(F0): ", alpha)
        print("curvature:", self.get_curvature())
        gamma     = 1.0

        if self.curvature > 0:
            self.Ftrans = -1 * Fparallel
            print("drag up directly")
        else:
            self.Ftrans = Fperp - gamma * Fparallel
        if self.ss:
            return self.Ftrans
        else:
            return self.Ftrans[:-3]

    def project_translt_rott(self, N, R0):
        if not self.noZeroModes: return N
        # Project out rigid translational mode
        for axisx in range(3):
            transVec = np.zeros((self.natom+3, 3))
            transVec[:, axisx] = 1.0
            transVec = vunit(transVec)
            N -= np.vdot(N, transVec)*transVec
        # Project out rigid rotational mode
        for axisx in ['x', 'y', 'z']:
            ptmp = R0.copy()
            # rotate a small angle around the center of mass
            ptmp.rotate(axisx, 0.02, center='COM', rotate_cell=False)
            rottVec = ptmp.get_positions() - R0.get_positions()
            rottVec = vunit(rottVec)
            #if np.vdot(N[:-3], rottVec) > 0.1: print "mainly rotation around "+axisx
            N[:-3] -= np.vdot(N[:-3], rottVec)*rottVec
        return N
 
    def iset_endpoint_pos(self, Ni, R0, Ri):
        # update the position of Ri
        dRvec = self.dR * Ni
        cell0 = R0.get_cell()
        cell1 = cell0 + np.dot(cell0, dRvec[-3:]) / self.jacobian
        Ri.set_cell(cell1, scale_atoms=True)
        vdir  = R0.get_scaled_positions()
        ratom = np.dot(vdir, cell1) + dRvec[:-3]
        Ri.set_positions(ratom)
 
    def rotation_update(self):
        # position of R1
        self.iset_endpoint_pos(self.N, self.R0, self.R1)
        # force of R1
        F1    = self.update_general_forces(self.R1)
        return F1
        
    def rotation_plane(self, Fperp, Fperp_old, Nold):
        if self.rotationOpt == 'sd':
            self.T     = vunit(Fperp)
        elif self.rotationOpt == 'cg':
            # determine self.T (big theta in the paper) with CG method
            a = abs(np.vdot(Fperp, Fperp_old))
            b = np.vdot(Fperp_old, Fperp_old)
            if a <= 0.5*b and b != 0:  
                gamma = np.vdot(Fperp, Fperp-Fperp_old) / b
            else:
                gamma = 0
            try:    self.Tnorm 
            except: self.Tnorm = 0.0
            Ttmp       = Fperp + gamma * self.T * self.Tnorm
            Ttmp       = Ttmp - np.vdot(Ttmp, self.N) * self.N
            self.Tnorm = np.linalg.norm(Ttmp)
            self.T     = vunit(Ttmp)
        elif self.rotationOpt == 'bfgs':
            ## BFGS from wiki
            Binv    =  self.Binv
            #s       =  (self.N - Nold).flatten()  * self.dR
            #g1      =  -Fperp.flatten() 
            #g0      =  -Fperp_old.flatten() 
            s       =  (self.N - Nold).flatten() 
            g1      =  -Fperp.flatten() / self.dR
            g0      =  -Fperp_old.flatten() / self.dR
            y       =  g1 - g0
            '''
            ## updating the inverse Hessian
            a = np.dot(s, y)
            b = np.dot(y, Binv.dot(y))
            c = np.outer(s, s)
            d = np.outer(y, s) 
            print "a, b:", a, b
            Binv      = Binv + (a+b) * c / a**2 - (Binv.dot(d) + d.T.dot(Binv)) / a
            self.Binv = Binv
            dr        = (Binv.dot(-g1)).reshape((-1, 3))
            '''
            ## updating Hessian rather than the inverse Hessian, see ase.optimize.BFGS
            a  = np.dot(s, y)
            dg = np.dot(self.B, s)
            b  = np.dot(s, dg)
            #print "a, b", a, b
            self.B  += np.outer(y, y) / a - np.outer(dg, dg) / b
            omega, V = np.linalg.eigh(self.B)
            dr       = np.dot(V, np.dot(-g1, V) / np.fabs(omega)).reshape((-1, 3))
            vd       = np.vdot(vunit(dr), vunit(Fperp))
            if vd < 0.05:  
                #print "////reset BFGS in rotation////"
                dr = Fperp
                self.B    = self.B0
            dr    -= np.vdot(dr, self.N) * self.N
            self.T = vunit(dr)
        
    def minmodesearch(self):
        # rotate dimer to the minimum mode direction
        # self.N, the dimer direction; 
        # self.T, the rotation direction, spans the rotation plane with self.N.

        # project out any rigid translation and rotation
        if not self.ss: self.N = self.project_translt_rott(self.N, self.R0)

        F0    = self.update_general_forces(self.R0)
        F1    = self.rotation_update()

        phi_min = 1.5
        Fperp   = F1 * 0.0 # avoid Fperp_old asignment error
        iteration = 0
        while abs(phi_min) > self.phi_tol and iteration < self.rotationMax:

            if iteration == 0: 
                F0perp    = F0 - np.vdot(F0, self.N) * self.N
                F1perp    = F1 - np.vdot(F1, self.N) * self.N
                Fperp     = 2.0 * (F1perp - F0perp)
                self.T    = vunit(Fperp)

            # project out any rigid translation and rotation
            if not self.ss: self.T = self.project_translt_rott(self.T, self.R0)

            # curvature and its derivative
            c0     = np.vdot(F0-F1, self.N) / self.dR
            c0d    = np.vdot(F0-F1, self.T) / self.dR * 2.0
            phi_1  = -0.5 * atan(c0d / (2.0 * abs(c0)))
            if abs(phi_1) <= self.phi_tol: break

            # calculate F_prime: force after rotating the dimer by phi_prime
            N1_prime = vunit(self.N * cos(phi_1) + self.T * sin(phi_1))
            self.iset_endpoint_pos(N1_prime, self.R0, self.R1_prime)
            F1_prime = self.update_general_forces(self.R1_prime)
            c0_prime = np.vdot(F0-F1_prime, N1_prime) / self.dR 
            
            # calculate phi_min
            b1 = 0.5 * c0d
            a1 = (c0 - c0_prime + b1 * sin(2 * phi_1)) / (1 - cos(2 * phi_1))
            a0 = 2 * (c0 - a1)
            phi_min = 0.5 * atan(b1 / a1)
            c0_min  = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)

            # check whether it is minimum or maximum
            if c0_min > c0 :
                phi_min += pi * 0.5
                c0_min   = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)
            ## for accurate BFGS s  
            if phi_min > pi * 0.5: phi_min -= pi
                 
            # update self.N
            Nold   = self.N
            self.N = vunit(self.N * cos(phi_min) + self.T * sin(phi_min))
            # project out any rigid translation and rotation
            if not self.ss: self.N = self.project_translt_rott(self.N, self.R0)
            c0     = c0_min

            # update F1 by linear extropolation
            F1 = F1 * (sin(phi_1 - phi_min) / sin(phi_1)) + F1_prime * (sin(phi_min) / sin(phi_1)) \
                 + F0 * (1.0 - cos(phi_min) - sin(phi_min) * tan(phi_1 * 0.5))

            F0perp    = F0 - np.vdot(F0, self.N) * self.N
            F1perp    = F1 - np.vdot(F1, self.N) * self.N
            Fperp_old = Fperp
            Fperp     = 2.0 * (F1perp - F0perp)
            # update self.T
            self.rotation_plane(Fperp, Fperp_old, Nold)

            iteration += 1
        self.curvature = c0
        return F0
        
