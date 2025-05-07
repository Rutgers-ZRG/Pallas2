#!/usr/bin/env python

import time
import copy
import numpy as np
import sys
import os

from math import sqrt, atan, cos, sin, tan, pi
#from ase  import atoms, units, io
from util import vunit, vmag, vrand

# Constants for numerical stability and algorithm parameters
ROTATION_ANGLE = 0.02  # Small rotation angle for projecting out rotational modes
BFGS_RESET_THRESHOLD = 0.05  # Threshold for BFGS reset in rotation_plane
BFGS_INITIAL_HESSIAN_FACTOR = 60.0  # Initial scaling factor for BFGS Hessian
MIN_DENOMINATOR = 1e-10  # Minimum denominator value to prevent division by zero

class SSDimer_atoms:

    def __init__(self, R0 = None, mode = None, maxStep = 0.2, dT = 0.1, dR = 0.001, 
                 phi_tol = 5, rotationMax = 4, ss = False, express=np.zeros((3,3)), 
                 rotationOpt = 'cg', weight = 1, noZeroModes = True):
        """
        Initialize the Solid-State Dimer method for transition state search.
        
        Parameters:
        R0      - an atoms object, which gives the starting point
        mode    - initial mode (will be randomized if one is not provided)
        maxStep - longest distance dimer can move in a single iteration
        dT      - quickmin timestep
        dR      - separation between the two images for rotation
        phi_tol - rotation converging tolerence, degree
        rotationMax - max rotations per translational step
        ss      - boolean, solid-state dimer or not
        express - 3*3 matrix, external stress tensor. Columns are the stress vectors. 
                 Needs to be in lower triangular form to avoid rigid rotation. 
        rotation_opt - the optimization method for the rotation part: 
                      choose from "sd" (steepest descent), "cg" (conjugate gradient), and "bfgs".
        noZeroModes  - boolean, project out the six zero modes or not. 
                      For some 2D analytical potential, it needs to be False.
        weight  - extra weight to put on the cell degrees of freedom.
                 
        """
        self.steps = 0
        self.dT = dT
        self.dR = dR
        self.phi_tol = phi_tol /180.0 * pi  # Convert degrees to radians
        self.R0 = R0
        self.N = mode
        self.natom = len(self.R0)
        
        # Initialize mode vector if not provided
        if self.N is None:
            print("Initialize mode randomly")
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
        self.noZeroModes = noZeroModes
        self.ss = ss
        self.express = express
        
        # Initialize rotation optimization variables
        self.T = np.zeros((self.natom+3, 3))
        self.Tnorm = 0.0
   
        # Set up cell parameters for solid-state calculations
        vol = self.R0.get_volume()
        avglen = (vol/self.natom)**(1.0/3.0)
        self.weight = weight
        self.jacobian = avglen * self.natom**0.5 * self.weight

        # Initialize BFGS matrices if needed
        if self.rotationOpt == 'bfgs':
            ndim = (self.natom + 3)*3
            self.Binv0 = np.eye(ndim) / BFGS_INITIAL_HESSIAN_FACTOR
            self.Binv = self.Binv0
            self.B0 = np.eye(ndim) * BFGS_INITIAL_HESSIAN_FACTOR
            self.B = self.B0


    # Pipe all the stuff from Atoms that is not overwritten.
    # Pipe all requests for get_original_* to self.atoms0.
    def __getattr__(self, attr):
        """Return any value of the Atoms object"""
        return getattr(self.R0, attr)

    def __len__(self):
        """Return the number of atoms (or atoms+3 for solid state)"""
        if self.ss:
            return self.natom+3
        else:
            return self.natom

    def get_positions(self):
        """
        Get positions in the generalized coordinate space.
        For solid state, returns zeros to make vector operations work properly.
        """
        r = self.R0.get_positions()
        if self.ss:
            # Return zeros, so the vector passed to set_positions is just dr in the generalized space. 
            # Otherwise, "+" and "-" operations for position vectors need to be redefined in the optimizer.
            # This trick only works for first order optimizers for sure, where no operation is applied 
            # to position vectors until the last update.
            Rc = np.vstack((r*0.0, self.R0.get_cell()*0.0))
            return Rc
        else:
            return r

    def set_positions(self, dr):
        """
        Set positions in the generalized coordinate space.
        For solid state, updates both atomic positions and cell vectors.
        """
        if self.ss:
            rcell = self.R0.get_cell()
            rcell += np.dot(rcell, dr[-3:]) / self.jacobian
            self.R0.set_cell(rcell, scale_atoms=True)
            ratom = self.R0.get_positions() + dr[:-3]
            self.R0.set_positions(ratom)
        else:
            # get_positions() returns non-zero values
            # thus this dr is the final positions, not just dr
            ratom = dr
            self.R0.set_positions(ratom)
    
    def update_general_forces(self, Ri):
        """
        Update the generalized forces (atomic forces, stress tensor).
        For solid state, combines forces and stress into a single vector.
        
        Parameters:
        Ri - Atoms object to get forces for
        
        Returns:
        Fc - Combined forces vector (natom+3, 3) including stress components
        """
        self.forceCalls += 1
        f = Ri.get_forces()
        vol = -Ri.get_volume()
        st = np.zeros((3,3))
        
        if self.ss:
            stt = Ri.get_stress()
            #following the order of get_stress in vasp.py
            #(the order of stress in ase are the same for all calculators)
            st[0][0] = stt[0] * vol  
            st[1][1] = stt[1] * vol
            st[2][2] = stt[2] * vol
            st[2][1] = stt[3] * vol
            st[2][0] = stt[4] * vol
            st[1][0] = stt[5] * vol
            st -= self.express * (-1)*vol
            
        Fc = np.vstack((f, st/self.jacobian))
        return Fc
  
    def get_curvature(self):
        """Return the curvature along the dimer axis."""
        return self.curvature

    def get_mode(self):
        """Return the current dimer mode (lowest curvature direction)."""
        if self.ss:
            return self.N
        else:
            return self.N[:-3]

    def get_forces(self):
        """
        Calculate forces for dimer method.
        Implements the modified force for climbing along the lowest curvature mode.
        
        Returns:
        Ftrans - Modified force vector for dimer translation
        """
        # Find the minimum mode direction and calculate forces
        F0 = self.minmodesearch()
        self.F0 = F0
        
        # Project forces along the dimer direction
        Fparallel = np.vdot(F0, self.N) * self.N
        Fperp = F0 - Fparallel
        
        # Calculate perpendicular component ratio for debugging
        alpha = vmag(Fperp)/vmag(F0)
        print("alpha=vmag(Fperp)/vmag(F0): ", alpha)
        print("curvature:", self.get_curvature())
        
        # Fixed scaling factor for parallel component
        gamma = 1.0

        # Determine the translation force based on curvature
        if self.curvature > 0:
            # If curvature is positive, we're at a minimum along N so invert the parallel component
            self.Ftrans = -1 * Fparallel
            print("drag up directly")
        else:
            # At a saddle point, modify the force to follow the minimum mode
            self.Ftrans = Fperp - gamma * Fparallel
            
        # Return full or truncated force vector depending on ss mode
        if self.ss:
            return self.Ftrans
        else:
            return self.Ftrans[:-3]

    def project_translt_rott(self, N, R0):
        """
        Project out rigid translation and rotation modes from vector N.
        These zero-energy modes should not be considered when searching for saddle points.
        
        Parameters:
        N  - Vector to be projected (natom+3, 3)
        R0 - Reference Atoms object
        
        Returns:
        N  - Vector with translation and rotation components removed
        """
        if not self.noZeroModes: 
            return N
            
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
            ptmp.rotate(axisx, ROTATION_ANGLE, center='COM', rotate_cell=False)
            rottVec = ptmp.get_positions() - R0.get_positions()
            rottVec = vunit(rottVec)
            N[:-3] -= np.vdot(N[:-3], rottVec)*rottVec
            
        return N
 
    def iset_endpoint_pos(self, Ni, R0, Ri):
        """
        Set the position of an endpoint (Ri) displaced from R0 along direction Ni.
        
        Parameters:
        Ni - Unit vector of displacement direction
        R0 - Reference configuration
        Ri - Endpoint to be updated
        """
        # Compute displacement vector with given magnitude
        dRvec = self.dR * Ni
        
        # Update cell for solid state calculations
        cell0 = R0.get_cell()
        cell1 = cell0 + np.dot(cell0, dRvec[-3:]) / self.jacobian
        Ri.set_cell(cell1, scale_atoms=True)
        
        # Update atomic positions
        vdir = R0.get_scaled_positions()
        ratom = np.dot(vdir, cell1) + dRvec[:-3]
        Ri.set_positions(ratom)
 
    def rotation_update(self):
        """
        Update the position of R1 and compute its forces.
        
        Returns:
        F1 - Forces at the R1 endpoint
        """
        # position of R1
        self.iset_endpoint_pos(self.N, self.R0, self.R1)
        # force of R1
        F1 = self.update_general_forces(self.R1)
        return F1
        
    def rotation_plane(self, Fperp, Fperp_old, Nold):
        """
        Determine the rotation direction T in the plane perpendicular to N.
        Uses one of several optimization methods (sd, cg, or bfgs).
        
        Parameters:
        Fperp     - Current perpendicular force
        Fperp_old - Previous perpendicular force
        Nold      - Previous dimer direction
        """
        if self.rotationOpt == 'sd':
            # Steepest descent: simply rotate toward the perpendicular force
            self.T = vunit(Fperp)
            
        elif self.rotationOpt == 'cg':
            # Conjugate gradient method for rotation
            a = abs(np.vdot(Fperp, Fperp_old))
            b = np.vdot(Fperp_old, Fperp_old)
            
            # Calculate the conjugate direction mixing parameter gamma
            if a <= 0.5*b and b > MIN_DENOMINATOR:  
                gamma = np.vdot(Fperp, Fperp-Fperp_old) / b
            else:
                gamma = 0
                
            # Compute new rotation direction with CG update
            Ttmp = Fperp + gamma * self.T * self.Tnorm
            Ttmp = Ttmp - np.vdot(Ttmp, self.N) * self.N  # Keep T perpendicular to N
            self.Tnorm = np.linalg.norm(Ttmp)
            self.T = vunit(Ttmp)
            
        elif self.rotationOpt == 'bfgs':
            # BFGS optimization for rotation direction
            Binv = self.Binv
            
            # Calculate step and gradient vectors for BFGS update
            s = (self.N - Nold).flatten() 
            g1 = -Fperp.flatten() / self.dR
            g0 = -Fperp_old.flatten() / self.dR
            y = g1 - g0
            
            # Update Hessian (not inverse Hessian) for better numerical stability
            a = np.dot(s, y)
            dg = np.dot(self.B, s)
            b = np.dot(s, dg)
            
            # Skip update if denominators are too small
            if abs(a) > MIN_DENOMINATOR and abs(b) > MIN_DENOMINATOR:
                self.B += np.outer(y, y) / a - np.outer(dg, dg) / b
                
            # Eigendecomposition to compute search direction
            omega, V = np.linalg.eigh(self.B)
            dr = np.dot(V, np.dot(-g1, V) / np.fabs(omega)).reshape((-1, 3))
            
            # Check for alignment between BFGS direction and steepest descent
            vd = np.vdot(vunit(dr), vunit(Fperp))
            if vd < BFGS_RESET_THRESHOLD:  
                # Reset BFGS if direction is suspicious
                dr = Fperp
                self.B = self.B0
                
            # Ensure rotation direction is perpendicular to dimer axis
            dr -= np.vdot(dr, self.N) * self.N
            self.T = vunit(dr)
        
    def minmodesearch(self):
        """
        Find the minimum curvature mode by rotating the dimer.
        
        Returns:
        F0 - Forces at the central point
        """
        # Project out any rigid translation and rotation
        if not self.ss: 
            self.N = self.project_translt_rott(self.N, self.R0)

        # Get initial forces
        F0 = self.update_general_forces(self.R0)
        F1 = self.rotation_update()

        # Initialize rotation parameters
        phi_min = 1.5  # Initial value > phi_tol to enter loop
        Fperp = F1 * 0.0  # Initialize to avoid assignment error
        iteration = 0
        
        # Main rotation loop
        while abs(phi_min) > self.phi_tol and iteration < self.rotationMax:
            # First iteration: compute perpendicular forces and initial rotation direction
            if iteration == 0: 
                F0perp = F0 - np.vdot(F0, self.N) * self.N
                F1perp = F1 - np.vdot(F1, self.N) * self.N
                # Factor of 2.0 comes from the dimer method formulation
                Fperp = 2.0 * (F1perp - F0perp)
                self.T = vunit(Fperp)

            # Project out any rigid translation and rotation from rotation direction
            if not self.ss: 
                self.T = self.project_translt_rott(self.T, self.R0)

            # Calculate curvature and its derivative
            c0 = np.vdot(F0-F1, self.N) / self.dR  # Curvature along N
            c0d = np.vdot(F0-F1, self.T) / self.dR * 2.0  # Derivative of curvature wrt rotation
            
            # Calculate initial rotation angle to approximately minimize curvature
            # Use epsilon in denominator to prevent division by zero
            phi_1 = -0.5 * atan(c0d / (2.0 * max(abs(c0), MIN_DENOMINATOR)))
            
            # Early exit if rotation angle is already below tolerance
            if abs(phi_1) <= self.phi_tol: 
                break

            # Calculate forces after rotating by phi_1
            N1_prime = vunit(self.N * cos(phi_1) + self.T * sin(phi_1))
            self.iset_endpoint_pos(N1_prime, self.R0, self.R1_prime)
            F1_prime = self.update_general_forces(self.R1_prime)
            c0_prime = np.vdot(F0-F1_prime, N1_prime) / self.dR 
            
            # Calculate optimal rotation angle using interpolation model
            # Model curvature as: c(φ) = a0/2 + a1*cos(2φ) + b1*sin(2φ)
            b1 = 0.5 * c0d
            a1 = (c0 - c0_prime + b1 * sin(2 * phi_1)) / max(1 - cos(2 * phi_1), MIN_DENOMINATOR)
            a0 = 2 * (c0 - a1)
            phi_min = 0.5 * atan(b1 / max(a1, MIN_DENOMINATOR))
            c0_min = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)

            # Check whether it is minimum or maximum by comparing curvatures
            if c0_min > c0:
                phi_min += pi * 0.5
                c0_min = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)
                
            # Normalize angle for BFGS accuracy
            if phi_min > pi * 0.5: 
                phi_min -= pi
                 
            # Update dimer direction
            Nold = self.N
            self.N = vunit(self.N * cos(phi_min) + self.T * sin(phi_min))
            
            # Project out any rigid translation and rotation
            if not self.ss: 
                self.N = self.project_translt_rott(self.N, self.R0)
                
            # Update curvature
            c0 = c0_min

            # Update F1 by linear extrapolation to avoid another force calculation
            # This formula derives from the dimer method equations for force interpolation
            F1 = F1 * (sin(phi_1 - phi_min) / max(sin(phi_1), MIN_DENOMINATOR)) + \
                 F1_prime * (sin(phi_min) / max(sin(phi_1), MIN_DENOMINATOR)) + \
                 F0 * (1.0 - cos(phi_min) - sin(phi_min) * tan(phi_1 * 0.5))

            # Calculate perpendicular forces for next iteration
            F0perp = F0 - np.vdot(F0, self.N) * self.N
            F1perp = F1 - np.vdot(F1, self.N) * self.N
            Fperp_old = Fperp
            Fperp = 2.0 * (F1perp - F0perp)
            
            # Update rotation direction using chosen optimization method
            self.rotation_plane(Fperp, Fperp_old, Nold)

            iteration += 1
            
        # Store final curvature and return forces
        self.curvature = c0
        return F0
        
