#!/usr/bin/env python
"""
Solid State Dimer Method for transition state searches.

This module provides an implementation of the dimer method for finding
transition states on potential energy surfaces, with extensions for
solid-state systems allowing cell optimization.

This implementation is derived from the TSASE (Transition State Library for ASE) 
package but has been significantly refactored and optimized.
"""

import numpy as np
from math import sqrt, atan, cos, sin, tan, pi
from typing import Optional, Union, Tuple, List, Any

# Numerical constants
ROTATION_ANGLE = 0.02  # Small rotation angle for projecting out rotational modes
BFGS_RESET_THRESHOLD = 0.05  # Threshold for BFGS reset in rotation_plane
BFGS_INITIAL_SCALE = 60.0  # Initial scaling factor for BFGS Hessian
EPSILON = 1e-10  # Small value to prevent division by zero
MAX_ROTATION_ITERATIONS = 10  # Maximum number of rotations in a single step

class SolidStateDimer:
    """
    Implementation of the Solid State Dimer method for transition state search.
    
    The dimer method is a minimum mode following algorithm that efficiently
    searches for transition states without requiring calculation of the Hessian matrix.
    This implementation extends the original dimer method to handle periodic systems
    with variable cell shapes.
    
    This class is derived from the TSASE (Transition State Library for ASE)
    package but has been significantly refactored and optimized.
    """

    def __init__(self, atoms=None, mode=None, max_step=0.2, time_step=0.1, 
                 dimer_separation=0.001, rotation_tolerance=15.0, max_rotations=4, 
                 solid_state=True, external_stress=None, rotation_method='cg', 
                 cell_weight=1.0, project_translations_rotations=True):
        """
        Initialize the Solid State Dimer method.
        
        Parameters
        ----------
        atoms : Atoms object
            The atomic system to perform the search on
        mode : ndarray, optional
            Initial mode for the dimer direction (will be randomized if not provided)
        max_step : float, default=0.2
            Maximum step size for translation
        time_step : float, default=0.1
            Time step for dynamics (used in QuickMin/Lanczos)
        dimer_separation : float, default=0.001
            Separation between the two dimer images
        rotation_tolerance : float, default=5.0
            Convergence tolerance for dimer rotation (degrees)
        max_rotations : int, default=4
            Maximum number of rotation steps per translation step
        solid_state : bool, default=False
            Whether to perform solid-state dimer (with cell optimization)
        external_stress : ndarray, optional
            External stress tensor (3x3 matrix, must be in lower triangular form)
        rotation_method : str, default='cg'
            Method for optimizing rotation: 'sd' (steepest descent), 
            'cg' (conjugate gradient), or 'bfgs'
        cell_weight : float, default=1.0
            Weight factor for cell degrees of freedom
        project_translations_rotations : bool, default=True
            Whether to project out translational and rotational modes
        """
        # Store reference to atoms object
        self.atoms = atoms
        self.num_atoms = len(atoms) if atoms is not None else 0
        
        # Dimer parameters
        self.max_step = max_step
        self.time_step = time_step
        self.dimer_separation = dimer_separation
        self.rotation_tolerance = rotation_tolerance * pi / 180.0  # Convert to radians
        self.max_rotations = max_rotations
        
        # System parameters
        self.solid_state = solid_state
        self.external_stress = np.zeros((3, 3)) if external_stress is None else external_stress
        self.cell_weight = cell_weight
        self.project_translations_rotations = project_translations_rotations
        
        # Force evaluation counter
        self.force_evaluations = 0
        
        # Initialize mode vector
        self._initialize_mode(mode)
        
        # Initialize rotation parameters
        self.rotation_method = rotation_method
        self.rotation_direction = np.zeros_like(self.mode)
        self.rotation_direction_norm = 0.0
        
        # Initialize cell parameters for solid state
        self._setup_cell_parameters()
        
        # Initialize dimer images (replicas of atoms)
        self._initialize_dimer_images()
        
        # Initialize optimization state
        self.curvature = None
        self.modified_forces = None
        
        # BFGS specific parameters
        if self.rotation_method == 'bfgs':
            self._setup_bfgs()

    def _setup_bfgs(self):
        """Initialize matrices for BFGS optimization method."""
        ndim = (self.num_atoms + 3) * 3 if self.solid_state else self.num_atoms * 3
        
        # Initial Hessian and its inverse
        self.bfgs_hessian_0 = np.eye(ndim) * BFGS_INITIAL_SCALE
        self.bfgs_hessian = self.bfgs_hessian_0.copy()
        
        self.bfgs_inv_hessian_0 = np.eye(ndim) / BFGS_INITIAL_SCALE
        self.bfgs_inv_hessian = self.bfgs_inv_hessian_0.copy()

    def _initialize_mode(self, mode):
        """Initialize the dimer mode (direction)."""
        if self.solid_state:
            mode_shape = (self.num_atoms + 3, 3)
        else:
            mode_shape = (self.num_atoms, 3)
            
        if mode is None:
            # Create random initial mode
            self.mode = self._random_vector(np.zeros(mode_shape))
            if self.solid_state:
                # Zero out cell components initially
                self.mode[-3:] = 0.0
        elif len(mode) == self.num_atoms and self.solid_state:
            # Extend mode with zeros for cell vectors
            self.mode = np.vstack((mode, np.zeros((3, 3))))
        else:
            self.mode = mode.copy()
            
        # Normalize the mode vector
        self.mode = self._normalize_vector(self.mode)

    def _setup_cell_parameters(self):
        """Set up parameters related to cell degrees of freedom."""
        if not self.solid_state or self.atoms is None:
            self.jacobian = 1.0
            return
            
        # Calculate Jacobian factor for cell degrees of freedom
        vol = self.atoms.get_volume()
        avg_length = (vol / self.num_atoms) ** (1.0 / 3.0)
        self.jacobian = avg_length * self.num_atoms**0.5 * self.cell_weight

    def _initialize_dimer_images(self):
        """Initialize the dimer endpoint images."""
        if self.atoms is None:
            return
            
        # Create copies for dimer endpoints
        self.dimer_image = self.atoms.copy()
        self.trial_image = self.atoms.copy()
        
        # Ensure calculators are propagated
        calc = self.atoms.calc
        self.dimer_image.calc = calc
        self.trial_image.calc = calc

    def _vector_magnitude(self, vector):
        """
        Calculate the magnitude (Euclidean norm) of a vector.
        
        Parameters
        ----------
        vector : ndarray
            Input vector or array
            
        Returns
        -------
        float
            Magnitude of the vector
        """
        return np.sqrt(np.vdot(vector, vector))

    def _normalize_vector(self, vector):
        """
        Normalize a vector to unit length.
        
        Parameters
        ----------
        vector : ndarray
            Input vector or array
            
        Returns
        -------
        ndarray
            Normalized vector with unit length
        """
        magnitude = self._vector_magnitude(vector)
        if magnitude < EPSILON:
            return vector
        return vector / magnitude

    def _random_vector(self, shape_like):
        """
        Generate a random unit vector with the same shape as the input.
        
        Parameters
        ----------
        shape_like : ndarray
            Array with the desired shape
            
        Returns
        -------
        ndarray
            Random unit vector with the same shape
        """
        random_vec = np.random.randn(*shape_like.shape)
        return self._normalize_vector(random_vec)

    def get_positions(self):
        """
        Get positions in the generalized coordinate space.
        
        For solid state calculations, returns zeros to make vector operations
        work properly with the optimizer.
        
        Returns
        -------
        ndarray
            Positions array (or zeros for solid state)
        """
        positions = self.atoms.get_positions()
        
        if self.solid_state:
            # Return zeros so the vector passed to set_positions is just
            # the displacement in the generalized space
            padded_zeros = np.vstack((
                np.zeros_like(positions),
                np.zeros_like(self.atoms.get_cell())
            ))
            return padded_zeros
        else:
            return positions

    def set_positions(self, displacement):
        """
        Set positions in the generalized coordinate space.
        
        For solid state, updates both atomic positions and cell vectors.
        
        Parameters
        ----------
        displacement : ndarray
            Displacement vector in the generalized coordinate space
        """
        if self.solid_state:
            # Update cell
            cell = self.atoms.get_cell()
            cell += np.dot(cell, displacement[-3:]) / self.jacobian
            self.atoms.set_cell(cell, scale_atoms=True)
            
            # Update atom positions
            positions = self.atoms.get_positions() + displacement[:-3]
            self.atoms.set_positions(positions)
        else:
            # For non-solid-state, displacement is the final positions
            self.atoms.set_positions(displacement)

    def _update_dimer_endpoint(self, direction, reference, endpoint):
        """
        Set the position of a dimer endpoint.
        
        Parameters
        ----------
        direction : ndarray
            Unit vector defining the displacement direction
        reference : Atoms
            Reference atomic configuration
        endpoint : Atoms
            Endpoint to be updated
        """
        # Calculate displacement vector
        displacement = self.dimer_separation * direction
        
        # Update cell for solid state
        cell0 = reference.get_cell()
        if self.solid_state:
            cell1 = cell0 + np.dot(cell0, displacement[-3:]) / self.jacobian
            endpoint.set_cell(cell1, scale_atoms=True)
            
            # Update atom positions in new cell
            scaled_positions = reference.get_scaled_positions()
            positions = np.dot(scaled_positions, cell1) + displacement[:-3]
            endpoint.set_positions(positions)
        else:
            # For non-solid-state, just translate atoms
            positions = reference.get_positions() + displacement
            endpoint.set_positions(positions)

    def _calculate_general_forces(self, atoms_obj):
        """
        Calculate the general forces (atomic forces and stress).
        
        For solid state systems, combines atomic forces and stress
        into a single generalized force vector.
        
        Parameters
        ----------
        atoms_obj : Atoms
            Atoms object to calculate forces for
            
        Returns
        -------
        ndarray
            Combined forces array
        """
        self.force_evaluations += 1
        forces = atoms_obj.get_forces()
        
        if self.solid_state:
            # Get stress and convert to 3x3 matrix
            stress = atoms_obj.get_stress()
            volume = -atoms_obj.get_volume()
            
            stress_matrix = np.zeros((3, 3))
            # Convert from Voigt notation to 3x3 matrix
            stress_matrix[0, 0] = stress[0] * volume
            stress_matrix[1, 1] = stress[1] * volume
            stress_matrix[2, 2] = stress[2] * volume
            stress_matrix[2, 1] = stress[3] * volume
            stress_matrix[2, 0] = stress[4] * volume
            stress_matrix[1, 0] = stress[5] * volume
            
            # Apply external stress
            stress_matrix -= self.external_stress * (-1) * volume
            
            # Combine forces and stress
            return np.vstack((forces, stress_matrix / self.jacobian))
        else:
            return forces

    def _project_out_translations_rotations(self, vector, atoms_obj):
        """
        Project out rigid translation and rotation modes from a vector.
        
        Parameters
        ----------
        vector : ndarray
            Vector to project
        atoms_obj : Atoms
            Reference atoms object
            
        Returns
        -------
        ndarray
            Vector with translation and rotation components removed
        """
        if not self.project_translations_rotations:
            return vector
            
        # Project out translations (3 modes)
        for axis in range(3):
            trans_vec = np.zeros_like(vector)
            trans_vec[:self.num_atoms, axis] = 1.0
            trans_vec = self._normalize_vector(trans_vec)
            vector -= np.vdot(vector, trans_vec) * trans_vec
            
        # Project out rotations (3 modes)
        for axis in ['x', 'y', 'z']:
            # Create rotated structure
            rotated = atoms_obj.copy()
            rotated.rotate(axis, ROTATION_ANGLE, center='COM', rotate_cell=False)
            
            # Calculate rotation vector
            rot_vec = rotated.get_positions() - atoms_obj.get_positions()
            rot_vec = self._normalize_vector(rot_vec.reshape(-1, 3))
            
            # Only project from atomic part, not cell part
            if self.solid_state:
                projection = np.vdot(vector[:-3].reshape(-1), rot_vec.reshape(-1))
                rot_vec_full = np.zeros_like(vector)
                rot_vec_full[:-3] = rot_vec.reshape(vector[:-3].shape)
                vector -= projection * rot_vec_full
            else:
                projection = np.vdot(vector.reshape(-1), rot_vec.reshape(-1))
                vector -= projection * rot_vec.reshape(vector.shape)
                
        return vector

    def _update_rotation_direction(self, perp_force, old_perp_force, old_mode):
        """
        Determine the optimal rotation direction using the specified method.
        
        Parameters
        ----------
        perp_force : ndarray
            Current perpendicular force
        old_perp_force : ndarray
            Previous perpendicular force
        old_mode : ndarray
            Previous dimer direction
        """
        if self.rotation_method == 'sd':
            # Steepest descent: simply rotate toward the perpendicular force
            self.rotation_direction = self._normalize_vector(perp_force)
            
        elif self.rotation_method == 'cg':
            # Conjugate gradient method
            dot_product = abs(np.vdot(perp_force, old_perp_force))
            old_force_sq = np.vdot(old_perp_force, old_perp_force)
            
            # Calculate mixing parameter
            if dot_product <= 0.5 * old_force_sq and old_force_sq > EPSILON:
                gamma = np.vdot(perp_force, perp_force - old_perp_force) / old_force_sq
            else:
                gamma = 0.0
                
            # Update rotation direction
            mixed_direction = perp_force + gamma * self.rotation_direction * self.rotation_direction_norm
            
            # Ensure orthogonality to mode
            mixed_direction -= np.vdot(mixed_direction, self.mode) * self.mode
            
            self.rotation_direction_norm = self._vector_magnitude(mixed_direction)
            self.rotation_direction = self._normalize_vector(mixed_direction)
            
        elif self.rotation_method == 'bfgs':
            # BFGS optimization for rotation
            
            # Calculate step and gradient vectors
            step = (self.mode - old_mode).flatten()
            grad_new = -perp_force.flatten() / self.dimer_separation
            grad_old = -old_perp_force.flatten() / self.dimer_separation
            grad_diff = grad_new - grad_old
            
            # BFGS Hessian update
            dot1 = np.dot(step, grad_diff)
            hessian_step = np.dot(self.bfgs_hessian, step)
            dot2 = np.dot(step, hessian_step)
            
            # Skip update if denominators are too small
            if abs(dot1) > EPSILON and abs(dot2) > EPSILON:
                self.bfgs_hessian += np.outer(grad_diff, grad_diff) / dot1 - np.outer(hessian_step, hessian_step) / dot2
                
            # Compute search direction using eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(self.bfgs_hessian)
            search_dir = np.dot(
                eigenvectors, 
                np.dot(-grad_new, eigenvectors) / np.abs(eigenvalues)
            ).reshape(self.mode.shape)
            
            # Check alignment with steepest descent direction
            alignment = np.vdot(
                self._normalize_vector(search_dir),
                self._normalize_vector(perp_force)
            )
            
            # Reset BFGS if search direction is suspicious
            if alignment < BFGS_RESET_THRESHOLD:
                search_dir = perp_force
                self.bfgs_hessian = self.bfgs_hessian_0.copy()
                
            # Ensure search direction is perpendicular to mode
            search_dir -= np.vdot(search_dir, self.mode) * self.mode
            self.rotation_direction = self._normalize_vector(search_dir)

    def find_minimum_mode(self):
        """
        Find the minimum curvature mode by rotating the dimer.
        
        This is the core algorithm of the dimer method that identifies
        the direction of lowest curvature (negative eigenvalue) on the
        potential energy surface.
        
        Returns
        -------
        ndarray
            Forces at the central point
        """
        # Project out rigid translations and rotations if requested
        if self.project_translations_rotations and not self.solid_state:
            self.mode = self._project_out_translations_rotations(self.mode, self.atoms)

        # Calculate initial forces
        central_forces = self._calculate_general_forces(self.atoms)
        
        # Set up and get forces for first dimer endpoint
        self._update_dimer_endpoint(self.mode, self.atoms, self.dimer_image)
        dimer_forces = self._calculate_general_forces(self.dimer_image)

        # Initialize rotation parameters
        phi_min = 1.5  # Initial value > tolerance to enter loop
        perp_force = np.zeros_like(dimer_forces)  # Initialize to avoid reference error
        iteration = 0
        
        # Main rotation loop
        while abs(phi_min) > self.rotation_tolerance and iteration < self.max_rotations:
            # First iteration: compute initial perpendicular forces
            if iteration == 0:
                # Calculate perpendicular components
                central_perp = central_forces - np.vdot(central_forces, self.mode) * self.mode
                dimer_perp = dimer_forces - np.vdot(dimer_forces, self.mode) * self.mode
                
                # Effective perpendicular force (factor of 2.0 from dimer method theory)
                perp_force = 2.0 * (dimer_perp - central_perp)
                self.rotation_direction = self._normalize_vector(perp_force)

            # Project out translations and rotations from rotation direction
            if self.project_translations_rotations and not self.solid_state:
                self.rotation_direction = self._project_out_translations_rotations(
                    self.rotation_direction, self.atoms
                )

            # Calculate curvature and its derivative
            curvature = np.vdot(central_forces - dimer_forces, self.mode) / self.dimer_separation
            curvature_derivative = np.vdot(central_forces - dimer_forces, self.rotation_direction) / self.dimer_separation * 2.0
            
            # Initial rotation angle estimate
            phi_1 = -0.5 * atan(curvature_derivative / (2.0 * max(abs(curvature), EPSILON)))
            
            # Early exit if rotation angle is already small
            if abs(phi_1) <= self.rotation_tolerance:
                break

            # Calculate forces for trial rotation
            trial_mode = self._normalize_vector(
                self.mode * cos(phi_1) + self.rotation_direction * sin(phi_1)
            )
            self._update_dimer_endpoint(trial_mode, self.atoms, self.trial_image)
            trial_forces = self._calculate_general_forces(self.trial_image)
            trial_curvature = np.vdot(central_forces - trial_forces, trial_mode) / self.dimer_separation
            
            # Fit curvature to c(φ) = a0/2 + a1*cos(2φ) + b1*sin(2φ)
            b1 = 0.5 * curvature_derivative
            a1 = (curvature - trial_curvature + b1 * sin(2 * phi_1)) / max(1 - cos(2 * phi_1), EPSILON)
            a0 = 2.0 * (curvature - a1)
            
            # Calculate optimal rotation angle
            phi_min = 0.5 * atan(b1 / max(a1, EPSILON))
            min_curvature = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)

            # Check if we found a minimum or maximum
            if min_curvature > curvature:
                phi_min += pi * 0.5
                min_curvature = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2 * phi_min)
                
            # Normalize angle for numerical stability
            if phi_min > pi * 0.5:
                phi_min -= pi
                
            # Save current mode for rotation optimization
            old_mode = self.mode.copy()
            
            # Update dimer direction
            self.mode = self._normalize_vector(
                self.mode * cos(phi_min) + self.rotation_direction * sin(phi_min)
            )
            
            # Project out translations and rotations
            if self.project_translations_rotations and not self.solid_state:
                self.mode = self._project_out_translations_rotations(self.mode, self.atoms)
                
            # Update curvature
            curvature = min_curvature

            # Update dimer forces using extrapolation
            safe_sin_phi1 = max(sin(phi_1), EPSILON)
            dimer_forces = (
                dimer_forces * (sin(phi_1 - phi_min) / safe_sin_phi1) +
                trial_forces * (sin(phi_min) / safe_sin_phi1) +
                central_forces * (1.0 - cos(phi_min) - sin(phi_min) * tan(phi_1 * 0.5))
            )

            # Calculate perpendicular forces for next iteration
            central_perp = central_forces - np.vdot(central_forces, self.mode) * self.mode
            dimer_perp = dimer_forces - np.vdot(dimer_forces, self.mode) * self.mode
            old_perp_force = perp_force
            perp_force = 2.0 * (dimer_perp - central_perp)
            
            # Calculate rotation direction for next iteration
            self._update_rotation_direction(perp_force, old_perp_force, old_mode)

            iteration += 1
            
        # Store final curvature and return forces
        self.curvature = curvature
        return central_forces

    def get_curvature(self):
        """
        Return the current curvature along the dimer axis.
        
        Returns
        -------
        float
            Curvature value (negative at saddle points)
        """
        return self.curvature

    def get_mode(self):
        """
        Return the current dimer mode (direction of lowest curvature).
        
        Returns
        -------
        ndarray
            Normalized mode vector
        """
        if self.solid_state:
            return self.mode
        else:
            return self.mode[:self.num_atoms]

    def get_forces(self):
        """
        Calculate modified forces for the dimer method.
        
        The dimer method modifies the potential energy surface forces to
        create an effective force that drives the system toward saddle points.
        
        Returns
        -------
        ndarray
            Modified force vector for dimer translation
        """
        # Find the minimum mode direction and calculate true forces
        true_forces = self.find_minimum_mode()
        
        # Project forces along the dimer direction
        parallel_component = np.vdot(true_forces, self.mode) * self.mode
        perpendicular_component = true_forces - parallel_component
        
        # Calculate ratio of perpendicular component (for diagnostic purposes)
        alpha = self._vector_magnitude(perpendicular_component) / self._vector_magnitude(true_forces)
        print(f"Perpendicular component ratio: {alpha:.4f}")
        print(f"Curvature: {self.curvature:.6f}")
        
        # Dimer force modification
        if self.curvature > 0:
            # At a minimum along mode direction - invert the parallel component
            self.modified_forces = -1.0 * parallel_component
            print("Climbing uphill along mode")
        else:
            # At a saddle point - follow perpendicular component but invert parallel
            gamma = 1.0  # Scaling factor for parallel component
            self.modified_forces = perpendicular_component - gamma * parallel_component
            
        # Return full or truncated force vector depending on ss mode
        if self.solid_state:
            return self.modified_forces
        else:
            return self.modified_forces[:self.num_atoms]

    def __len__(self):
        """Return the number of degrees of freedom."""
        if self.solid_state:
            return self.num_atoms + 3
        else:
            return self.num_atoms

    def __getattr__(self, attr):
        """Pass through attributes not found to the atoms object."""
        if hasattr(self, 'atoms'):
            return getattr(self.atoms, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
        
