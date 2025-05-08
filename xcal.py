import numpy as np
import ase.io
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculatorSetupError, all_changes

from mpi4py import MPI

import itertools

# Note: There is a bug in the libfp C implementation where the assignment array (ci)
# returned by get_fp_dist can contain uninitialized values for some atom indices.
# This calculator implements a workaround by filtering and validating ci indices.

# try:
#     from numba import jit, float64, int32
#     use_numba = True
# except ImportError:
#     use_numba = False
#     # Define dummy decorator and type aliases if Numba is not available
def jit(*args, **kwargs):
    return lambda func: func

float64 = int32 = lambda: None

# try:
#     import libfp
# except:
#     from reformpy import libfppy as libfp
#     print("Warning: Failed to import libfp. Using Python version of libfppy (python implementation of libfp) instead, which may affect performance.")
import libfp
# import libfppy 
#################################### ASE Reference ####################################
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py       #
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/vasp.py        #
#        https://wiki.fysik.dtu.dk/ase/development/calculators.html                   #
#######################################################################################

class XCalculator(Calculator):
    """ASE interface for Reform, with the Calculator interface.

        Implemented Properties:

            'energy': Sum of atomic fingerprint distance (L2 norm of two atomic
                                                          fingerprint vectors)

            'energies': Per-atom property of 'energy'

            'forces': Gradient of fingerprint energy, using Hellmann–Feynman theorem

            'stress': Cauchy stress tensor using finite difference method

            'stresses': Per-atom property of 'stress'

        Parameters:

            atoms:  object
                Attach an atoms object to the calculator.

            contract: bool
                Calculate fingerprint vector in contracted Guassian-type orbitals or not

            ntype: int
                Number of different types of atoms in unit cell

            nx: int
                Maximum number of atoms in the sphere with cutoff radius for specific cell site

            lmax: int
                Integer to control whether using s orbitals only or both s and p orbitals for
                calculating the Guassian overlap matrix (0 for s orbitals only, other integers
                will indicate that using both s and p orbitals)

            cutoff: float
                Cutoff radius for f_c(r) (smooth cutoff function) [amp], unit in Angstroms

    """
    # name = 'fingerprint'
    # ase_objtype = 'fingerprint_calculator'  # For JSON storage

    implemented_properties = [ 'energy', 'forces', 'stress' ]
    # implemented_properties += ['energies', 'stresses'] # per-atom properties

    default_parameters = {
                          'contract': False,
                          'ntyp': 1,
                          'nx': 200,
                          'lmax': 0,
                          'cutoff': 5.5,
                          'znucl': None
                          }

    nolabel = True

    def __init__(self,
                 atoms = None,
                 comm = None, # MPI communicator
                 parallel = False, # Switch to enable/disable MPI for this calculator
                 ntyp = 1,
                 fp0 = None,
                 ci = None, 
                 **kwargs
                ):
        
        """Initialize the Calculator with optional MPI awareness.

        Parameters
        ----------
        comm : MPI communicator or None
            If None, defaults to MPI.COMM_WORLD (or a 'fake' serial communicator).
        parallel : bool
            If True, the code runs only on rank 0 and then broadcasts results.
            If False, it computes in pure serial on each rank (original behavior).
        """
        # Initialize results dictionary first
        self.results = {}
        
        # Initialize other attributes
        self._energy = None
        self._forces = None
        self._stress = None
        self._atoms = None
        self._types = None
        self.cell_file = 'POSCAR'
        self.default_parameters = {}
        
        # If no communicator is given, we can default to COMM_WORLD
        # or a "serial" communicator if you prefer:
        if comm is None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.rank = comm.Get_rank()
        self.parallel = parallel
        self.fp0 = fp0
        self.ntyp = ntyp
        self.ci = ci if ci is not None else []
        self._ci_initialized = False

        # Initialize parameter dictionaries
        self._store_param_state()  # Initialize an empty parameter state

        # Call parent class initialization
        Calculator.__init__(self,
                          atoms=atoms,
                          **kwargs)

        # Set up atoms after parent initialization
        if atoms is None:
            atoms = ase.io.read(self.cell_file)
        self.atoms = atoms
        self.atoms_save = None

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        fingerprint Calculator.
        """
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects
            self.directory = str(kwargs.pop('directory'))

        if 'txt' in kwargs:
            self.txt = kwargs.pop('txt')

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        if 'command' in kwargs:
            self.command = kwargs.pop('command')

        
        self.default_parameters.update(Calculator.set(self, **kwargs))
        changed_parameters.update(Calculator.set(self, **kwargs))

        if changed_parameters:
            self.clear_results()  # We don't want to clear atoms
        for key in kwargs:
            self.default_parameters[key] = kwargs[key]
            self.results.clear()

    def reset(self):
        self.atoms = None
        self.clear_results()
        # self._ci_initialized = False

    def clear_results(self):
        self.results.clear()
        # Also clear _types to ensure it's regenerated from current atoms
        self._types = None

    def restart(self):
        """Reset calculation results but preserve internal state."""
        # Clear calculated properties but preserve _ci_initialized and ci
        self._energy = None
        self._forces = None
        self._stress = None
        # Do NOT reset _ci_initialized as it's an optimization
        # Reset types to ensure they're regenerated from current atoms
        self._types = None

    def check_restart(self, atoms = None):
        """Check if we need to restart calculations for given atoms.
        Returns True if calculations need to be restarted, False otherwise."""
        if atoms is None:
            atoms = self.atoms
            
        # If we don't have saved atoms yet, we definitely need to restart
        if self.atoms_save is None:
            self.atoms_save = atoms.copy()
            self.restart()
            return True
            
        # Check if the atoms are essentially the same (positions might differ slightly)
        # by comparing number of atoms, chemical symbols, and checking if positions are close
        if (len(atoms) == len(self.atoms_save) and
            atoms.get_chemical_symbols() == self.atoms_save.get_chemical_symbols() and
            np.allclose(atoms.positions, self.atoms_save.positions, atol=1e-10) and
            np.allclose(atoms.cell, self.atoms_save.cell, atol=1e-10)):
            # Atoms are essentially the same, no need to restart
            return False
        else:
            # Atoms have changed significantly, save the new state and restart
            self.atoms_save = atoms.copy()
            self.restart()
            return True

    def calculate(self,
                  atoms = None,
                  properties = [ 'energy', 'forces', 'stress' ],
                  system_changes = tuple(all_changes),
                 ):
        """Do a fingerprint calculation in the specified directory.
        This will read VASP input files (POSCAR) and then execute
        fp_GD.
        """
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)

        # Wrap atoms back into the cell before calculating
        if atoms is not None:
            atoms_copy = atoms.copy()
            atoms = atoms_copy
            
        # Call parent calculate with wrapped atoms
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Use self.atoms which should be properly set by the parent calculate method
        atoms = self.atoms

        # Calculate requested properties and store in results
        if 'energy' in properties:
            self.results['energy'] = self.get_potential_energy(atoms)
            
        if 'forces' in properties:
            self.results['forces'] = self.get_forces(atoms)
            
        if 'stress' in properties:
            self.results['stress'] = self.get_stress(atoms)

    def check_state(self, atoms, tol = 1e-15):
        """Check for system changes since last calculation."""
        def compare_dict(d1, d2):
            """Helper function to compare dictionaries"""
            # Use symmetric difference to find keys which aren't shared
            # for python 2.7 compatibility
            if set(d1.keys()) ^ set(d2.keys()):
                return False

            # Check for differences in values
            for key, value in d1.items():
                if np.any(value != d2[key]):
                    return False
            return True

        # First we check for default changes
        system_changes = Calculator.check_state(self, atoms, tol = tol)

        '''
        # We now check if we have made any changes to the input parameters
        # XXX: Should we add these parameters to all_changes?
        for param_string, old_dict in self.param_state.items():
            param_dict = getattr(self, param_string)  # Get current param dict
            if not compare_dict(param_dict, old_dict):
                system_changes.append(param_string)
        '''

        return system_changes


    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            default_parameters = self.default_parameters.copy()
            )

    # Below defines some functions for faster access to certain common keywords

    @property
    def contract(self):
        """Access the contract in default_parameters dict"""
        return self.default_parameters['contract']

    @contract.setter
    def contract(self, contract):
        """Set contract in default_parameters dict"""
        self.default_parameters['contract'] = contract

    @property
    def ntyp(self):
        """Access the ntyp in default_parameters dict.
        Note: This value will be overridden by the actual number of unique types
        in the atoms object when performing calculations."""
        return self.default_parameters['ntyp']

    @ntyp.setter
    def ntyp(self, ntyp):
        """Set ntyp in default_parameters dict.
        Warning: This is used as a fallback only. The actual number
        of types is determined from the atoms object during calculation."""
        self.default_parameters['ntyp'] = ntyp

    @property
    def nx(self):
        """Access the nx in default_parameters dict"""
        return self.default_parameters['nx']

    @nx.setter
    def nx(self, nx):
        """Set ntyp in default_parameters dict"""
        self.default_parameters['nx'] = nx

    @property
    def lmax(self):
        """Access the lmax in default_parameters dict"""
        return self.default_parameters['lmax']

    @lmax.setter
    def lmax(self, lmax):
        """Set ntyp in default_parameters dict"""
        self.default_parameters['lmax'] = lmax

    @property
    def cutoff(self):
        """Access the cutoff in default_parameters dict"""
        return self.default_parameters['cutoff']

    @cutoff.setter
    def cutoff(self, cutoff):
        """Set cutoff in default_parameters dict"""
        self.default_parameters['cutoff'] = cutoff

    @property
    def znucl(self):
        """Access the znucl array in default_parameters dict"""
        return self.default_parameters['znucl']

    @znucl.setter
    def znucl(self, znucl):
        """Direct access for setting the znucl"""
        if isinstance(znucl, (list, np.ndarray)):
            znucl = list(znucl)
        self.set(znucl = znucl)

    @property
    def types(self):
        """Get the types array, using the atoms object if available."""
        if self._types is None and self._atoms is not None:
            self._types = read_types(self._atoms)
        return self._types

    @types.setter
    def types(self, types):
        """Set the types array manually or based on the atoms object."""
        if types is not None:
            # Convert to numpy array and ensure int type
            self._types = np.array(types)
        else:
            if self._atoms is not None:
                self._types = read_types(self._atoms)
            else:
                self._types = np.array([])

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        """Set the atoms and update the types accordingly.
        Preserves calculation results if the atoms are essentially the same."""
        if atoms is None:
            self._atoms = None
            self._types = None
            self.clear_results()
            return
            
        # Check if atoms have changed significantly
        needs_reset = False
        if self._atoms is None:
            needs_reset = True
        elif (len(atoms) != len(self._atoms) or
              atoms.get_chemical_symbols() != self._atoms.get_chemical_symbols() or
              not np.allclose(atoms.positions, self._atoms.positions, atol=1e-10) or
              not np.allclose(atoms.cell, self._atoms.cell, atol=1e-10)):
            needs_reset = True
            
        # Make a copy and wrap the atoms
        atoms_copy = atoms.copy()
        
        # Store the wrapped atoms
        self._atoms = atoms_copy
        
        # Update types
        self._types = read_types(atoms_copy)
        
        # Clear results if needed
        if needs_reset:
            self.clear_results()

    def get_potential_energy(self, atoms = None, **kwargs):
        """MPI-aware energy computation."""
        if atoms is None:
            atoms = self.atoms
            
        # Ensure atoms are wrapped into the cell
        atoms_copy = atoms.copy()
            
        if self.check_restart(atoms_copy) or self._energy is None:

            if self.parallel:
                # Only rank 0 does the real calculation
                if self.rank == 0:
                    energy = self._compute_energy(atoms_copy)
                else:
                    energy = None

                # Broadcast from rank 0 to all other ranks
                energy = self.comm.bcast(energy, root=0)
                self._energy = energy

            else:
                # Serial fallback (the original behavior)
                self._energy = self._compute_energy(atoms_copy)

        return self._energy

    def get_forces(self, atoms = None, **kwargs):
        """MPI-aware forces computation."""
        if atoms is None:
            atoms = self.atoms
            
        # Ensure atoms are wrapped into the cell
        atoms_copy = atoms.copy()

        if self.check_restart(atoms_copy) or self._forces is None:

            if self.parallel:
                if self.rank == 0:
                    forces = self._compute_forces(atoms_copy)
                else:
                    forces = None
                forces = self.comm.bcast(forces, root=0)
                self._forces = forces
            else:
                self._forces = self._compute_forces(atoms_copy)

        return self._forces

    def get_stress(self, atoms = None, **kwargs):
        """MPI-aware stress computation."""
        if atoms is None:
            atoms = self.atoms
            
        # Ensure atoms are wrapped into the cell
        atoms_copy = atoms.copy()

        if self.check_restart(atoms_copy) or self._stress is None:

            if self.parallel:
                if self.rank == 0:
                    stress = self._compute_stress(atoms_copy)
                else:
                    stress = None
                stress = self.comm.bcast(stress, root=0)
                self._stress = stress
            else:
                self._stress = self._compute_stress(atoms_copy)

        return self._stress

    def _initialize_ci(self, atoms):
        """Initialize or update ci assignments if needed."""
        if self._ci_initialized:
            return
            
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        
        # Always regenerate types from current atoms
        types = read_types(atoms)
        
        znucl = self.znucl
        
        # Debugging output to help diagnose issues
#        print(f"Initialize CI: atoms={len(rxyz)}, types={len(types)}, shapes match: {len(rxyz) == len(types)}")
        
        lat = np.array(lat)
        rxyz = np.array(rxyz)
        types = np.array(types)
        znucl = np.int32(znucl)
        cutoff = np.float64(self.cutoff)
        nx = np.int32(self.nx)
        
        cell = (lat, rxyz, types, znucl)
        fp, _ = libfp.get_dfp(cell, cutoff=cutoff, log=False, natx=nx)
        fp = np.float64(fp)
        fp0 = np.float64(self.fp0)
        
        fpdist, ci = libfp.get_fp_dist(fp, fp0, types, assignment=True)
        # Store the updated assignment indices
        
        # Filter out invalid ci values (workaround for libfp bug)
        nat = len(fp)
        valid_ci = []
        for i in range(nat):
            if i < len(ci) and isinstance(ci[i], (int, np.integer)) and 0 <= ci[i] < len(self.fp0):
                valid_ci.append(ci[i])
            else:
                # Use closest match based on fingerprint distance
                dists = []
                for j in range(len(self.fp0)):
                    diff = fp[i] - self.fp0[j]
                    dists.append(np.sqrt(np.dot(diff, diff)))
                valid_ci.append(np.argmin(dists))
        
        self.ci = valid_ci
        # print ('ci', self.ci)
        self._ci_initialized = True
        
    def _compute_energy(self, atoms):
        """Actual energy computation using fingerprint method."""
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        znucl = self.znucl
        
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        
        # Always regenerate types from current atoms
        types = read_types(atoms)
        
        lat = np.array(lat)
        rxyz = np.array(rxyz)
        types = np.array(types)  # Make sure types is a proper numpy array
        znucl = np.int32(znucl)
        # Calculate ntyp based on actual unique types rather than using the default
        ntyp = np.int32(len(set(types)))
        nx = np.int32(nx)
        lmax = np.int32(lmax)
        cutoff = np.float64(cutoff)
        
        cell = (lat, rxyz, types, znucl)
        
        # Debugging output to help diagnose issues
        # print(f"Energy calc: atoms={len(rxyz)}, types={len(types)}, shapes match: {len(rxyz) == len(types)}")
        
        fp, dfp = libfp.get_dfp(cell, cutoff=cutoff, log=False, natx=nx)
        fp = np.float64(fp)
        dfp = np.array(dfp)
        fp0 = np.float64(self.fp0)
    
        # Verify dimensions
        nat = len(fp)
        if len(self.fp0) != nat:
            print(f"Warning: Dimension mismatch between fp ({nat}) and fp0 ({len(self.fp0)})")

        fpdist, ci = libfp.get_fp_dist(fp, fp0, types, assignment=True)
        fpe = fpdist
        self.ci = ci
        return fpe

    def _compute_forces(self, atoms):
        """Actual forces computation using fingerprint method."""
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        znucl = self.znucl
        
        # Make sure ci is initialized before computing forces
        if not self._ci_initialized:
            self._initialize_ci(atoms)
        
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        
        # Always regenerate types from current atoms
        types = read_types(atoms)
        
        lat = np.array(lat)
        rxyz = np.array(rxyz)
        types = np.array(types)  # Make sure types is a proper numpy array
        znucl = np.int32(znucl)
        # Calculate ntyp based on actual unique types rather than using the default
        ntyp = np.int32(len(set(types)))
        nx = np.int32(nx)
        lmax = np.int32(lmax)
        cutoff = np.float64(cutoff)

        cell = (lat, rxyz, types, znucl)
        
        # Debugging output to help diagnose issues
        # print(f"Forces calc: atoms={len(rxyz)}, types={len(types)}, shapes match: {len(rxyz) == len(types)}")
        
        fp, dfp = libfp.get_dfp(cell, cutoff=cutoff, log=False, natx=nx)
        fp = np.float64(fp)
        dfp = np.array(dfp)
        fp0 = np.array(self.fp0)
        
        # Verify dimensions
        nat = len(fp)
        if len(self.fp0) != nat:
            print(f"Warning: Dimension mismatch between fp ({nat}) and fp0 ({len(self.fp0)})")
        
        nat = len(fp)
        fpf = np.zeros((nat, 3))

        # print ('ci update? ', self.ci)

        for k in range(nat):
            # Check if self.fp0 exists and ci[k] is valid
            if self.fp0 is None:
                raise ValueError("self.fp0 is None, cannot compute forces")
            
            # Ensure the index is valid
            idx = self.ci[k]
            fp12 = (fp[k] - self.fp0[idx])
            for l in range(3):
                ff = 0.0
                for i in range(nat):
                    ff += -np.dot(dfp[i][k][l], fp12)
                fp12_norm = np.linalg.norm(fp12)
                if fp12_norm < 1e-10:  # Prevent division by zero
                    fpf[k, l] = 0.0
                else:
                    fpf[k, l] = ff/fp12_norm

        return fpf

    def _compute_stress(self, atoms):
        """Actual stress computation using virial theorem."""
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        # delta = self.default_parameters.get('fd_delta', 1e-3)
        # return self._compute_stress_fd(atoms)     
        return self._compute_stress_virial(atoms)

        # Use a direct force calculation method to avoid recursion
        # forces = self._calculate_forces_for_stress(atoms)
        
        # lat = np.array(lat)
        # rxyz = np.array(rxyz)
        # forces = np.array(forces)
        
        # stress = get_stress(lat, rxyz, forces)
        # return stress
        # mode = self.default_parameters.get('stress_mode', 'analytical')

        # if mode == 'analytical':
        #     return self._compute_stress_virial(atoms)

        # elif mode == 'finite_diff':
        #     delta = self.default_parameters.get('fd_delta', 1e-3)
        #     return self._compute_stress_fd(atoms, delta)

        # else:
        #     raise ValueError(f"Unknown stress_mode '{mode}'.")
        

    def _compute_stress_virial(self, atoms):
        # 1. make sure forces are up-to-date -------------------------------
        forces = self._compute_forces(atoms)        # shape (nat, 3)

        # 2. centre positions if ΣF ≠ 0 (numerical noise)
        rxyz = atoms.get_positions()
        if abs(forces.sum(0)).max() > 1e-8:
            rxyz -= rxyz.mean(0)

        # 3. basic virial
        stress = -np.einsum('ia,ib->ab', forces, rxyz) / atoms.get_volume()

        # 4. add explicit ∂E/∂ε if your descriptor depends on cell ---------
        # stress += self._pulay_term(atoms)      # implement if needed

        # 5. symmetrise (kills  ~10⁻⁵ eV Å⁻³ noise)
        stress = 0.5 * (stress + stress.T)

        # 6. convert to Voigt (xx,yy,zz,yz,xz,xy)
        return np.array([stress[0,0], stress[1,1], stress[2,2],
                        stress[1,2], stress[0,2], stress[0,1]], float)


    

    def _compute_stress_fd(self, atoms: Atoms, delta: float = 1e-3):
        """
        σ_αβ = (1/V) ∂E/∂ε_αβ   →   central finite difference with ±δ strain.
        Off-diagonal strains use ε = γ/2, so we multiply by 2 at the end.
        """
        # Reference energy and volume
        e0 = self._compute_energy(atoms)
        V0 = atoms.get_volume()
        C0 = atoms.get_cell()          # 3×3 matrix

        stress_voigt = np.zeros(6)
        _voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for iv, (a, b) in enumerate(_voigt_pairs):
            # ε tensor with only one non-zero component
            eps = np.zeros((3, 3))
            eps[a, b] = eps[b, a] = delta   # symmetric strain

            # +δ strain
            atoms_p = atoms.copy()
            atoms_p.set_cell((np.eye(3) + eps) @ C0, scale_atoms=True)
            e_plus = self._compute_energy(atoms_p)

            # −δ strain
            atoms_m = atoms.copy()
            atoms_m.set_cell((np.eye(3) - eps) @ C0, scale_atoms=True)
            e_minus = self._compute_energy(atoms_m)

            dE = (e_plus - e_minus) / (2.0 * delta)
            σ = dE / V0

            # Off-diagonals: ε_yz, ε_xz, ε_xy are half the engineering shear γ,
            # so multiply the derivative by 2 to get the Cauchy stress.
            if iv >= 3:
                σ *= 2.0

            stress_voigt[iv] = σ

        return stress_voigt


        
    def _calculate_forces_for_stress(self, atoms):
        """Direct calculation of forces without using get_forces to avoid recursion."""
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        znucl = self.znucl
        
        # Make sure ci is initialized before computing forces
        if not self._ci_initialized:
            self._initialize_ci(atoms)
            
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        
        # Always regenerate types from current atoms
        types = read_types(atoms)
        
        # Debugging output to help diagnose issues
        # print(f"Stress forces calc: atoms={len(rxyz)}, types={len(types)}, shapes match: {len(rxyz) == len(types)}")
        
        lat = np.array(lat)
        rxyz = np.array(rxyz)
        types = np.array(types)  # Make sure types is a proper numpy array
        znucl = np.int32(znucl)
        # Calculate ntyp based on actual unique types rather than using the default
        ntyp = np.int32(len(set(types)))
        nx = np.int32(nx)
        lmax = np.int32(lmax)
        cutoff = np.float64(cutoff)

        cell = (lat, rxyz, types, znucl)
        fp, dfp = libfp.get_dfp(cell, cutoff=cutoff, log=False, natx=nx)
        fp = np.float64(fp)
        dfp = np.array(dfp)
        fp0 = np.array(self.fp0)
        
        nat = len(fp)
        fpf = np.zeros((nat, 3))

        for k in range(nat):
            # Check if self.fp0 exists and ci[k] is valid
            if self.fp0 is None:
                raise ValueError("self.fp0 is None, cannot compute forces")
            
            idx = self.ci[k]
                
            fp12 = (fp[k] - self.fp0[idx])
            for l in range(3):
                ff = 0.0
                for i in range(nat):
                    ff += -np.dot(dfp[i][k][l], fp12)
                fp12_norm = np.linalg.norm(fp12)
                if fp12_norm < 1e-10:  # Prevent division by zero
                    fpf[k, l] = 0.0
                else:
                    fpf[k, l] = ff/fp12_norm

        return fpf

    def test_energy_consistency(self, atoms = None, **kwargs):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        types = self.types
        znucl = self.znucl
        
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        if types is None:
            types = read_types(atoms)
        
        lat = np.array(lat)
        rxyz = np.array(rxyz)
        types = np.int32(types)
        znucl =  np.int32(znucl)
        ntyp =  np.int32(ntyp)
        nx = np.int32(nx)
        lmax = np.int32(lmax)
        cutoff = np.float64(cutoff)   
        
        rxyz_delta = np.zeros_like(rxyz)
        rxyz_disp = np.zeros_like(rxyz)
        rxyz_left = np.zeros_like(rxyz)
        rxyz_mid = np.zeros_like(rxyz)
        rxyz_right = np.zeros_like(rxyz)
        
        nat = len(rxyz)
        del_fpe = 0.0
        iter_max = 100
        step_size = 1.e-5
        rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
                                0.5*np.ones((nat, 3), dtype = np.float64) )
        
        for i_iter in range(iter_max):
            rxyz_disp += 2.0*rxyz_delta
            rxyz_left = rxyz.copy() + 2.0*i_iter*rxyz_delta
            rxyz_mid = rxyz.copy() + 2.0*(i_iter+1)*rxyz_delta
            rxyz_right = rxyz.copy() + 2.0*(i_iter+2)*rxyz_delta
            
            fp_left, dfp_left = libfp.get_dfp((lat, rxyz_left, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fp_mid, dfp_mid = libfp.get_dfp((lat, rxyz_mid, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fp_right, dfp_right = libfp.get_dfp((lat, rxyz_right, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fpe_left, fpf_left = get_ef(fp_left, dfp_left, ntyp, types)
            fpe_mid, fpf_mid = get_ef(fp_mid, dfp_mid, ntyp, types)
            fpe_right, fpf_right = get_ef(fp_right, dfp_right, ntyp, types)

            for i_atom in range(nat):
                del_fpe += ( -np.dot(rxyz_delta[i_atom], fpf_left[i_atom]) - \
                            4.0*np.dot(rxyz_delta[i_atom], fpf_mid[i_atom]) - \
                            np.dot(rxyz_delta[i_atom], fpf_right[i_atom]) )/3.0
        
        rxyz_final = rxyz + rxyz_disp
        fp_init = libfp.get_lfp((lat, rxyz, types, znucl),
                                cutoff = cutoff, log = False, natx = nx)
        fp_final = libfp.get_lfp((lat, rxyz_final, types, znucl),
                                cutoff = cutoff, log = False, natx = nx)
        e_init = get_fpe(fp_init, ntyp, types)
        e_final = get_fpe(fp_final, ntyp, types)
        e_diff = e_final - e_init
        
        print ( "Numerical integral = {0:.6e}".format(del_fpe) )
        print ( "Fingerprint energy difference = {0:.6e}".format(e_diff) )
        if np.allclose(del_fpe, e_diff, rtol=1e-6, atol=1e-6, equal_nan=False):
            print("Energy consistency test passed!")
        else:
            print("Energy consistency test failed!")


    def test_force_consistency(self, atoms = None, **kwargs):

        from ase.calculators.test import numeric_force

        indices = range(len(atoms))
        f = atoms.get_forces()[indices]
        print('{0:>16} {1:>20}'.format('eps', 'max(abs(df))'))
        for eps in np.logspace(-1, -8, 8):
            fn = np.zeros((len(indices), 3))
            for idx, i in enumerate(indices):
                for j in range(3):
                    fn[idx, j] = numeric_force(atoms, i, j, eps)
            print('{0:16.12f} {1:20.12f}'.format(eps, abs(fn - f).max()))


        print ( "Numerical forces = \n{0:s}".\
               format(np.array_str(fn, precision=6, suppress_small=False)) )
        print ( "Fingerprint forces = \n{0:s}".\
               format(np.array_str(f, precision=6, suppress_small=False)) )
        if np.allclose(f, fn, rtol=1e-6, atol=1e-6, equal_nan=False):
            print("Force consistency test passed!")
        else:
            print("Force consistency test failed!")


def check_atoms(atoms: Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, Atoms):
        raise CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))




# @jit('Tuple((float64, float64[:,:]))(float64[:,:], float64[:,:,:,:], int32, \
#       int32[:])', nopython=True)
def get_ef(fp, dfp, ntyp, types):
    nat = len(fp)
    e = 0.
    fp = np.ascontiguousarray(fp)
    dfp = np.ascontiguousarray(dfp)
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.vdot(vij, vij)
                    e0 += t
            e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
        # print ("e0", e0)
        e += e0
    # print ("e", e)

    force_0 = np.zeros((nat, 3))
    force_prime = np.zeros((nat, 3))

    for k in range(nat):
        for ityp in range(ntyp):
            itype = ityp + 1
            for i in range(nat):
                for j in range(nat):
                    if  types[i] == itype and types[j] == itype:
                        vij = fp[i] - fp[j]
                        dvij = dfp[i][k] - dfp[j][k]
                        for l in range(3):
                            t = -2 * np.vdot(vij, dvij[l])
                            force_0[k][l] += t
                for m in range(3):
                    t_prime = 2.0 * np.vdot(fp[i],dfp[i][k][m]) / (np.linalg.norm(fp[i]) ** 4)
                    force_prime[k][m] += t_prime
    force = force_0 + force_prime
    force = force - np.sum(force, axis=0)/len(force)
    # return ((e+1.0)*np.log(e+1.0)-e), force*np.log(e+1.0) 
    return e, force


@jit('(float64)(float64[:,:], int32, int32[:])', nopython=True)
def get_fpe(fp, ntyp, types):
    nat = len(fp)
    e = 0.
    fp = np.ascontiguousarray(fp)
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.vdot(vij, vij)
                    e0 += t
            e0 += 1.0/(np.linalg.norm(fp[i]) ** 2)
        e += e0
    # return ((e+1.0)*np.log(e+1.0)-e)
    return e



# @jit('(float64[:])(float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def get_stress(lat, rxyz, forces):
    """
    Compute the stress tensor analytically using the virial theorem.

    Parameters:
    - lat: (3, 3) array of lattice vectors.
    - rxyz: (nat, 3) array of atomic positions in Cartesian coordinates.
    - forces: (nat, 3) array of forces on each atom.

    Returns:
    - stress_voigt: (6,) array representing the stress tensor in Voigt notation.
    """
    # Ensure inputs are NumPy arrays with correct data types
    lat = np.asarray(lat)
    rxyz = np.asarray(rxyz)
    forces = np.asarray(forces)

    # Compute the cell volume
    cell_vol = np.abs(np.linalg.det(lat))

    # Initialize the stress tensor
    stress_tensor = np.zeros((3, 3))

    # Compute the stress tensor using the virial theorem
    nat = rxyz.shape[0]
    for i in range(nat):
        for m in range(3):
            for n in range(3):
                stress_tensor[m, n] -= forces[i, m] * rxyz[i, n]

    # Divide by the cell volume
    stress_tensor /= cell_vol
    # stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

    # Ensure the stress tensor is symmetric (if applicable)
    # stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

    # Convert the stress tensor to Voigt notation
    # The Voigt notation order is: [xx, yy, zz, yz, xz, xy]
    stress_voigt = np.array([
        stress_tensor[0, 0],  # xx
        stress_tensor[1, 1],  # yy
        stress_tensor[2, 2],  # zz
        stress_tensor[1, 2],  # yz
        stress_tensor[0, 2],  # xz
        stress_tensor[0, 1],  # xy
    ], float)

    return stress_voigt


def read_types(atoms: Atoms):
    """
    Reads atomic types from an ASE Atoms object and returns an array of types.
    """
    atom_symbols = atoms.get_chemical_symbols()
    
    # Track order of appearance and count unique elements
    unique_symbols = []
    symbol_to_idx = {}
    
    for symbol in atom_symbols:
        if symbol not in symbol_to_idx:
            symbol_to_idx[symbol] = len(unique_symbols)
            unique_symbols.append(symbol)
    
    # Count occurrences of each symbol
    counts = [atom_symbols.count(symbol) for symbol in unique_symbols]
    # Create types array based on order of appearance
    types = [symbol_to_idx[symbol] + 1 for symbol in atom_symbols]

    # Make sure to return a numpy array of the correct type
    return np.array(types)