"""NequIP dual-model calculator: energy+forces from one model, stress from another.

Wraps two TorchScript NequIP models into a single ASE calculator.
The stress model output is converted from NequIP internal units (kbar-like)
to eV/A^3 via the factor 1/1602.1766.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class NequIPDualCalc(Calculator):
    """ASE calculator using separate NequIP models for E+F and stress.

    Parameters
    ----------
    ef_model_path : str — path to energy+forces TorchScript model (.pth).
    stress_model_path : str — path to stress TorchScript model (.pth).
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, ef_model_path, stress_model_path, **kwargs):
        super().__init__(**kwargs)
        from nequip.ase.nequip_calculator import nequip_calculator
        self._ef_calc = nequip_calculator(ef_model_path)
        self._s_calc = nequip_calculator(stress_model_path)

    def calculate(self, atoms=None, properties=None,
                  system_changes=tuple(all_changes)):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        atoms = self.atoms

        # Energy + forces from EF model
        ef_atoms = atoms.copy()
        ef_atoms.calc = self._ef_calc
        self.results['energy'] = ef_atoms.get_potential_energy()
        self.results['forces'] = ef_atoms.get_forces()

        # Stress from stress model (unit conversion)
        if 'stress' in properties:
            s_atoms = atoms.copy()
            s_atoms.calc = self._s_calc
            raw_stress = s_atoms.get_stress()
            self.results['stress'] = -raw_stress / 1602.1766208
