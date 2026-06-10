import pytest

from pallas.structure import enthalpy

pytestmark = pytest.mark.unit


def test_enthalpy_is_e_plus_pv():
    # press is in eV/A^3 internally: H = E + P*V, no hidden unit factors.
    assert enthalpy(energy=-10.0, volume=100.0, press=0.09362) == pytest.approx(
        -10.0 + 9.362, abs=1e-6)


def test_enthalpy_zero_pressure():
    assert enthalpy(energy=-3.5, volume=250.0, press=0.0) == pytest.approx(-3.5)
