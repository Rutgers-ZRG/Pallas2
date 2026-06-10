import pytest

from pallas import PallasConfig

pytestmark = pytest.mark.unit

GPA_PER_EVA3 = 160.21766208


def test_pressure_gpa_converts_to_ev_a3():
    cfg = PallasConfig(znucl=[6], pressure_gpa=15.0)
    assert cfg.press == pytest.approx(15.0 / GPA_PER_EVA3)


def test_press_still_accepted_as_ev_a3():
    cfg = PallasConfig(znucl=[6], press=0.0936)
    assert cfg.press == pytest.approx(0.0936)


def test_both_pressure_specs_conflict():
    with pytest.raises(ValueError):
        PallasConfig(znucl=[6], press=0.1, pressure_gpa=15.0)
