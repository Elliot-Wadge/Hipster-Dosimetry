from HipsterDosimetry.calibration import RationalCalibration
import pytest
import numpy as np


def test_initialization():
    cal = RationalCalibration()
    with pytest.raises(ValueError):
        cal.inverse(10)
    with pytest.raises(ValueError):
        cal.fit(np.array([1,2]), np.array([3,4]))
    with pytest.raises(ValueError):
        cal.fit(np.array([1,2,3,4]), np.array([1,2,3]))
    doses = np.array([0,1,2,3,4])
    ODs = np.array([0,0.477,0.564,0.602,0.623])
    cal.fit(doses, ODs)
    cal.inverse(0.5)


def test_inversion():
    cal = RationalCalibration()
    cal.a, cal.b, cal.c = 0.2,0.1,0.15
    dose = 5
    OD = cal.forward(dose)
    inverse_dose = cal.inverse(OD)
    assert np.isclose(inverse_dose, dose)


def test_fit():
    cal = RationalCalibration()
    a,b,c = 0.2,0.7,0.15
    doses = np.array([0,1,2,3,4])
    ODs = cal._forward(doses, a, b, c)
    cal.fit(doses, ODs)
    # check that the fit is accurate
    assert np.all(np.isclose([a,b,c], [cal.a,cal.b,cal.c], atol=1e-4, rtol=1e-4))
    # check the fit parameters are properly updated and called in the calibration
    calc_doses = cal.inverse(ODs)
    assert np.all(np.isclose(doses, calc_doses, atol=1e-4, rtol=1e-4))
    

