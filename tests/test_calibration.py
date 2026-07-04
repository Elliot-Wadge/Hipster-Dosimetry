from HipsterDosimetry.calibration import RationalCalibration, objective, objective_derivative, objective_second_derivative
import pytest
import numpy as np
import plotly.graph_objects as go


def get_calibration(a,b,c):
    cal = RationalCalibration()
    doses = np.array([0,1,2,3,4])
    ODs = cal._forward(doses, a, b, c)
    cal.fit(doses, ODs)
    return cal

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
    a,b,c = 2,-0.3,0.15
    doses = np.array([0,1,2,3,4])
    ODs = cal._forward(doses, a, b, c)
    cal.fit(doses, ODs, p0=[0.5,-0.1,0.1])
    # check that the fit is accurate
    assert np.all(np.isclose([a,b,c], [cal.a,cal.b,cal.c], atol=1e-4, rtol=1e-4))
    # check the fit parameters are properly updated and called in the calibration
    calc_doses = cal.inverse(ODs)
    assert np.all(np.isclose(doses, calc_doses, atol=1e-4, rtol=1e-4))
    

def test_derivatives():
    b = -np.array([0.2, 0.4, 0.5])
    a = abs(b)*4+0.78
    c = np.array([0.2, 0.6, 0.2])
    disturbance = 0.8577
    z = 0.2
    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)
    ODs = np.array([OD_1, OD_2, OD_3])
    disturbances = np.linspace(0.5,2,1000)
    scores = np.empty(disturbances.shape)
    dscores = np.empty(disturbances.shape)
    ddscores = np.empty(disturbances.shape)
    for i,disturbed in enumerate(disturbances):
        scores[i] = objective(ODs*disturbed, a, b, c)
        dscores[i] = objective_derivative(disturbed, ODs, a, b, c)
        ddscores[i] = objective_second_derivative(disturbed, ODs, a, b, c)

    gradient1 = np.gradient(scores, disturbances[1]-disturbances[0])
    gradient2 = np.gradient(dscores, disturbances[1]-disturbances[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=disturbances, y=scores, mode='lines', name='objective'))
    fig.add_trace(go.Scatter(x=disturbances, y=dscores, mode='lines', name='first'))
    fig.add_trace(go.Scatter(x=disturbances, y=ddscores, mode='lines', name='second'))
    fig.add_trace(go.Scatter(x=disturbances, y=gradient1, mode='lines', name='gradient1'))
    fig.add_trace(go.Scatter(x=disturbances, y=gradient2, mode='lines', name='gradient2'))
    fig.update_layout(yaxis_range=(-1,1))
    fig.show()
    gradient1 = np.gradient(scores, disturbances[1]-disturbances[0])
    gradient2 = np.gradient(dscores, disturbances[1]-disturbances[0])
    assert np.all(np.isclose(gradient1[1:-1], dscores[1:-1], atol=1e-3))
    assert np.all(np.isclose(gradient2[1:-1], ddscores[1:-1], atol=1e-3))

if __name__ == '__main__':
    test_derivatives()