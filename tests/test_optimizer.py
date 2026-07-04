from HipsterDosimetry.calibration import RationalCalibration, objective, objective_derivative, objective_second_derivative
from HipsterDosimetry.optimizer import newton_raphson
import numpy as np
from numba import njit


def get_calibration(a,b,c):
    cal = RationalCalibration()
    doses = np.array([0,1,2,3,4])
    ODs = cal._forward(doses, a, b, c)
    cal.fit(doses, ODs)
    return cal


def make_do(OD, a, b, c):
    @njit
    def f(x):
        return objective_derivative(x, OD, a, b, c)
    return f


def make_ddo(OD, a, b, c):
    @njit
    def f(x):
        return objective_second_derivative(x, OD, a, b, c)
    return f


def test_root_find():
    
    b = np.array([0.2, 0.4, 0.5])
    a = b*4+0.78
    c = np.array([0.2, 0.6, 0.2])
    disturbances = np.random.random((500,)) + 0.5


    for disturbance in disturbances:
        z = 0.5
        cal_1 = get_calibration(a[0],b[0],c[0])
        OD_1 = disturbance*cal_1.forward(z)
        cal_2 = get_calibration(a[1],b[1],c[1])
        OD_2 = disturbance*cal_2.forward(z)
        cal_3 = get_calibration(a[2],b[2],c[2])
        OD_3 = disturbance*cal_3.forward(z)
        ODs = np.array([OD_1, OD_2, OD_3])
        x = newton_raphson(1, ODs, a, b, c, 1e-8, np.array([0.3, 2]))
        assert np.isclose(1/disturbance, x, atol=1e-8)


def test_root_find_speed(benchmark):
    b = np.array([0.2, 0.4, 0.5])
    a = b*4+0.78
    c = np.array([0.2, 0.6, 0.2])
    disturbance = 1
    z = 3
    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)
    ODs = np.array([OD_1, OD_2, OD_3])
    disturbances = np.linspace(0.5,1.2,1000)
    scores = np.empty(disturbances.shape)
    dscores = np.empty(disturbances.shape)
    ddscores = np.empty(disturbances.shape)


    # warmup
    x = newton_raphson(1, ODs, a, b, c, 1e-8, np.array([0.5, 1.5]))
    def run():
        newton_raphson(1, ODs, a, b, c, 1e-8, np.array([0.5, 1.5]))

    benchmark(run)


if __name__ == '__main__':
    test_root_find()