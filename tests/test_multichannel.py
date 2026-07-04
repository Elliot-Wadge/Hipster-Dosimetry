from HipsterDosimetry import apply_calibration, RationalCalibration
import numpy as np


def make_grid(nx=101, ny=101, spacing=1.0):
    x = np.linspace(0, (nx-1)*spacing, nx)
    y = np.linspace(0, (ny-1)*spacing, ny)
    xv, yv = np.meshgrid(x, y)
    return x, y, xv, yv

def gaussian_2D(xx,yy,sigma,x0,y0):
    return np.exp(-((xx-x0)**2 + (yy-y0)**2)/2/sigma**2)


def get_calibration(a,b,c):
    cal = RationalCalibration()
    doses = np.array([0,1,2,3,4])
    ODs = cal._forward(doses, a, b, c)
    cal.fit(doses, ODs)
    return cal

def test_pure_dose_signal():
    x, y, xx, yy = make_grid()
    z = gaussian_2D(xx, yy, 10, 0, 0)*5+1

    b = -np.array([0.2, 0.4, 0.5])
    a = abs(b)*4+0.78
    c = np.array([0.2, 0.6, 0.2])
    disturbance = 1
    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)

    measured = np.stack((OD_1, OD_2, OD_3), axis=2)
    dose, delta, od = apply_calibration(measured, cal_1, cal_2, cal_3)
    assert np.all(dose.shape == measured.shape)
    assert np.all(od.shape == measured.shape)
    assert len(delta == np.prod(z.shape))
    print(np.sum(~np.isclose(np.ones(delta.shape), delta)))
    assert np.all(np.isclose(np.ones(delta.shape), delta))
    assert np.all(np.isclose(z,dose[:,:,0]))
    

def test_disturbed1_dose_signal():
    x, y, xx, yy = make_grid()
    z = gaussian_2D(xx, yy, 10, 0, 0)*3+1

    disturbance = 1.1
    b = -np.array([0.2, 0.4, 0.5])
    a = abs(b)*4+0.78
    c = np.array([0.2, 0.6, 0.2])

    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)

    measured = np.stack((OD_1, OD_2, OD_3), axis=2)
    dose, delta, od = apply_calibration(measured, cal_1, cal_2, cal_3)
    
    assert np.all(dose.shape == measured.shape)
    assert np.all(od.shape == measured.shape)
    assert len(delta == np.prod(z.shape))
    print(np.sum(~np.isclose(z,dose[:,:,0])))
    assert np.all(np.isclose(z,dose[:,:,0]))
    assert np.all(np.isclose(1/disturbance*np.ones(delta.shape), delta))


def test_disturbed2_dose_signal():
    x, y, xx, yy = make_grid()
    z = gaussian_2D(xx, yy, 10, 0, 0)*5+1

    disturbance = 0.85
    b = -np.array([0.2, 0.4, 0.5])
    a = abs(b)*7+0.78
    c = np.array([0.2, 0.6, 0.2])

    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)

    measured = np.stack((OD_1, OD_2, OD_3), axis=2)
    dose, delta, od = apply_calibration(measured, cal_1, cal_2, cal_3)
    
    assert np.all(dose.shape == measured.shape)
    assert np.all(od.shape == measured.shape)
    assert len(delta == np.prod(z.shape))
    print(np.sum(~np.isclose(z,dose[:,:,0])))
    print(measured[~np.isclose(z,dose[:,:,0])])
    assert np.all(np.isclose(z,dose[:,:,0]))
    assert np.all(np.isclose(1/disturbance*np.ones(delta.shape), delta))


def test_speed(benchmark):
    x, y, xx, yy = make_grid()
    z = gaussian_2D(xx, yy, 10, 0, 0)*3+1
    
    disturbance = 0.85
    b = -np.array([0.2, 0.4, 0.5])
    a = abs(b)*4+0.78
    c = np.array([0.2, 0.6, 0.2])


    cal_1 = get_calibration(a[0],b[0],c[0])
    OD_1 = disturbance*cal_1.forward(z)
    cal_2 = get_calibration(a[1],b[1],c[1])
    OD_2 = disturbance*cal_2.forward(z)
    cal_3 = get_calibration(a[2],b[2],c[2])
    OD_3 = disturbance*cal_3.forward(z)

    measured = np.stack((OD_1, OD_2, OD_3), axis=2)

    
    dose, delta, od = apply_calibration(measured, cal_1, cal_2, cal_3)
    
    def run_metric():
        reg = apply_calibration(measured, cal_1, cal_2, cal_3)
        return reg
    
    dose, delta, od = benchmark(run_metric)



if __name__ == '__main__':
    test_disturbed1_dose_signal()