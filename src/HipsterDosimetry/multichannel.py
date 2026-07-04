import numpy.typing as npt
import numpy as np
from .calibration import Calibration, objective_derivative, objective_second_derivative
from scipy.optimize import minimize_scalar, differential_evolution, dual_annealing, shgo
from .optimizer import newton_raphson
from numba import njit
import time

def make_dobj(OD, a, b, c):
    @njit
    def f(x):
        return objective_derivative(x, OD, a, b, c)
    return f

def make_ddobj(OD, a, b, c):
    @njit
    def f(x):
        return objective_second_derivative(x, OD, a, b, c)
    return f



def objective_function(delta:float, optical_densities:npt.NDArray[np.float64], calibrations:tuple[Calibration]):
    s = 0
    indexing = [[0,1],[1,2],[0,2]]
    for pair in indexing:
        calibration1 = calibrations[pair[0]]
        calibration2 = calibrations[pair[1]]
        od1 = optical_densities[pair[0]]
        od2 = optical_densities[pair[1]]
        s += (calibration1(delta*od1) - calibration2(delta*od2))**2
    return s


def apply_calibration(measured_dose:npt.NDArray[np.float64], 
                      calibration_r:Calibration, 
                      calibration_g:Calibration,
                      calibration_b:Calibration):
    
    flat_measured = measured_dose.reshape(-1, *measured_dose.shape[2:])
    
    deltas = np.empty(len(flat_measured), dtype=np.float64)
    Dose = np.empty((len(flat_measured),3), dtype=np.float64)
    optical_density = np.empty((len(flat_measured),3), dtype=np.float64)
    i = 0
    a = np.array([calibration_r.a, calibration_g.a, calibration_b.a])
    b = np.array([calibration_r.b, calibration_g.b, calibration_b.b])
    c = np.array([calibration_r.c, calibration_g.c, calibration_b.c])
    
    for rgb in flat_measured:
        red,green,blue = rgb
        
        # res = minimize_scalar(objective_function, args=(np.array([r,g,b]), (calibration_r, calibration_g, calibration_b)), bounds=(0.8,1.2), tol=1e-8)
        # delta = res.x
        # res = shgo(objective_function, 
        #            bounds=[(0.8, 1.2)], 
        #            options = dict(f_tol=1e-6),
        #            args=(np.array([r,g,b]), (calibration_r, calibration_g, calibration_b)))
        # delta = res.x[0]
        delta = newton_raphson(1, rgb, a, b, c, 1e-12, np.array([0.5, 1.5]))
        deltas[i] = delta
        Dose[i] = np.array([calibration_r(red*delta), calibration_g(green*delta), calibration_b(blue*delta)])
        optical_density[i] = np.array([red*delta, green*delta, blue*delta])

        i += 1

    Dose = Dose.reshape((measured_dose.shape))
    deltas = deltas.reshape(*measured_dose.shape[0:2])
    optical_density = optical_density.reshape((measured_dose.shape))
    return Dose, deltas, optical_density