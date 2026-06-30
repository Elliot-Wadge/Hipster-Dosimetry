import numpy.typing as npt
import numpy as np
import itertools
from .calibration import Calibration
from scipy.optimize import minimize_scalar, differential_evolution, dual_annealing, shgo
from numba import njit


def objective_function(delta:float, optical_densities:np.array, calibrations:tuple[Calibration]):
    sum = 0
    indexing = [[0,1],[1,2],[0,2]]
    for pair in indexing:
        calibration1 = calibrations[pair[0]]
        calibration2 = calibrations[pair[1]]
        od1 = optical_densities[pair[0]]
        od2 = optical_densities[pair[1]]
        sum += (calibration1(delta*od1) - calibration2(delta*od2))**2
    return sum


def apply_calibration(measured_dose:npt.NDArray[np.float], 
                      calibration_r:Calibration, 
                      calibration_g:Calibration,
                      calibration_b:Calibration):
    
    flat_measured = measured_dose.reshape(-1, *measured_dose.shape[2:])
    
    deltas = np.empty(len(flat_measured), dtype=np.float64)
    Dose = np.empty((len(flat_measured),3), dtype=np.float64)
    optical_density = np.empty((len(flat_measured),3), dtype=np.float64)
    i = 0
    for eval in flat_measured:
        r,g,b = eval
        # res = minimize_scalar(objective_function, args=(np.array([r,g,b]), (calibration_r, calibration_g, calibration_b)), bounds=(0.8,1.2), tol=1e-8)
        # delta = res.x
        res = shgo(objective_function, 
                   bounds=[(0.8, 1.2)], 
                   options = dict(f_tol=1e-6),
                   args=(np.array([r,g,b]), (calibration_r, calibration_g, calibration_b)))
        delta = res.x[0]
        deltas[i] = delta
        Dose[i] = np.array([calibration_r(r*delta), calibration_g(g*delta), calibration_b(b*delta)])
        optical_density[i] = np.array([r*delta, g*delta, b*delta])

        i += 1

    Dose = Dose.reshape((measured_dose.shape))
    deltas = deltas.reshape(*measured_dose.shape[0:2])
    optical_density = optical_density.reshape((measured_dose.shape))
    return Dose, deltas, optical_density