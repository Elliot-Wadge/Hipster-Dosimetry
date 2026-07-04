import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod
from numba.experimental import jitclass
from numba import njit


class Calibration(ABC):

    @abstractmethod
    def forward(self, Dose) -> float|npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def inverse(self, optical_density) -> float|npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def fit(self, Doses, optical_densities) -> tuple:
        pass


    def __call__(self, optical_density):
        return self.inverse(optical_density)


@njit
def numba_forward(Dose, a:float, b:float, c:float) -> float|npt.NDArray[np.float64]:
    return -np.log10((a+b*Dose)/(c+Dose))


@njit
def numba_inverse(optical_density:float|npt.NDArray, a:float|npt.NDArray, b:float|npt.NDArray, c:float|npt.NDArray) -> float|npt.NDArray[np.float64]:
    return (c*10**(-optical_density)-a)/(b-10**(-optical_density))


@njit
def first_derivative_inverse(distortion:float, optical_density:float|npt.NDArray, a:float|npt.NDArray, b:float|npt.NDArray, c:float|npt.NDArray):
    opt_d = optical_density*distortion
    t = 10.0**(-opt_d)
    return t*np.log(10)*optical_density*(a-c*b)/(b-t)**2


@njit
def second_derivative_inverse(distortion:float, optical_density:float|npt.NDArray, a:float|npt.NDArray, b:float|npt.NDArray, c:float|npt.NDArray):
    opt_d = optical_density*distortion
    t = 10.0**(-opt_d)
    return np.log(10)**2 * optical_density**2 * (c*b-a) * t * (b+t) / (b-t)**3


@njit
def objective(optical_density:npt.NDArray, a, b, c):
    Doses = numba_inverse(optical_density, a, b, c)
    return (Doses[0] - Doses[1])**2 + (Doses[0] - Doses[2])**2 + (Doses[2] - Doses[1])**2


@njit
def objective_derivative(distortion:float, optical_density:npt.NDArray, a:npt.NDArray, b:npt.NDArray, c:npt.NDArray) -> float:
    Doses = numba_inverse(distortion*optical_density, a, b, c)
    derivatives = first_derivative_inverse(distortion, optical_density, a, b, c)
    res = 2*((Doses[0] - Doses[1])*(derivatives[0]-derivatives[1]) + (Doses[0] - Doses[2])*(derivatives[0] - derivatives[2]) + (Doses[2] - Doses[1])*(derivatives[2] - derivatives[1]))
    return res


@njit
def objective_second_derivative(distortion:float, optical_density:npt.NDArray, a:npt.NDArray, b:npt.NDArray, c:npt.NDArray) -> float:
    Doses = numba_inverse(distortion*optical_density, a, b, c)
    first_derivatives = first_derivative_inverse(distortion, optical_density, a, b, c)
    second_derivatives = second_derivative_inverse(distortion, optical_density, a, b, c)
    res = 2 * ((first_derivatives[0] - first_derivatives[1])**2 + (Doses[0] - Doses[1])*(second_derivatives[0] - second_derivatives[1]) + \
               (first_derivatives[0] - first_derivatives[2])**2 + (Doses[0] - Doses[2])*(second_derivatives[0] - second_derivatives[2]) + \
               (first_derivatives[2] - first_derivatives[1])**2 + (Doses[2] - Doses[1])*(second_derivatives[2] - second_derivatives[1]))
    return res


class RationalCalibration(Calibration):

    a:float|None=None
    b:float|None=None
    c:float|None=None
    args:list=[]

    def forward(self, Dose:float|npt.NDArray[np.float64]) -> float|npt.NDArray[np.float64]:
        if self.a is None or self.b is None or self.c is None:
            raise ValueError('The calibration parameters have not been fitted, self.fit must be called first')
        return numba_forward(Dose, self.a, self.b, self.c)
    
    
    @staticmethod
    def _forward(Dose, a, b, c) -> float|npt.NDArray[np.float64]:
        return -np.log10((a+b*Dose)/(c+Dose))
    
    
    @staticmethod
    def _inverse(optical_density, a, b, c) -> float|npt.NDArray[np.float64]:
        return (c*10**(-optical_density)-a)/(b-10**(-optical_density))

    
    def inverse(self, optical_density:float|npt.NDArray[np.float64]) -> float|npt.NDArray[np.float64]:
        if self.a is None or self.b is None or self.c is None:
            raise ValueError('The calibration parameters have not been fitted, self.fit must be called first')
        return numba_inverse(optical_density, self.a, self.b, self.c)
    

    def fit(self, Doses:npt.NDArray[np.float64], optical_densities:npt.NDArray[np.float64], **kwargs) -> tuple:
        
        if len(optical_densities) < 3:
            raise ValueError('At least three calibration points are required')
        if len(optical_densities) != len(Doses):
            raise ValueError(f'the len of optical density array ({len(optical_densities)}) must match the len of dose array ({len(Doses)})')
        
        p0 = kwargs.pop('p0', [0.1,0.1,0.1])
        pOpt, pCov = curve_fit(self._forward, Doses, optical_densities, p0=p0, **kwargs)
        self.a, self.b, self.c = pOpt
        self.args = [self.a, self.b, self.c]
        return pOpt, pCov

