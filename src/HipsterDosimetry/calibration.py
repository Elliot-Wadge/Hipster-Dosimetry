import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class Calibration(ABC):

    @abstractmethod
    def forward(self, Dose):
        pass

    @abstractmethod
    def inverse(self, optical_density):
        pass

    @abstractmethod
    def fit(self, Doses, optical_densities):
        pass

    def __call__(self, optical_density:float|npt.NDArray[np.float]):
        return self.inverse(optical_density)



class RationalCalibration(Calibration):
    a:float|None=None
    b:float|None=None
    c:float|None=None

    def forward(self, Dose:float|npt.NDArray[np.float]):
        if not np.all([self.a, self.b, self.c]):
            raise ValueError('The calibration parameters have not been fitted, self.fit must be called first')
        return -np.log10((self.a+self.b*Dose)/(self.c+Dose))
    

    @staticmethod
    def _forward(Dose, a, b, c):
        return -np.log10((a+b*Dose)/(c+Dose))
    

    @staticmethod
    def _inverse(optical_density, a, b, c):
        return (c*10**(-optical_density)-a)/(b-10**(-optical_density))


    def inverse(self, optical_density:float|npt.NDArray[np.float]):
        if not np.all([self.a, self.b, self.c]):
            raise ValueError('The calibration parameters have not been fitted, self.fit must be called first')
        return (self.c*10**(-optical_density)-self.a)/(self.b-10**(-optical_density))
    

    def fit(self, Doses:npt.NDArray[np.float], optical_densities:npt.NDArray[np.float], **kwargs):
        
        if len(optical_densities) < 3:
            raise ValueError('At least three calibration points are required')
        if len(optical_densities) != len(Doses):
            raise ValueError(f'the len of optical density array ({len(optical_densities)}) must match the len of dose array ({len(Doses)})')
        
        p0 = kwargs.pop('p0', [0.1,0.1,0.1])
        pOpt, pCov = curve_fit(self._forward, Doses, optical_densities, p0=p0, **kwargs)
        self.a, self.b, self.c = pOpt

        return pOpt, pCov

