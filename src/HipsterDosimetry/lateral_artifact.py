from pathlib import Path
import numpy as np
import numpy.typing as npt
import skimage as ski

def apply_LA_correction(target_file:str, correction_file:str) -> npt.NDArray:
    '''correct a scanned film for the lateral artifact, this effect is corrected before converting to OD and is done 
    on a channel by channel basis. The red channel has the largest lateral artifacts. For the correction to be applied the scan
    must cover the entire lateral range of the scanner.
    
    :param target_file: the filename of the scan to be corrected
    :param correction_file: the filename of the file containing the lateral corrections for the three channels'''

    target_path = Path(target_file)
    correction_path = Path(correction_file)

    if not target_path.is_file():
        raise ValueError(f'{target_file} does not exist')
    
    if not correction_path.is_file():
        raise ValueError(f'{correction_file} does not exist')
    
    img = ski.io.imread(target_path)
    LA = np.genfromtxt('calibrations/LA.csv', delimiter=',')
    LA = LA.T[:, np.newaxis, :]
    try:
        res = img / LA
    except ValueError:
        raise ValueError('The dimension of the lateral correction and the scanned image are not compatible. Ensure that the entire lateral range of the scan is included in both the lateral correction calculation and the scan being corrected.')

    return res


def make_LA_correction(director:str):
    pass
    
