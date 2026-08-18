from pathlib import Path
import numpy as np
import numpy.typing as npt
import skimage as ski
import scipy

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
    LA = np.genfromtxt(correction_file, delimiter=',')
    LA = LA.T[:, np.newaxis, :]
    try:
        res = img / LA
    except ValueError:
        raise ValueError('The dimension of the lateral correction and the scanned image are not compatible. Ensure that the entire lateral range of the scan is included in both the lateral correction calculation and the scan being corrected.')

    return res



def process_dark_field(dir):
    correction_dir = Path(dir)
    images = list(correction_dir.glob('*.tif'))
    
    # Sort images based on top region mean (Channel 0)op_means = []
    top_means = []
    for file in images:
        img = ski.io.imread(file)
        top_means.append(np.mean(img[:len(img)//2, :, 0]))
    
    order = np.argsort(top_means)
    images = [images[idx] for idx in order]

    
    first_img = ski.io.imread(images[0])
    h, w, c = first_img.shape
    
    stitched_sum = np.zeros((c, h, w), dtype=np.float64)
    weight_sum = np.zeros((c, h, w), dtype=np.float64)

    for k, file in enumerate(images):
        raw_img = ski.io.imread(file)
        
        for i in range(c):
            img = raw_img[:, :, i].astype(np.float64)
            
            
            mask = img < 5e4
            mask = scipy.ndimage.binary_closing(mask, np.ones((30, 30)), iterations=1, border_value=1)
            mask = scipy.ndimage.binary_erosion(mask, np.ones((80, 80)), iterations=1, border_value=1)
            
            if not np.any(mask):
                continue

           
            weight = scipy.ndimage.distance_transform_edt(mask)
            if weight.max() > 0:
                weight = weight / weight.max()

            
            existing_mask = weight_sum[i] > 0
            overlap = existing_mask & mask
            
            if np.any(overlap) and k > 0:
                
                current_avg = stitched_sum[i][overlap] / weight_sum[i][overlap]
                incoming_avg = img[overlap]
                
                norm = np.mean(current_avg) / np.mean(incoming_avg)
                img_scaled = img * norm
            else:
                img_scaled = img

            stitched_sum[i] += img_scaled * weight
            weight_sum[i] += weight

    # Avoid division by zero where no masks covered
    valid_pixels = weight_sum > 0
    final_stitched = np.zeros_like(stitched_sum)
    final_stitched[valid_pixels] = stitched_sum[valid_pixels] / weight_sum[valid_pixels]

    return final_stitched
