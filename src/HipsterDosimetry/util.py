from pathlib import Path
import skimage as ski
import numpy as np

def combine_tif_images(dir:str):
    p = Path(dir)
    
    track_dictionary = {}
    for file in p.glob('*.tif'):
        name = str(file).split('.tif')[0][:-3]
        exists = track_dictionary.get(name)
        
        if exists:
            continue

        print(f'{name}[0-9][0-9][0-9].tif')
        matches = p.glob(f'*.tif')
        print(list(matches))
        images = np.array([ski.io.imread(match) for match in matches])
        combined_image = np.mean(images, axis=0)
        print(combined_image.shape)
        ski.io.imsave(f'{name}_combined.tif', combined_image)


if __name__ == '__main__':
    combine_tif_images('scans/calibration/')