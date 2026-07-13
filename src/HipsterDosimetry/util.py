from pathlib import Path
import skimage as ski
import numpy as np

def combine_tif_images(dir:str):
    p = Path(dir)
    
    track_dictionary = {}
    all_files = list(p.glob('*.tif'))
    save_path = Path(dir + "combined/")
    save_path.mkdir(parents=True, exist_ok=True)
    


    for file in all_files:
        base_name = file.stem[:-3]


        if base_name in track_dictionary:
            continue
        track_dictionary[base_name] = True
        print(f'{base_name}.tif')
        matches = [
            f for f in all_files 
            if f.stem.startswith(base_name) and f.stem[-3:].isdigit()
        ]
        
        print(f"Found {len(matches)} matches: {[f.name for f in matches]}")
        
        if not matches:
            continue
        print(matches)
        images = np.array([ski.io.imread(str(match)) for match in matches])
        print(images.shape)
        combined_image = np.mean(images, axis=0)
        print(combined_image.shape)
        ski.io.imsave(save_path / Path(f'{base_name}_combined.tif'), combined_image)


if __name__ == '__main__':
    combine_tif_images('scans/calibration/')