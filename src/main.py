from HipsterDosimetry.util import combine_tif_images
from HipsterDosimetry import RationalCalibration, apply_calibration, objective, objective_derivative, objective_second_derivative
import numpy as np
import plotly.graph_objects as go
import skimage as ski
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import datetime

def apply_LA(file):
    img = ski.io.imread(file)

    shape = img.shape
    LA = np.genfromtxt('calibrations/LA.csv', delimiter=',')
    LA = LA.T[:, np.newaxis, :]
    res = img / LA
    test = np.ones(img.shape) / LA
    
    res = res.reshape((shape))
    ski.io.imsave('LA_correct.tif', res.astype(np.uint16))


def convert_image():
    doses, red, green, blue = np.genfromtxt('scans/measurements/cal_LA.csv', unpack=True, delimiter=',', skip_header=1)
    red = np.log10(65535/red)
    green = np.log10(65535/green)
    blue = np.log10(65535/blue)
    cal_r = RationalCalibration()
    cal_g = RationalCalibration()
    cal_b = RationalCalibration()
    cal_r.fit(doses, red)
    print(cal_r.args)
    cal_g.fit(doses, green)
    print(cal_g.args)
    cal_b.fit(doses, blue)
    print(cal_b.args)

    ddoses = np.linspace(np.min(doses), np.max(doses), 200)
    od_curve_r = cal_r.forward(ddoses)
    od_curve_g = cal_g.forward(ddoses)
    od_curve_b = cal_b.forward(ddoses)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=doses, y=red, name='red', mode='markers', marker_color='red'))
    fig.add_trace(go.Scatter(x=ddoses, y=od_curve_r, name='red', mode='lines', line_color='red'))
    fig.add_trace(go.Scatter(x=doses, y=green, name='green', mode='markers', marker_color='green'))
    fig.add_trace(go.Scatter(x=ddoses, y=od_curve_g, name='green', mode='lines', line_color='green'))
    fig.add_trace(go.Scatter(x=doses, y=blue, name='blue', mode='markers', marker_color='blue'))
    fig.add_trace(go.Scatter(x=ddoses, y=od_curve_b, name='blue', mode='lines', line_color='blue'))
    fig.show()

    img = ski.io.imread('LA_correct.tif')
    img = img[200:660,300:810]
    img = np.log10(65535/img)


    # disturbances = np.linspace(0.8,1.4,1000)
    # scores = np.empty(disturbances.shape)
    # dscores = np.empty(disturbances.shape)
    # ddscores = np.empty(disturbances.shape)
    # ODs = img[500,400]
    # a = np.array([cal_r.a, cal_g.a, cal_b.a])
    # b = np.array([cal_r.b, cal_g.b, cal_b.b])
    # c = np.array([cal_r.c, cal_g.c, cal_b.c])
    # print(cal_r.inverse(ODs[0]), cal_g.inverse(ODs[1]), cal_b.inverse(ODs[2]))

    # for i,disturbed in enumerate(disturbances):
    #     scores[i] = objective(ODs*disturbed, a, b, c)
    #     dscores[i] = objective_derivative(disturbed, ODs, a, b, c)
    #     ddscores[i] = objective_second_derivative(disturbed, ODs, a, b, c)

    # gradient1 = np.gradient(scores, disturbances[1]-disturbances[0])
    # gradient2 = np.gradient(dscores, disturbances[1]-disturbances[0])
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=disturbances, y=scores, mode='lines', name='objective'))
    # fig.add_trace(go.Scatter(x=disturbances, y=dscores, mode='lines', name='first'))
    # fig.add_trace(go.Scatter(x=disturbances, y=ddscores, mode='lines', name='second'))
    # fig.add_trace(go.Scatter(x=disturbances, y=gradient1, mode='lines', name='gradient1'))
    # fig.add_trace(go.Scatter(x=disturbances, y=gradient2, mode='lines', name='gradient2'))
    # fig.update_layout(yaxis_range=(-1,1))
    # fig.show()
    
    dose, delta, od = apply_calibration(img, cal_r, cal_g, cal_b)
    ski.io.imsave('dose.tif', dose[:,:,0].astype(np.uint16))
    np.savetxt('dose.csv', dose[:,:,0], delimiter=',')
    print(dose.shape)
    print(delta.shape)
    print(od.shape)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img[:,:,0], name='dose', colorscale='gray'))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=dose[:,:,0], name='dose', colorscale='gray'))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=delta, name='delta', colorscale='gray'))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=od[:,:,0], name='od', colorscale='gray'))
    fig.show()

def save_dose_to_rtdose(dose_array_cgy, filename="film_dose_rt.dcm", dpi=72):
    """
    Saves a 2D numpy array of dose (in cGy) as a standard-compliant DICOM RTDOSE file.
    
    Parameters:
    - dose_array_cgy: 2D numpy array (float) containing dose values.
    - filename: Output file path.
    - dpi: Scanner resolution (DPI) used to generate the film image.
    """
    # 1. Spatial Math: Convert DPI to pixel spacing in mm
    pixel_spacing_mm = 25.4 / dpi # ~0.3528 mm/pixel at 72 DPI
    
    # 2. Convert 2D array to 3D DICOM format: (Frames, Rows, Columns)
    # DICOM RTDOSE expects a 3D volume, even for a single 2D plane
    dose_3d = np.expand_dims(dose_array_cgy, axis=0) # shape becomes (1, Rows, Columns)
    print(dose_3d.shape)
    num_frames, rows, cols = dose_3d.shape
    
    # 3. Scaling float dose to integers
    # Store dose as integers to prevent float rounding issues in some DICOM viewers.
    # DoseGridScaling * PixelData_Integer = True Dose (in Gy)
    # Let's write the dose in Gy. If dose_array is cGy, we divide by 100 first.
    max_dose_cgy = np.max(dose_3d)
    max_dose_gy = max_dose_cgy / 100.0
    
    # We map the maximum dose to ~60,000 to maximize uint16 dynamic range
    integer_scale_limit = 60000
    dose_grid_scaling = max_dose_gy / integer_scale_limit
    
    # Scale and cast to uint16
    dose_gy = dose_3d / 100.0
    pixel_array_uint16 = (dose_gy / dose_grid_scaling).astype(np.uint16)

    # 4. Initialize File Metadata Header
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 222
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    # RT Dose Storage SOP Class UID
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2' 
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.2.135.102.1.1' # Standard open source root

    # 5. Build Main Dataset
    ds = pydicom.dataset.FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient Demographics & Study Info (DoseLab needs these populated)
    ds.PatientName = "Film^Dosimetry"
    ds.PatientID = "FILM-QA-001"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # DateTime elements
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.SeriesTime = dt.strftime('%H%M%S')
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')
    
    # 6. RTDOSE-Specific Coordinates & Geometry
    ds.Modality = "RTDOSE"
    # Dose values represent physical dose (GY)
    ds.DoseUnits = "GY"                
    ds.DoseType = "PHYSICAL"
    # Setting dose summation type as PLAN (standard default)
    ds.DoseSummationType = "PLAN"      
    
    # Set the Dose Grid Scaling factor
    ds.DoseGridScaling = dose_grid_scaling 
    
    # Geometry Definitions: Set origin [0,0,0] for the film center-left
    # ImagePositionPatient defines [x, y, z] of the top-left pixel of the first slice
    # We will center the grid relative to 0.0 in X and Y
    x_origin = -((cols * pixel_spacing_mm) / 2.0)
    y_origin = -((rows * pixel_spacing_mm) / 2.0)
    z_origin = 0.0
    
    ds.ImagePositionPatient = [x_origin, y_origin, z_origin]
    # Patient orientation: Row direction = X (+Left), Column direction = Y (+Posterior)
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] 
    
    # Grid spacing definitions
    ds.PixelSpacing = [pixel_spacing_mm, pixel_spacing_mm] # Row and column spacing
    ds.SliceThickness = "1.0"
    ds.NumberOfFrames = str(num_frames) # 1 frame for 2D planes
    ds.GridFrameOffsetVector = [0.0]    # Only 1 slice, so offset is 0.0
    
    # Grid sizing
    ds.Rows = rows
    ds.Columns = cols
    
    # 7. Image Pixel Module Definitions
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 # Unsigned integer (uint16)
    
    # Write pixel data
    ds.PixelData = pixel_array_uint16.tobytes()
    
    # 8. Write file to disk
    ds.save_as(filename, write_like_original=False)
    print(f"Successfully wrote RTDOSE DICOM file: '{filename}' ({cols}x{rows} @ {pixel_spacing_mm:.4f}mm spacing)")

def save_dose_as_dicom():
    dose = np.genfromtxt('dose.csv', delimiter=',')
    # dose = dose[200:660,300:810]
    print(dose.shape)
    save_dose_to_rtdose(dose)

if __name__ == '__main__':
    # combine_tif_images('scans/measurements/')
    convert_image()
    save_dose_as_dicom()
    # apply_LA('scans/measurements/combined/flattened_combined.tif')
    # apply_LA('scans/measurements/combined/6X_a_10cm_combined.tif')
    pass