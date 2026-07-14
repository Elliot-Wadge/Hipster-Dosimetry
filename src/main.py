from HipsterDosimetry.util import combine_tif_images
from HipsterDosimetry import RationalCalibration, apply_calibration
import numpy as np
import plotly.graph_objects as go
import skimage as ski


if __name__ == '__main__':
    doses, red, green, blue = np.genfromtxt('scans/measurements/cal.csv', unpack=True, delimiter=',', skip_header=1)
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

    img = ski.io.imread('scans/measurements/combined/6X_a_10cm_combined.tif')
    img = np.log10(65535/img)
    dose, delta, od = apply_calibration(img, cal_r, cal_b, cal_g)
    print(dose.shape)
    print(delta.shape)
    print(od.shape)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=dose[:,:,0], name='dose', colorscale='gray'))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=delta, name='delta', colorscale='gray'))
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=od[:,:,0], name='od', colorscale='gray'))
    fig.show()
