import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as spl 


def get_hdr(density, q = 68):
    inds = density > np.percentile(density, q = 100. - q)
    return inds, min(density[inds])


def get_hdr_bounds(data, density, q = 68):
    inds, p = get_hdr(density, q = q)
    hdr = data[inds]
    return min(hdr), max(hdr), p


def hdr1d(data, density, regions = [68, 10], norm = None, logScale = False, xBins = None,
               **kwargs):
    if xBins is None:
        xBins = np.linspace(min(data), max(data), 20)
    if norm is None:
        density = density/get_hdr(density, q = 68)[1]
    else:
        density = density/norm
    x = xBins[:-1] + np.diff(xBins)/2.
    y = np.zeros(len(x))
    for i, (l, u) in enumerate(zip(xBins[:-1], xBins[1:])):
        p = density[(data >= l) & (data < u)]
        y[i] = max(p) if len(p) != 0 else 0.
    plt.plot(x, y, **kwargs)


def hdr2d(xData, yData, density, regions = [68, 10], norm = None, logScale = False, **kwargs):
    #if density is None:
    #    raise ValueError("density must be given.")
    # Remove possible duplicated parameters
    for key in ['c', 'color']:
        if key in kwargs:
            kwargs.pop(key)
    if norm is None:
        density = density/get_hdr(density, q = 68)[1]
    else:
        density = density/norm
    #        
    c = np.full(len(density), -1.)
    for q in np.sort(regions)[::-1]:
        inds, p = get_hdr(density, q)
        c[inds] = p
    # Exclude all points beyond the largest region
    cond = c > 0.
    x = np.array(xData[cond])
    y = np.array(yData[cond])
    c = np.array(c[cond])
    if logScale:
        c = np.log(c)
    # Sort the data such that lower density regions are plotted first
    inds = np.argsort(c)
    x = x[inds]
    y = y[inds]
    c = c[inds]
    plt.scatter(x, y, c = c, **kwargs)


