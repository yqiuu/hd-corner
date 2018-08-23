import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spl


def get_hdr(density, q = 68):
    inds = density > np.percentile(density, q = 100. - q)
    return inds, min(density[inds])


def get_hdr_bounds(data, density, q = 68):
    inds, p = get_hdr(density, q = q)
    hdr = data[inds]
    return min(hdr), max(hdr), p


def hdr1d(data, density, regions = [68, 10], norm = None, logScale = False, bins = 20, **kwargs):
    if np.isscalar(bins):
        bins = np.linspace(min(data), max(data), bins)
    elif type(bins) is dict:
        bins = bins[data.name]
    if norm is None:
        density = density/max(density)
    else:
        density = density/norm
    xp = bins[:-1] + np.diff(bins)/2.
    yp = np.zeros(len(xp))
    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        p = density[(data >= l) & (data < u)]
        yp[i] = max(p) if len(p) != 0 else 0.
    x = np.linspace(xp[0], xp[-1], 100)
    plt.plot(x, spl(xp, yp)(x), **kwargs)


def hdr2d(xData, yData, density, regions = [68, 10], norm = None, logScale = False, **kwargs):
    #if density is None:
    #    raise ValueError("density must be given.")
    # Remove possible duplicated parameters
    for key in ['c', 'color']:
        if key in kwargs:
            kwargs.pop(key)
    if norm is None:
        density = density/max(density)
    else:
        density = density/norm
    #        
    if regions is None:
        c = density
    else:
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


