import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.interpolate import InterpolatedUnivariateSpline as spl


def get_hdr(density, q = 68):
    inds = density > np.percentile(density, q = 100. - q)
    if any(inds):
        return inds, min(density[inds])
    else:
        maximum = max(density)
        inds = density == maximum
        return inds, maximum


def get_hdr_bounds(data, density, q = 68):
    inds, p = get_hdr(density, q = q)
    hdr = data[inds]
    return min(hdr), max(hdr), p


def hdr1d(data, density, norm = None, bins = 20, **kwargs):
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


def plot_hdr_bounds(xData, yData = None, density = None, regions = [68], **kwargs):
    if density is None:
        raise ValueError("density must be given!")
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '--'
    for q in regions:
        lower, upper, _ = get_hdr_bounds(xData, density, q)
        plt.axvline(lower, **kwargs)
        if upper != lower:
            plt.axvline(upper, **kwargs)
    if yData is not None:
        for q in regions:
            lower, upper, _ = get_hdr_bounds(yData, density, q)
            plt.axhline(lower, **kwargs)
            if upper != lower:
                plt.axhline(upper, **kwargs)


def plot_best_fit(xData, yData = None, density = None, best = None, kwargsDot = {}, **kwargs):
    if density is not None:
        idx, _ = get_hdr(density, q = 0.)
        bestX = np.unique(xData[idx])[0]
    elif best is not None:
        bestX = best[xData.name]
    else:
        raise ValueError("Either density and best must be given!")
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '--'
    plt.axvline(bestX, **kwargs)
    if yData is not None:
        try:
            bestY = np.unique(yData[idx])[0]
        except NameError:
            bestY = best[yData.name]
        plt.axhline(bestY, **kwargs)
        if len(kwargsDot) == 0:
           plt.plot(bestX, bestY, marker = 'o', markersize = 5)
        else:
           plt.plot(bestX, bestY, **kwargsDot)


def corner(data, density, upper = False, visible = False, cax = 'default',
           regions = [68, 10], norm = None, bins = 20, logScale = False,
           kwargs1d = {}, kwargs2d = {}, **kwargs):
    g = sns.PairGrid(data, **kwargs)
    g.map_diag(hdr1d, density = density, norm = norm, bins = bins, **kwargs1d)
    if upper:
        g.map_upper(hdr2d, density = density, regions = regions, norm = norm, logScale = logScale,
                    **kwargs2d)
    else:
        g.map_lower(hdr2d, density = density, regions = regions, norm = norm, logScale = logScale,
                    **kwargs2d)
    ndim = len(g.axes)
    for iRow in range(ndim):
        for iCol in range(ndim):
            if (iRow == ndim - 1 & iCol == ndim - 1):
                if norm is None:
                    maxY = 1.2
                else:
                    maxY = 1.2*max(density)/norm
                g.diag_axes[iRow].set_ylim(0., maxY)
            if visible:
                continue
            if upper:
                if iRow > iCol:
                    g.axes[iRow, iCol].axis('off')
            else:
                if iRow < iCol:
                    g.axes[iRow, iCol].axis('off')
    if cax is not None:
        if cax == 'default':
            cax = g.fig.add_axes([.98, .10, .02, .8])
        plt.colorbar(cax = cax, spacing = 'porporational')
        plt.subplots_adjust(right = .92)
    plt.subplots_adjust(hspace = 0., wspace = 0.)
    return g


