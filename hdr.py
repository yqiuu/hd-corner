import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import InterpolatedUnivariateSpline as spl


def get_hdr(prob, q = 68):
    inds = prob > np.percentile(prob, q = 100. - q)
    if any(inds):
        return inds, min(prob[inds])
    else:
        maximum = max(prob)
        inds = prob == maximum
        return inds, maximum


def get_hdr_bounds(data, prob, q = 68):
    inds, p = get_hdr(prob, q = q)
    hdr = data[inds]
    return min(hdr), max(hdr), p


def plot_hdr1d(data, prob, bins = 20, smooth = True, **kwargs):
    if np.isscalar(bins):
        bins = np.linspace(min(data), max(data), bins)
    elif type(bins) is dict:
        bins = bins[data.name]
    xp = bins[:-1] + np.diff(bins)/2.
    yp = np.zeros(len(xp))
    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        p = prob[(data >= l) & (data < u)]
        yp[i] = max(p) if len(p) != 0 else 0.
    x = np.linspace(xp[0], xp[-1], 100)
    if smooth:
        plt.plot(x, spl(xp, yp)(x), **kwargs)
    else:
        plt.plot(xp, yp, **kwargs)


def plot_hdr2d(xData, yData, prob, regions = [10, 68], colors = None, **kwargs):
    # Add default parameters
    if 's' not in kwargs:
        kwargs['s'] = 1
    # Remove possible duplicated parameters
    for key in ['c', 'color']:
        if key in kwargs:
            kwargs.pop(key)
    #        
    if regions is None:
        inds = np.argsort(prob)
        plt.scatter(xData[inds], yData[inds], c = prob[inds], **kwargs)
    else:
        if colors is None:
            colors = sns.color_palette("GnBu", n_colors = len(regions))
        for q, c in zip(np.sort(regions)[::-1], colors):
            inds, _ = get_hdr(prob, q)
            if type(c) is not str:
                c = [c]
            plt.scatter(xData[inds], yData[inds], c = c, **kwargs)


def plot_hdr_bounds(xData, yData = None, prob = None, regions = [68], **kwargs):
    if prob is None:
        raise ValueError("prob must be given!")
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '--'
    for q in regions:
        lower, upper, _ = get_hdr_bounds(xData, prob, q)
        plt.axvline(lower, **kwargs)
        if upper != lower:
            plt.axvline(upper, **kwargs)
    if yData is not None:
        for q in regions:
            lower, upper, _ = get_hdr_bounds(yData, prob, q)
            plt.axhline(lower, **kwargs)
            if upper != lower:
                plt.axhline(upper, **kwargs)


def plot_best_fit(xData, yData = None, prob = None, best = None, kwargsDot = {}, **kwargs):
    if prob is not None:
        idx, _ = get_hdr(prob, q = 0.)
        bestX = np.unique(xData[idx])[0]
    elif best is not None:
        bestX = best[xData.name]
    else:
        raise ValueError("Either prob and best must be given!")
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


class corner(sns.PairGrid):
    def __init__(self, data, prob, quick = True, norm = None, **kwargs):
        if type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns = ['x%d'%i for i in range(data.shape[-1])])
        kwargs['diag_sharey'] = False
        kwargs['despine']     = False
        super().__init__(data, **kwargs)
        #
        if norm is None:
            prob = prob/max(prob)
        else:
            prob = prob/norm
        self.prob = prob
        #
        if quick:
            self.map_diag_with_prob(plot_hdr1d)
            self.map_lower_with_prob(plot_hdr2d)
        else:
            self.map_diag(self._do_nothing)
        yMax = 1.2*max(prob)
        for iA, ax in enumerate(self.diag_axes):
            ax.axis('on')
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible = False)
            ax.set_ylim(0, yMax)
            ax.set_yticks(np.linspace(0, yMax, 7)[1::2])
            ax.yaxis.tick_right()
            #
            ax = self.axes[iA, iA]
            ax.tick_params(axis = 'y', which = 'both', length = 0.)
            if iA == 0:
                plt.setp(ax.get_yticklabels(), visible = False)
        #
        ndim = len(self.axes)
        for iRow in range(ndim):
            for iCol in range(iRow + 1, ndim):
                self.axes[iRow, iCol].axis('off')
        plt.subplots_adjust(hspace = 0., wspace = 0.)


    def _add_axis_labels(self):
        # Disable the origianl class method
        pass

    
    def _do_nothing(self, xData = None, yData = None, **kwargs):
        pass

    
    def map_diag_with_prob(self, func, **kwargs):
        self.map_diag(func, prob = self.prob, **kwargs)


    def map_lower_with_prob(self, func, **kwargs):
        self.map_lower(func, prob = self.prob, **kwargs)
