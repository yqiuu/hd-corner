import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from corner import quantile, hist2d


__all__ = [
    'get_hdr',
    'get_hdr_bounds',
    'plot_hdr1d',
    'plot_hdr2d',
    'plot_marginal2d',
    'plot_colormap',
    'plot_hdr_bounds',
    'plot_best_fit',
    'corner'
]


def get_hdr(prob, q = 68, weights = None):
    inds = prob > quantile(prob, q = 1. - q/100., weights = weights)
    if any(inds):
        return inds, min(prob[inds])
    else:
        maximum = max(prob)
        inds = prob == maximum
        return inds, maximum


def get_hdr_bounds(data, prob, q = 68, weights = None):
    inds, p = get_hdr(prob, q = q, weights = weights)
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


def set_default_params(kwargs):
    # Add default parameters
    if 's' not in kwargs:
        kwargs['s'] = 10
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    # Remove possible duplicated parameters
    for key in ['c', 'color']:
        if key in kwargs:
            kwargs.pop(key)
    return kwargs


def plot_marginal2d(xData, yData, **kwargs):
    kwargs.update(plot_datapoints = False, plot_density = False, no_fill_contours = True)
    if 'color' not in kwargs:
        kwargs.update(color = 'k')
    hist2d(np.asarray(xData), np.asarray(yData), **kwargs)


def plot_hdr2d(xData, yData, prob, regions = [10, 68, 95], colors = None, **kwargs):
    kwargs = set_default_params(kwargs)
    if colors is None:
        colors = sns.color_palette("Greys", n_colors = len(regions))
    for q, c in zip(np.sort(regions)[::-1], colors):
        inds, _ = get_hdr(prob, q)
        if type(c) is not str:
            c = [c]
        plt.scatter(xData[inds], yData[inds], c = c, **kwargs)


def plot_colormap(xData, yData, prob, frac = 100., **kwargs):
    grid = kwargs['_grid']; kwargs.pop('_grid')
    kwargs = set_default_params(kwargs)
    inds = np.argsort(prob)[int((1 - frac/100.)*len(prob)):]
    grid.cplot = plt.scatter(xData[inds], yData[inds], c = np.log10(prob[inds]), **kwargs)


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
        if len(kwargsDot) != 0:
           plt.plot(bestX, bestY, **kwargsDot)


class corner(sns.PairGrid):
    def __init__(self, data, prob, quick = True, norm = None, **kwargs):
        if type(data) is not pd.DataFrame:
            data = pd.DataFrame(data, columns = ['x%d'%i for i in range(data.shape[-1])])
        kwargs['diag_sharey'] = False
        kwargs['despine']     = False
        super().__init__(data, **kwargs)
        #
        if norm is not None:
            prob = prob/norm
        self.prob = prob
        #
        if quick:
            self.map_diag(sns.kdeplot, legend = False, color = 'k')
            self.map_diag_with_prob(plot_hdr1d)
            self.map_lower_with_prob(plot_hdr2d)
        else:
            self.map_diag(self._do_nothing)
        for iA, ax in enumerate(self.diag_axes):
            ax.axis('on')
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible = False)
            ax.yaxis.tick_right()
            ax.set_yticklabels([])
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
        #
        self.cax = None
        self.cplot = None


    def _add_axis_labels(self):
        # Disable the origianl class method
        pass

    
    def _do_nothing(self, xData = None, yData = None, **kwargs):
        pass

    
    def map_diag_with_prob(self, func, **kwargs):
        self.map_diag(func, prob = self.prob, **kwargs)


    def map_lower_with_prob(self, func, **kwargs):
        if func == plot_colormap:
            kwargs.update(_grid = self)
        self.map_lower(func, prob = self.prob, **kwargs)


    def add_caxis(self, rect = [0.1, 0.0, 0.8, 0.01], orientation = 'horizontal', **kwargs):
        cax = self.fig.add_axes(rect)
        self.fig.colorbar(self.cplot, cax = cax, orientation = orientation, **kwargs)
        self.cax = cax
        return cax


    def set_labels(self, labels, **kwargs):
        if len(labels) != len(self.axes):
            raise ValueError("label number mismatch")
        # Set column labels
        for ax, label in zip(self.axes[1:, 0], labels[1:]):
            ax.set_ylabel(label, **kwargs)
        # Set row labels
        for ax, label in zip(self.axes[-1, :], labels):
            ax.set_xlabel(label, **kwargs)


    def set_range(self, bounds):
        for ax, b in zip(self.axes[-1, :], bounds):
            ax.set_xlim(*b)
        for ax, b in zip(self.axes[:, 0], bounds):
            ax.set_ylim(*b)


    def set_diag_ylim(self, bounds):
        for ax, b in zip(self.diag_axes, bounds):
            ax.set_ylim(*b)


    def set_ticks(self, ticks, **kwargs):
        for ax, t in zip(self.axes[-1, :], ticks):
            ax.set_xticks(t, **kwargs)
        for ax, t in zip(self.axes[:, 0], ticks):
            ax.set_yticks(t, **kwargs)


    def set_diag_yticks(self, ticks):
        for ax, t in zip(self.diag_axes, ticks):
            ax.set_yticks(t)
