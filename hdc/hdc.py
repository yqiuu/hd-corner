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
    'Corner'
]


def _set_default_params(kwargs, cmap = None):
    # Add default parameters
    if 's' not in kwargs:
        kwargs.update(s = 10)
    if 'marker' not in kwargs:
        kwargs.update(marker = 'o')
    # Remove possible duplicated parameters
    for key in ['c', 'color']:
        if key in kwargs:
            kwargs.pop(key)
    #
    if cmap is not None:
        kwargs.update(cmap = cmap)
    return kwargs


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


def plot_marginal2d(xData, yData, **kwargs):
    kwargs.update(plot_datapoints = False, plot_density = False, no_fill_contours = True)
    if 'color' not in kwargs:
        kwargs.update(color = 'k')
    hist2d(np.asarray(xData), np.asarray(yData), **kwargs)


def plot_hdr2d(xData, yData, prob, regions = [10, 68, 95], colors = None, **kwargs):
    kwargs = _set_default_params(kwargs)
    if colors is None:
        colors = sns.color_palette("Greys", n_colors = len(regions))
    for q, c in zip(np.sort(regions)[::-1], colors):
        inds, _ = get_hdr(prob, q)
        if type(c) is not str:
            c = [c]
        plt.scatter(xData[inds], yData[inds], c = c, **kwargs)


def plot_colormap(
    data_x, data_y, data_z, order='none', frac=1., scale='linear', **kwargs
):
    if order == 'ascending':
        inds = np.argsort(data_z)
    elif order == 'descending':
        inds = np.argsort(data_z)[::-1]
    elif order == 'none':
        inds = np.arange(len(data_z))
    else:
        raise ValueError(
            "Choose order from 'ascending', 'descending' and 'none'."
        )
    if scale == 'log':
        data_z = np.log10(data_z)
    elif scale == 'linear':
        pass
    else:
        raise ValueError("Choose scale from 'linear' and 'log'.")
    inds = inds[int((1 - frac)*len(data_z)):]
    return plt.scatter(data_x[inds], data_y[inds], c=data_z[inds], **kwargs)


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


class Corner:
    def __init__(self, ndim, mode=1, figsize=None):
        if mode == 0:
            ndim = ndim - 1
        if figsize == None:
            figsize = (3*ndim, 3*ndim)
        fig, axes = plt.subplots(
            figsize=figsize, nrows=ndim, ncols=ndim, sharex=True,
        )
        self.ndim = ndim
        self.fig = fig
        self.axes = axes
        self._mode = mode
        if mode == 0:
            self._origin = 0
            self.diag_axes = None
        else:
            self._origin = 1
            self.diag_axes = np.asarray([axes[i, i] for i in range(ndim)])
        if mode < 2:
            for i_row in range(ndim):
                for i_col in range(i_row + 1, ndim):
                    axes[i_row, i_col].axis('off')
        self._hide_yticklabels()
        plt.subplots_adjust(wspace=0., hspace=0.)


    def map_lower(self, func, data_xy, data_z=None, **kwargs):
        origin = self._origin
        for i_row in range(origin, self.ndim):
            for i_col in range(i_row - origin + 1):
                plt.sca(self.axes[i_row, i_col])
                if data_z is None:
                    func(
                        data_xy[:, i_col], data_xy[:, i_row - origin + 1],
                        **kwargs
                    )
                else:
                    func(
                        data_xy[:, i_col], data_xy[:, i_row - origin + 1],
                        data_z, **kwargs
                    )


    def map_upper(self, func, data_xy, data_z=None, **kwargs):
        if self._mode < 2:
            raise ValueError("Wrong mode to plot upper panels.")
        for i_row in range(self.ndim - 1):
            for i_col in range(i_row + 1, self.ndim):
                plt.sca(self.axes[i_row, i_col])
                if data_z is None:
                    func(data_xy[:, i_col], data_xy[:, i_row], **kwargs)
                else:
                    func(data_xy[:, i_col], data_xy[:, i_row], data_z, **kwargs)


    def map_diag(self, func, data_xy, data_z=None, **kwargs):
        for i_a, ax in enumerate(self.diag_axes):
            plt.sca(ax)
            if data_z is None:
                func(data_xy[:, i_a], **kwargs)
            else:
                func(data_xy[:, i_a], data_z, **kwargs)


    def set_labels(self, labels, **kwargs):
        if len(labels) != len(self.axes):
            raise ValueError("label number mismatch")
        # Set column labels
        for ax, label in zip(self.axes[1:, 0], labels[1:]):
            ax.set_ylabel(label, **kwargs)
        # Set row labels
        for ax, label in zip(self.axes[-1, :], labels):
            ax.set_xlabel(label, **kwargs)


    def set_ranges(self, ranges):
        for i_row in range(self.ndim):
            for i_col in range(self.ndim):
                if i_row != i_col:
                    self.axes[i_row, i_col].set_ylim(*ranges[i_col])
        for ax, r in zip(self.axes[-1], ranges):
            ax.set_xlim(*r)


    def set_diag_ylim(self, ranges):
        for ax, r in zip(self.diag_axes, ranges):
            ax.set_ylim(*r)


    def set_ticks(self, ticks, **kwargs):
        for ax, t in zip(self.axes[-1, :], ticks):
            ax.set_xticks(t, **kwargs)
        for ax, t in zip(self.axes[:, 0], ticks):
            ax.set_yticks(t, **kwargs)


    def set_diag_yticks(self, ticks):
        for ax, t in zip(self.diag_axes, ticks):
            ax.set_yticks(t)


    def add_caxis(self,
        rect=[0.1, 0.0, 0.8, 0.01], orientation='horizontal', **kwargs
    ):
        cax = self.fig.add_axes(rect)
        cbar = self.fig.colorbar(
            self._cplot, cax = cax, orientation = orientation, **kwargs
        )
        self.cax = cax
        self.cbar = cbar
        return cax, cbar


    def _hide_yticklabels(self):
        for ax in self.axes[:, 1:].flat:
            ax.set_yticklabels([])
