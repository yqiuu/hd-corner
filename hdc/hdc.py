import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
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


def get_hdr(prob, q=.68, weights=None):
    cond = prob > quantile(prob, q=1.-q, weights=weights)
    if any(cond):
        return cond, min(prob[cond])
    else:
        maximum = max(prob)
        cond = prob == maximum
        return cond, maximum


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


def plot_marginal2d(data_x, data_y, **kwargs):
    kwargs.update(plot_datapoints=False, plot_density=False, no_fill_contours=True)
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('levels', [.38, .68, .95])
    hist2d(np.asarray(data_x), np.asarray(data_y), **kwargs)


def plot_hdr2d(
    data_x, data_y, data_z, weights=None, regions=[.1, .68, .95], colors=None, **kwargs
):
    kwargs = _set_default_params(kwargs)
    if colors is None:
        colors = sns.color_palette("Greys", n_colors=len(regions))
    cond_arr = []
    cond_prev = np.full(len(data_z), False)
    for q in np.sort(regions):
        cond, _ = get_hdr(data_z, q, weights)
        cond_arr.append(cond & ~cond_prev)
        cond_prev = cond
    for cond, c in zip(reversed(cond_arr), colors):
        if type(c) is not str:
            c = [c]
        plt.scatter(data_x[cond], data_y[cond], c=c, **kwargs)


def plot_colormap(
    data_x, data_y, data_z, mode='none', frac=1., scale='linear', **kwargs
):
    if mode == 'maximize':
        inds = np.argsort(data_z)
    elif mode == 'minimize':
        inds = np.argsort(data_z)[::-1]
    elif mode == 'none':
        inds = np.arange(len(data_z))
    else:
        raise ValueError(
            "Choose mode from 'maximize', 'minimize' and 'none'."
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
    def __init__(self, ndim, no_diag=False, figsize=None):
        self._lbounds = np.full(ndim, np.inf, np.double)
        self._ubounds = np.full(ndim, -np.inf, np.double)
        if no_diag:
            ndim = ndim - 1
        if figsize == None:
            figsize = (2.5*ndim, 2.5*ndim)
        fig, axes = plt.subplots(figsize=figsize, nrows=ndim, ncols=ndim)
        self.ndim = ndim
        self.fig = fig
        self.axes = axes
        self._no_diag = no_diag
        if no_diag:
            self._origin = 0
            self.diag_axes = None
        else:
            self._origin = 1
            self.diag_axes = np.asarray([axes[i, i] for i in range(ndim)])
        if no_diag < 2:
            for i_row in range(ndim):
                for i_col in range(i_row + 1, ndim):
                    axes[i_row, i_col].axis('off')
        self._hide_yticklabels()
        plt.subplots_adjust(wspace=0., hspace=0.)


    def marginal_distributions(self,
        data_xy, color='b', figsize=None, kwargs_1d=None, kwargs_2d=None
    ):
        if kwargs_1d is None:
            kwargs_1d = {}
        kwargs_1d.setdefault('bins', 20)
        kwargs_1d.setdefault('color', color)
        kwargs_1d.setdefault('histtype', 'step')
        kwargs_1d.setdefault('density', True)
        if kwargs_2d is None:
            kwargs_2d = {}
        kwargs_2d.setdefault('color', color)
        self.map_diag(plt.hist, data_xy, **kwargs_1d)
        self.map_corner(plot_marginal2d, data_xy, **kwargs_2d)


    def colormaps(self,
        data_xy, data_z, mode='minimize', frac=1., scale='linear', **kwargs
    ):
        self.map_corner(
            plot_colormap, data_xy, data_z, mode=mode, frac=frac, scale=scale, **kwargs
        )


    def map_corner(self, func, data_xy, data_z=None, loc='lower', **kwargs):
        def plot(d_x, d_y):
            if data_z is None:
                self._plot = func(d_x, d_y, **kwargs)
            else:
                self._plot = func(d_x, d_y, data_z, **kwargs)

        origin = self._origin
        if loc == 'lower':
            for i_row in range(origin, self.ndim):
                for i_col in range(i_row - origin + 1):
                    plt.sca(self.axes[i_row, i_col])
                    plot(data_xy[:, i_col], data_xy[:, i_row - origin + 1])
        elif loc == 'upper':
            for i_row in range(self.ndim - 1):
                for i_col in range(i_row + 1, self.ndim):
                    plt.sca(self.axes[i_row, i_col])
                    plot(data_xy[:, i_col], data_xy[:, i_row])
        else:
            raise ValueError("Choose loc from 'lower' and 'upper'.")
        
        self.set_default_axes(data_xy)


    def map_diag(self, func, data_xy, data_z=None, **kwargs):
        for i_a, ax in enumerate(self.diag_axes):
            plt.sca(ax)
            if data_z is None:
                func(data_xy[:, i_a], **kwargs)
            else:
                func(data_xy[:, i_a], data_z, **kwargs)
        self.set_default_axes(data_xy)


    def set_default_axes(self, data_xy):
        if not self._no_diag:
            self.axes[0, 0].set_yticks([])
        #
        lbounds = np.min(data_xy, axis=0)
        lbounds = np.min(np.vstack([lbounds, self._lbounds]), axis=0)
        ubounds = np.max(data_xy, axis=0)
        ubounds = np.max(np.vstack([ubounds, self._ubounds]), axis=0)
        self.set_ranges(np.vstack([lbounds, ubounds]).T)
        self._lbounds = lbounds
        self._ubounds = ubounds
        n_ticks = 4
        #
        for ax in self.axes[-1]:
            plt.sca(ax)
            plt.xticks(rotation=45)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(n_ticks, prune='lower'))
        for ax in self.axes[:, 0]:
            plt.sca(ax)
            plt.yticks(rotation=45)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(n_ticks, prune='lower'))


    def set_labels(self, labels, **kwargs):
        origin = self._origin
        naxis = self.ndim
        if len(labels) != naxis + 1 - origin :
            raise ValueError("label number mismatch")
        # Set column labels
        for ax, label in zip(self.axes[origin:, 0], labels[1:]):
            ax.set_ylabel(label, **kwargs)
        # Set row labels
        for ax, label in zip(self.axes[-1, :naxis], labels[:naxis]):
            ax.set_xlabel(label, **kwargs)


    def set_ranges(self, ranges):
        origin = self._origin
        for i_row in range(self.ndim):
            for i_col in range(self.ndim):
                ax = self.axes[i_row, i_col]
                ax.set_xlim(*ranges[i_col])
                if i_row != i_col or origin == 0:
                    ax.set_ylim(*ranges[i_row - origin + 1])


    def set_diag_ylim(self, ranges):
        for ax, r in zip(self.diag_axes, ranges):
            ax.set_ylim(*r)


    def set_ticks(self, ticks, **kwargs):
        origin = self._origin
        for i_row in range(self.ndim):
            for i_col in range(self.ndim):
                ax = self.axes[i_row, i_col]
                ax.set_xticks(ticks[i_col])
                if i_row != i_col or origin == 0:
                    ax.set_yticks(ticks[i_row - origin + 1], **kwargs)


    def set_diag_yticks(self, ticks):
        for ax, t in zip(self.diag_axes, ticks):
            ax.set_yticks(t)


    def add_caxis(self,
        rect=[0.1, 0.0, 0.8, 0.01], orientation='horizontal', **kwargs
    ):
        cax = self.fig.add_axes(rect)
        cbar = self.fig.colorbar(
            self._plot, cax=cax, orientation=orientation, **kwargs
        )
        self.cax = cax
        self.cbar = cbar
        return cax, cbar


    def _hide_yticklabels(self):
        for ax in self.axes[:, 1:].flat:
            ax.set_yticklabels([])
