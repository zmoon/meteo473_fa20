"""
utils
might move to data package or another later?
"""
import matplotlib.pyplot as plt
import numpy as np


def add121(ax, *, c="0.7", lw=2, limits="orig"):
    """Add 1-1 line to `ax`.

    Parameters
    ----------
    ax : plt.axes
    c :
        color
    lw : float
        line width
    limits : str {'orig', 'max'}
        Keep the original limits (`'orig'`) or take max range (`'max'`;
        in which case the axes should also be squared).

    """

    # original xlim
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # find max range
    xmin, xmax = xlim
    ymin, ymax = ylim
    xymin, xymax = min(xmin, ymin), max(xmax, ymax)

    # plot the line
    ax.plot([xymin, xymax], [xymin, xymax], "-", c=c, lw=lw, zorder=1, label="1-1")

    # reset lims to original
    if limits == "orig":
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.autoscale(enable=True, axis="both", tight=True)
        ax.axis("square")


def subplots_share_labels(fig=None, *, xlabel=None, ylabel=None):
    """For array of axes `axs`, properly share x and y labels
    (x on bottom only, y on left only).

    Note also

       for ax in axs.flat:
           ax.label_outer()

    which is simpler but only works if labels are in standard position.
    """
    # Get geom
    if fig is None:
        fig = plt.gcf()
    axs = fig.axes  # a list, not array
    m, n = axs[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry()

    # Form array
    axs_arr = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            axs_arr[i, j] = axs[i * n + j]

    # Try to get labels if they weren't provided
    ax0 = axs_arr[-1, 0] if axs_arr[0, 0].xaxis.get_label_position() == "bottom" else axs_arr[0, 0]
    if xlabel is None:
        xlabel = ax0.get_xlabel()
    if ylabel is None:
        ylabel = ax0.get_ylabel()

    # Need both
    if not (xlabel and ylabel):
        raise ValueError("xlabel and ylabel not provided or could not be detected")

    if ax0.xaxis.get_label_position() == "bottom":  # default
        ix = m - 1  # last row
    else:
        ix = 0

    if ax0.yaxis.get_label_position() == "left":  # default
        jy = 0  # first column
    else:
        jy = n - 1

    for i in range(m):
        for j in range(n):
            if j == jy:
                axs_arr[i, j].set_ylabel(ylabel)
            else:
                axs_arr[i, j].set_ylabel("")

            if i == ix:
                axs_arr[i, j].set_xlabel(xlabel)
            else:
                axs_arr[i, j].set_xlabel("")
