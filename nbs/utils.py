"""
utils
might move to data package or another later?
"""
import matplotlib.pyplot as plt


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
