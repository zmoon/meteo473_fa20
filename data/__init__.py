import numpy as np
import xarray as xr

from .create_nc import write_nc


def add_xy_coords(ds, dx, dy, *, units="km", center=True):
    """Add x and y coordinate variables to the lat/lon ones,
    for a grid with constant dx and dy.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset with `lat` and `lon` coordinate variables.
    dx, dy : float
        Constant x and y grid spacings.
    units : str, optional
        Units, default: km.
    center : bool, optional
        Whether to center the grid so (0,0) is in the middle of the domain.
        If false, (0,0) will be the bottom left corner (assuming the lat
        and lon coordinates are increasing).

    Returns
    -------
    xarray.Dataset
        With the new x and y coordinate variables.
    """
    n_x = ds.lon.size
    n_y = ds.lat.size
    x_const = np.arange(n_x, dtype=np.float) * dx
    y_const = np.arange(n_y, dtype=np.float) * dy

    if center:
        x_const -= x_const.mean()
        y_const -= y_const.mean()

    ds = ds.assign_coords(
        {
            "x": (
                "lon",
                x_const,
                {"long_name": rf"$x$ (const $\Delta x = {dx}$)", "units": units},
            ),
            "y": (
                "lat",
                y_const,
                {"long_name": rf"$y$ (const $\Delta y = {dy}$)", "units": units},
            ),
        }
    )

    return ds


def regrid_xy(ds):
    """Regrid from lat/lon grid to x/y grid."""
    raise NotImplementedError
