import warnings

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


MEAN_EARTH_RADIUS = 6.371e6  # mean Earth radius (m)


def latlon_to_xy_sphere(lat_deg, lon_deg, r_e=MEAN_EARTH_RADIUS):
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    y = r_e * lat_rad
    x = r_e * lon_rad * np.cos(lat_rad)
    return x, y


def xy_to_latlon_sphere(x_m, y_m, r_e=MEAN_EARTH_RADIUS):
    lat_rad = y_m / r_e
    lon_rad = x_m / (r_e * np.cos(lat_rad))
    lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad)
    return lat_deg, lon_deg


# def regrid_latlon_to_xy(ds):
#     """Regrid from lat/lon grid to a similar x/y grid using xESMF."""
#     import xesmf as xe

#     # current grid
#     lat0 = ds.lat.values
#     lon0 = ds.lon.values
#     Lon0, Lat0 = np.meshgrid(lon0, lat0)

#     # desired grid
#     # for now, shooting for linearly spaced (in xy on sphere) between x and y boundaries
#     x0, y0 = latlon_to_xy_sphere(Lat0, Lon0)
#     x1, x2 = x0[0], x0[-1]
#     y1, y2 = y0[0], y0[-1]
#     nx, ny = x0.size, y0.size
#     x = np.linspace(x1, x2, nx)
#     y = np.linspace(y1, y2, ny)
#     lat

#     new_grid = xr.Dataset(
#         {
#             'lat': (['lat'], lat),
#             'lon': (['lon'], lon),
#         }
#     )

#     regridder = xe.Regridder(ds, new_grid, 'bilinear')

#     return regridder(ds)


def regrid_latlon_const(ds, *, nlat=None, nlon=None, method="bilinear"):
    """Regrid to a grid with the same lat/lon boundaries but constant dlat,dlon."""
    import xesmf as xe

    # to float64
    ds["lat"] = ds.lat.astype(np.float64)
    ds["lon"] = ds.lon.astype(np.float64)

    # original
    lat0 = ds.lat.values
    lon0 = ds.lon.values
    dlat0 = np.diff(lat0)
    dlon0 = np.diff(lon0)

    # total distance from the grid cell center on one boundary to the other
    # dlat_tot = abs(lat0[-1] - lat0[0])
    # dlon_tot = abs(lon0[-1] - lon0[0])

    assert np.all(dlat0 >= 0)  # monotonic increasing
    assert np.all(dlon0 >= 0)

    # number of points
    nlat = nlat or lat0.size
    nlon = nlon or lon0.size

    # ----
    # new grid -- evenly spaced in lat and lon

    # should be able to do this but it sometimes doesn't work
    # lat_edges = np.linspace(lat0[0] - 0.5 * dlat0[0], lat0[-1] + 0.5 * dlat0[-1], nlat + 1)
    # lon_edges = np.linspace(lon0[0] - 0.5 * dlon0[0], lon0[-1] + 0.5 * dlon0[-1], nlon + 1)

    # so ignoring some of the edge
    # lat_edges = np.linspace(lat0[0], lat0[-1], nlat+1)
    # lon_edges = np.linspace(lon0[0], lon0[-1], nlon+1)

    # centers from edges
    # lat = lat_edges[:-1] + 0.5 * np.diff(lat_edges)
    # lon = lon_edges[:-1] + 0.5 * np.diff(lon_edges)
    # TODO: need to properly make sure none of the new grid cell edges extend beyond old!
    # ESMF seems very sensitive to this

    # this is fine for non-conservative?
    lat = np.linspace(lat0[0], lat0[-1], nlat)
    lon = np.linspace(lon0[0], lon0[-1], nlon)

    # whatever makes it mess up seems to be related to points or edges outside of the original
    # but in the example going to global grid that happens and it works fine...
    if lon[0] < lon0[0] or lon[-1] > lon0[-1] or lat[0] < lat0[0] or lat[-1] > lat0[-1]:
        warnings.warn(
            "One of the new centers is outside the original.\n"
            f"Original: ({lat0[0]}, {lon0[0]}) ({lat0[-1]}, {lon0[-1]})  (lat, lon; lower-left and upper-right corners)\n"
            f"New:      ({lat[0]}, {lon[0]}) ({lat[-1]}, {lon[-1]})"
        )

    new_grid = xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
        }
    )
    # TODO: match attrs for lat/lon and dataset to original

    regridder = xe.Regridder(ds, new_grid, method=method, extrap_method="inverse_dist")
    # setting `extrap_method` seems to alleviate the issues was having (setting all to one value)
    # TODO: document what the options are/mean...

    return regridder(ds, keep_attrs=True)
