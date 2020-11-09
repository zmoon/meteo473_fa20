# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import interact
from utils import add121
from utils import subplots_share_labels

# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# %matplotlib widget

plt.rcParams.update(
    {
        "figure.autolayout": True,
    }
)

# %% [markdown]
# Load the data.

# %%
ds = xr.open_dataset("../data/data.nc")
ds

# %% [markdown]
# ## Vertical velocity maps
#
# Vertical velocity associated with convection

# %% [markdown]
# > Use pcolormesh to generate maps of vertical velocity. Create one map every 1000 geopotential meters from 1000 up to 29000.  Use these maps to inform and illustrate your answers to the following questions. [10 pts]
# >
# > * How high does the convection go (in kilometers).  [5 pts]
# > * Across what range of altitudes (in kilometers) can you pick out the gravity waves?  [5pts]
# > * What shape are the gravity wave patterns on these maps and does this shape evolve with height? If so, how? [5pts]

# %% [markdown]
# We can build an interactive selector like we have done before.

# %%
fig = plt.figure(figsize=(8, 6))


def plot_w_hgt(hgt=17000, symlog=False, symlog_linthresh=1.0, contourf=False, nlevs=60):
    da = ds.w.sel(hgt=hgt)
    fig.clf()
    ax = plt.axes()
    norm = None if not symlog else mpl.colors.SymLogNorm(linthresh=symlog_linthresh, base=10)
    fn = da.plot if not contourf else partial(da.plot.contourf, levels=nlevs)
    # note nlevs doesn't seem to work properly with SymLogNorm (actual # of levels come out less)
    fn(ax=ax, norm=norm)


interact(plot_w_hgt, hgt=(1000, 29000, 500), symlog_linthresh=(0.1, 2.0, 0.05), nlevs=(2, 200, 1))

# %% [markdown]
# Vertical cross-sections could help us see this in one figure, albeit for a limited part of the domain. That's kind of what we are doing in the next part though.

# %%
is_hgt_range = (ds.hgt >= 1000) & (ds.hgt <= 29000)

# recall 144 is the latitude of the gridpt-wise min in surface pressure
# from `ds.psfc.argmin(dim=["lat", "lon"])`
ds.w.isel(lat=144, hgt=is_hgt_range).plot.contourf(levels=60, size=5, aspect=1.8)

# %% [markdown]
# ## Convective gravity waves

# %% [markdown]
# > Generate southwest-to-northeast vertical cross sections of vertical velocity, potential temperature, and perturbation potential temperature (i.e. potential temperature minus its height-dependent average) through one of the strong horse-shoe shaped convectively generated gravity waves northwest of the eye. [20 pts]
# >
# > If these waves tilt, they are vertically propagating.  If they don’t, they are horizontally propagating.  Which is the case?  [5 pts]
#
# There is one horseshoe with center around 128.4 degLon, 12.2 degLat in the UT/LS region.

# %%
# Add dtheta to ds
ds["dtheta"] = ds["theta"] - ds.theta.mean(dim=["lat", "lon"])
ds.dtheta.attrs.update(
    long_name=r"Potential temperature difference wrt. height level mean $\theta'$", units="K"
)

# %%
# First method - going by gridpoints
ds_hsc = ds.sel(lon=128.4, lat=12.2, method="nearest")
ilat_hsc, ilon_hsc = (
    np.where(ds.lat.values == ds_hsc.lat.values)[0][0],
    np.where(ds.lon.values == ds_hsc.lon.values)[0][0],
)
print(ds_hsc.lat.values, ds_hsc.lon.values)

# Extend SW to NE
n_ph = 20  # number of points to either side of center point
ilat_hs = ilat_hsc + np.arange(-n_ph, n_ph + 1)
ilon_hs = ilon_hsc + np.arange(-n_ph, n_ph + 1)
lat_hs1 = ds.lat.isel(lat=ilat_hs)
lon_hs1 = ds.lon.isel(lon=ilon_hs)
print(lat_hs1.values)
print(lon_hs1.values, end="\n\n")

# Here are many attempts to do the vectorized indexing that don't work
# print(ds.hfx[xr.DataArray(ilat_hs, dims="lat"), xr.DataArray(ilon_hs, dims="lon")])
# ds.hfx[xr.DataArray(ilat_hs, dims="lat"), xr.DataArray(ilat_hs, dims="lat")]
# ds.hfx[xr.DataArray(ilat_hs, dims="lon"), xr.DataArray(ilat_hs, dims="lon")]
# print(ds.hfx.isel(lon=xr.DataArray(ilon_hs, dims="lon"), lat=xr.DataArray(ilon_hs, dims="lon")))
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilat_hs, dims="lat")))

# This one does work!
# xarray used to have `.sel_points` and `isel_points` that made this more straightforward and less weird, but no more.
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilon_hs, dims="lat")))
# Or this. Just have to keep the `dims` the same it seems.
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lon"), lon=xr.DataArray(ilon_hs, dims="lon")))

ds_hs1 = ds.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilon_hs, dims="lat"))
assert np.all(ds_hs1.lat.values == lat_hs1.values) and np.all(ds_hs1.lon.values == lon_hs1.values)

# %%
is_hgt_range = (ds.hgt > 13000) & (ds.hgt < 28000)

to_plot = ["w", "theta", "dtheta"]

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)

for vn, ax in zip(to_plot, axs.flat):
    ds_hs1[vn].isel(hgt=is_hgt_range).plot.contourf(levels=30, ax=ax)

# Prepare to label cross section end points
coords_a = ds_hs1.lat.values[0], ds_hs1.lon.values[0]
s_coords_a = f"({coords_a[0]:.2f}°N, {coords_a[1]:.2f}°E)"
coords_b = ds_hs1.lat.values[-1], ds_hs1.lon.values[-1]
s_coords_b = f"({coords_b[0]:.2f}°N, {coords_b[1]:.2f}°E)"

# Adjustments
for i, ax in enumerate(axs.flat):
    if i >= 1:
        ax.set_title(s_coords_a, loc="left", fontsize=10)
        ax.set_title(s_coords_b, loc="right", fontsize=10)
    if i < len(axs) - 1:
        ax.set_xlabel("")
    else:
        ax.set_xlabel(r"Latitude of SW$\to$NE gridpt-wise cross section")

# subplots_share_labels(axs)  # TODO: fix input to use asarray and use current fig if None and deal with 1 col or 1 row only
for ax in axs.flat:
    ax.label_outer()  # simpler than my fn...

fig.set_tight_layout(dict(h_pad=0.3))

# %% [markdown]
# 👆 We can see some waviness in the standard $\theta$ plot. It becomes much more apparent when subtracting the mean of each level ($\theta'$).

# %%
# Second method - interpolation, choosing A and B points and interpolating on line connecting them
# TODO

# %% [markdown]
# ## Momentum flux

# %% [markdown]
# > Select an area 100x100 grid points centered on one (for example the one at 70 EW and 205 NS) of these convective cells. Compute the momentum flux components (W*U and W*V) for all levels in this area.  Average over each level and plot the profiles of the U and V components of momentum flux as a function of height from 1 to 29 km of geopotential height.  Also plot the profiles of the direction and magnitude of the flux. [20 pts]
# >
# > Is the convective transport of momentum upward or downward (explain your reasoning)? Hint: think about the wind direction in your flux calculation box.  If U and V are of the same sign as Uflux and Vflux then the momentum transport is upward.  Note: you may find it helpful to plot U and V profiles as well before trying to work this out.  At what level is the convective momentum transport most extreme?  How closely does the top of the convective transport of momentum compare with the top of convection that you worked out above? [5 pts]

# %% [markdown]
# > Is the gravity wave transport of momentum upward or downward? Is it more extreme than the convective transport or less so?  [5 pts]

# %% [markdown]
# ## Shear

# %% [markdown]
# > Plot, for the entire domain, the shear vectors between 9000 geopotential meters and 20000 geopotential meters over vertical velocity at 20000 geopotential meters.  [10 pts]
# >
# > Use this figure to illustrate the discussion in your report about why gravity waves in the stratosphere over the typhoon’s convection are horseshoe shaped with orientation changing with azimuth from the storm center.  Also discuss why the stratospheric gravity waves are much more circular in the northeast and southeast parts of your domain.  [10 pts]
