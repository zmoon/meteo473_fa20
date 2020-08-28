# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %matplotlib widget

plt.rcParams.update(
    {"figure.autolayout": True,}
)

# %% [markdown]
# Load the data.

# %%
ds = xr.open_dataset("../data/data.nc")
ds

# %% [markdown]
# ## Grids
#
# > All figures for this and subsequent Milestones should have the axes labeled in physical units, i.e. kilometers (grid spacing is 3 km)
#
# Let's see what the km grid spacing actually would be (since this is a lat/lon grid, 3 km is presumably just an estimate).
#
# Note that we did check when creating the nc file that XLONG and XLAT only varied in one dimension.
# * lon only varies in column (2nd) dimension
#   e.g. `lon[0,:] - lon[-1,:]` is all zeros
#
# * lat only varies in row (1st) dimension
#   e.g. `lat[:,0] - lat[:,-1]` is all zeros

# %%
lat = ds.lat.values
lon = ds.lon.values

# equal spacing?
dlat = np.diff(lat)
dlon = np.diff(lon)

try:
    assert np.all(dlat == dlat[0])
    assert np.all(dlon == dlon[0])
except AssertionError:
    print("spacing is not equal...")


latc = lat[:-1] + 0.5 * dlat
lonc = lon[:-1] + 0.5 * dlon

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 4))

ax1.plot(latc, dlat)
ax1.set_xlabel("lat (deg.)")
ax1.set_ylabel(r"$\Delta$lat (deg.)")
ax1.grid(True)

ax2.plot(lonc, dlon)
ax2.set_xlabel("lon (deg.)")
ax2.set_ylabel(r"$\Delta$lon (deg.)")
ax2.grid(True)

fig.tight_layout()

# %% [markdown]
# 👆 The latitude spacing in degrees decreases with latitude, so as to keep the grid cells closer to square. The longitude spacing in degrees is mostly constant, with some blips that could be due to rounding error (the lat/lon are stored as float32) or could correspond to nested domain boundaries.
#
# Now we will proceed to calculate the grid cell sizes in meters (m). First, we convert the lat/lon coordinates to x/y points on the sphere, using the mean Earth radius.

# %%
# 300x300
Lon, Lat = np.meshgrid(lon, lat)  # note: still deg. here

Lon_rad, Lat_rad = np.deg2rad(Lon), np.deg2rad(Lat)

r_e = 6.371e6  # mean Earth radius (m)

Y = r_e * Lat_rad
X = r_e * Lon_rad * np.cos(Lat_rad)

dY = np.diff(Y, axis=0)
dX = np.diff(X, axis=1)

dLon = np.diff(Lon, axis=1)
dLat = np.diff(Lat, axis=0)

# TODO: lat/lon centers but by X,Y distance (m), not deg.
Lonc = Lon[:, :-1] + dLon
Latc = Lat[:-1, :] + dLat
# note the sizes are differnet here
# maybe should make both 299x299 at this point

# %%
# TODO: draw x/y grid in lat/lon space and lat/lon grid in x/y space
# to show the distortion. (need to decrease number of lines to 10x10 or something)

# %%
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=True)

# with pcolormesh we can specify the rectangle edges
#   for the deltas, these are the original lat/lon grid cell centers
# or specify the centers and use `shading="auto"`
# im = ax1.pcolormesh(Lonc[:-1, :], Latc[:, :-1], dX[1:, :]/1000, shading="auto")
im = ax1.pcolormesh(Lon, Lat, dX[1:, :] / 1000, shading="flat")
cb = plt.colorbar(im, ax=ax1, label="$\Delta x$ [km]")

im = ax2.pcolormesh(Lon, Lat, dY[:, 1:] / 1000, shading="flat")
cb = plt.colorbar(im, ax=ax2, label="$\Delta y$ [km]")

# %%
# difference from zonal mean
# (looks pretty much the same as subtracting first column)
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=True)

im = ax1.pcolormesh(
    Lon, Lat, (dX[1:, :] - dX[1:, :].mean(axis=1)[:, np.newaxis]) / 1000, shading="flat"
)
cb = plt.colorbar(im, ax=ax1, label="$\Delta x$ [km]")

im = ax2.pcolormesh(
    Lon, Lat, (dY[:, 1:] - dY[:, 1:].mean(axis=1)[:, np.newaxis]) / 1000, shading="flat"
)
cb = plt.colorbar(im, ax=ax2, label="$\Delta y$ [km]")


# %% [markdown]
# 👆 There is no variation in $\Delta y$ zonally, so subtracting the zonal mean zeroes it out. $\Delta x$, however, does have some noise, especially around 132 degE, as we saw in $\Delta$lon.

# %%
# TODO: could move to a utils module and add more features
def add121(ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xmin, xmax = xlim
    ymin, ymax = ylim
    xymin, xymax = min(xmin, ymin), max(xmax, ymax)
    ax.plot([xymin, xymax], [xymin, xymax], "-", c="0.7", zorder=1, label="1-1")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 9), sharex=True, sharey=True)

ax1.plot(dX.flat, dY.flat, ".", ms=2, alpha=0.3)
ax1.set(title="all points", ylabel="$\Delta y$ [km]")
add121(ax1)
ax1.legend()

ax2.plot(dX[:-1].mean(axis=1), dY.mean(axis=1), ".", ms=4, alpha=0.7)
ax2.set(title="zonal mean", ylabel="$\Delta y$ [km]", xlabel="$\Delta x$ [km]")
add121(ax2)

# %% [markdown]
# 👆 This scatterplot tells us that $\Delta x$ and $\Delta y$ (in meters) are almost the same everywhere in the lat/lon grid. The previous plot shows the grid cell sizes (in both $x$ and $y$) get smaller as latitude increases (i.e., as we move farther from the equator).

# %%
# just seeing what qrain_cmp looks like
import matplotlib as mpl

plt.figure()
im = ds.qrain_cmp.plot(
    norm=mpl.colors.LogNorm(vmin=1e-8, vmax=0.015),
    vmin=None,
    vmax=None,  # mpl still gets mad the xarray is passing these
    cmap="gnuplot",
)
