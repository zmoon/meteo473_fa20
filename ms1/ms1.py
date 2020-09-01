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

r_e = 6.371e6  # mean Earth radius (m)


def latlon_to_xy(lat_deg, lon_deg):
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    y = r_e * lat_rad
    x = r_e * lon_rad * np.cos(lat_rad)
    return x, y


def xy_to_latlon(x_m, y_m):
    lat_rad = y_m / r_e
    lon_rad = x_m / (r_e * np.cos(lat_rad))
    lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad)
    return lat_deg, lon_deg


# convert the lat/lon meshgrid to x/y
X, Y = latlon_to_xy(Lat, Lon)
assert np.allclose(np.hstack((Lat, Lon)), np.hstack((xy_to_latlon(X, Y))))

# if X and Y are the grid cell centers,
# these are the distance between grid cell centers, not strictly the grid cell sizes
# since the diff then is really telling us 0.5*dX[i] + 0.5*dX[i+1], etc.
# and we know that that grid spacings are not constant in this data
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
# 👆 There is no variation in $\Delta y$ zonally (because $y$ only depends on lat, and our lat only varies in one dim), so subtracting the zonal mean zeroes it out. $\Delta x$, however, does have some noise, especially around 132 degE, as we saw in $\Delta$lon.

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
# just seeing what qrain_cmp looks like / testing log color norm with xarray
import matplotlib as mpl

plt.figure()
im = ds.qrain_cmp.plot(
    norm=mpl.colors.LogNorm(vmin=1e-8, vmax=0.015),
    vmin=None,
    vmax=None,  # mpl 3.3 gets mad because xarray is still passing these
    cmap="gnuplot",
)

# %% [markdown]
# Having computed what the true new grid should be, we can compare how data looks in that grid compared to assuming a constant $\Delta x$, $\Delta y$.
#
# The milestone instructions suggested 3 km. We saw that the actual values range from about 3.34 to 3.44 km. There are a few things we could do:
# * assign the center of our lat/lon grid as (0, 0)
#   - we have 300x300 so somewhere in between 150 and 151, if doing it gridpoint-wise
#   - taking the mean lat/lon would be fine for lon (zonal dimension) but not lat
# * assign the center of our calculated x/y grid (km) as (0, 0)
#   - again, gridpoint-wise this would between points 150 and 151
#   - could take the mean x, y and use that. might not be too bad if we did that and used the mean delta values as well to construct the new grid.
# * set lower left or upper right corner as (0, 0)
#   - this would be pretty boring
# * find the coordinates of the minimum central pressure and set that as (0, 0)
# * leave x, y as distance on the sphere from 0 deg. lat/lon

# %%
