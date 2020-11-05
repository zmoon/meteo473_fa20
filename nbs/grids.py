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
import sys

sys.path.append("../")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import data
from utils import add121

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
# ## Grids
#
# In MS1, we are told to assume 3 km grid spacing. This notebook examines how faithful this is to the actual data.
#
# > All figures for this and subsequent Milestones should have the axes labeled in physical units, i.e. kilometers (grid spacing is 3 km)

# %% [markdown]
# ### Grid spacing in degrees
#
# Since we have only the coordinates of the grid cell centers, we can only examine the spacing between grid cells, not their sizes (it is possible to go from grid cell edges to centers, but not the other way around without making assumptions).
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

# %% [markdown]
# ### Grid spacing in kilometers
#
# Now we will proceed to calculate the grid cell sizes in meters (m). First, we convert the lat/lon coordinates to x/y points on the sphere, using the mean Earth radius.

# %%
# 300x300
Lon, Lat = np.meshgrid(lon, lat)  # note: still deg. here

# convert the lat/lon meshgrid to x/y
X, Y = data.latlon_to_xy_sphere(Lat, Lon)
assert np.allclose(np.hstack((Lat, Lon)), np.hstack((data.xy_to_latlon_sphere(X, Y))))

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
cb = plt.colorbar(im, ax=ax1, label=r"$\Delta x$ [km]")

im = ax2.pcolormesh(Lon, Lat, dY[:, 1:] / 1000, shading="flat")
cb = plt.colorbar(im, ax=ax2, label=r"$\Delta y$ [km]")

# %%
# difference from zonal mean
# (looks pretty much the same as subtracting first column)
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=True)

im = ax1.pcolormesh(
    Lon, Lat, (dX[1:, :] - dX[1:, :].mean(axis=1)[:, np.newaxis]) / 1000, shading="flat"
)
cb = plt.colorbar(im, ax=ax1, label=r"$\Delta x$ [km]")

im = ax2.pcolormesh(
    Lon, Lat, (dY[:, 1:] - dY[:, 1:].mean(axis=1)[:, np.newaxis]) / 1000, shading="flat"
)
cb = plt.colorbar(im, ax=ax2, label=r"$\Delta y$ [km]")


# %% [markdown]
# 👆 There is no variation in $\Delta y$ zonally (because $y$ only depends on lat, and our lat only varies in one dim), so subtracting the zonal mean zeroes it out. $\Delta x$, however, does have some noise, especially around 132 degE, as we saw in $\Delta$lon.

# %% [markdown]
# ### Grid spacing scatter

# %%
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)

ax1.plot(dX.flat, dY.flat, ".", ms=2, alpha=0.3)
ax1.set(title="all points", ylabel=r"$\Delta y$ [km]")
add121(ax1)
ax1.legend()

ax2.plot(dX[:-1].mean(axis=1), dY.mean(axis=1), ".", ms=4, alpha=0.7)
ax2.set(title="zonal mean", ylabel=r"$\Delta y$ [km]", xlabel=r"$\Delta x$ [km]")
add121(ax2)

# %% [markdown]
# 👆 This scatterplot tells us that $\Delta x$ and $\Delta y$ (in meters) are almost the same everywhere in the lat/lon grid. The previous plot shows the grid cell sizes (in both $x$ and $y$) get smaller as latitude increases (i.e., as we move farther from the equator).

# %% [markdown]
# ### Compare to true grid
#
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
d_xy_const_km = 3.0

x_const = np.arange(0, 300, dtype=np.float) * d_xy_const_km * 1000
x_const -= x_const.mean()
y_const = x_const.copy()
X_const, Y_const = np.meshgrid(x_const, y_const)

# true
X0, Y0 = data.latlon_to_xy_sphere(Lat, Lon)
X0_1 = X0 - np.mean(X0)  # subtract overall mean
Y0_1 = Y0 - np.mean(Y0)
X0_2 = X0 - np.mean(X0, axis=1)[:, np.newaxis]  # subtract zonal mean from each row
Y0_2 = Y0 - np.mean(Y0, axis=0)[np.newaxis, :]  # subtract meriodonal mean from each col

fig, [ax1, ax1_2, ax2] = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7, 10))

ax1.set_title("true (subtracting overall mean x and y values)")
ax1.pcolormesh(X0_1 / 1000, Y0_1 / 1000, ds.hfx, shading="auto")

ax1_2.set_title("true (accounting for shift in central x value etc.)")
ax1_2.pcolormesh(X0_2 / 1000, Y0_2 / 1000, ds.hfx, shading="auto")

ax2.set_title(f"using a constant grid spacing of {d_xy_const_km} km")
ax2.pcolormesh(X_const / 1000, Y_const / 1000, ds.hfx, shading="auto")
ax2.set_xlabel("$x$ (km) (difference from mean)")

for ax in [ax1, ax1_2, ax2]:
    ax.set_aspect("equal")
    ax.set_ylabel("$y$ (km) (difference from mean)")


# %% [markdown]
# 👆 We see that in the real data, the central $x$ values shift to the west as you move from south to north. If we change how we center the data to account for this, we can see the slight distortion due to the changing in grid cell size (from 3.44 to 3.34 km moving from south to north). Constant grid spacing of 3 km is a bit too small, we can see, but 3.4 km looks nice.

# %% [markdown]
# ## Add x/y as 2-d coordinates
#
# As we saw, since our data are on a lat/lon grid (and an unequal one at that), the $x$ and $y$ values will vary in both dimensions.
#
# We can add `X` and `Y` computed above as coordinates to our `ds`.

# %%
ds = ds.assign_coords(
    {
        "xs": (("lat", "lon"), X, {"long_name": "$x$ on Earth sphere", "units": "m"}),
        "ys": (("lat", "lon"), Y, {"long_name": "$y$ on Earth sphere", "units": "m"}),
    }
)

ds

# %%
ds.hfx.plot(x="xs", y="ys", size=4)

# %% [markdown]
# 👆 We see that we get the same tilt in the plot that we saw in the comparison above.
#
# 👇 This is consistent with the $x$ positions on the sphere changing in both the lat and lon directions.

# %%
ds.xs.plot(size=3)

# %% [markdown]
# ## Map projections
#
# We can use Cartopy to get a sense of the scale of our data domain and its position on the globe.

# %%
proj = ccrs.Robinson()

fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": proj})
ax.coastlines()
ax.set_extent([0, 180, 0, 90])
ax.gridlines(draw_labels=True)

p = ds.hfx.plot(
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),  # the data's projection
    ax=ax,
    cbar_kwargs=dict(orientation="horizontal", shrink=0.5, pad=0.07),
)

# %% [markdown]
# In projections with rectangular lat/lon cells, the domain looks square again (like it does in lat/lon space). Note that the `x="lon", y="lat"` we used above is not necessary here.
#
# For example:

# %%
proj = ccrs.Mercator()

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": proj})
ax.coastlines()
ax.gridlines(draw_labels=True)

p = ds.hfx.plot(
    transform=ccrs.PlateCarree(),  # the data's projection
    ax=ax,
    cbar_kwargs=dict(orientation="horizontal", shrink=0.5, pad=0.07),
)
