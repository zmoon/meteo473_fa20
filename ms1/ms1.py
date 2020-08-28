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

# %% [markdown]
# Load the data.

# %%
ds = xr.open_dataset("../data/data.nc")
ds

# %% [markdown]
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

ax2.plot(lonc, dlon)
ax2.set_xlabel("lon (deg.)")
ax2.set_ylabel(r"$\Delta$lon (deg.)")

fig.tight_layout()

# %%
Lon, Lat = np.meshgrid(lon, lat)
