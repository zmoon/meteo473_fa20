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
# %% [markdown]
# # Regridding
# %% [markdown]
# ```{warning}
# `xesmf` required!
# ```
#
# ```{margin}
# A margin note!
# ```
# %% tags=["hide-cell"]
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import data
from utils import add121

# #%matplotlib widget

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
# ## Regridding

# %%
# ESMF really has trouble with this dataset when points in the new grid are outside.
# which is supposed to be fine based on other examples, but somehow it breaks everything

ds2 = data.regrid_latlon_const(ds.isel(hgt=0), nlat=300, nlon=300)
# explicit grid cell bounds `lat_b` `lon_b` are needed to use the conservative algo

ds2.theta.plot(size=4)

# %%
ds.theta.isel(hgt=0).plot(size=4)

# %%
ds2.lat

# %%
ds.lat

# %%
ds2.lon

# %%
ds.lon

# %%
ds["lon"] = ds.lon.astype(np.float64)

# %%
ds.lon.dtype

# %%
ds.lat.dtype

# %% [markdown]
# ### xESMF example

# %%
import xesmf as xe

ds3 = xr.tutorial.open_dataset("rasm").rename({"xc": "lon", "yc": "lat"})

# ds3 = ds3.sel(x=ds3.x < 150)

new_grid = xe.util.grid_global(5, 4)

ds3_new = xe.Regridder(ds3, new_grid, "bilinear")(ds3)

ds3_new.isel(time=0).Tair.plot(x="lon", y="lat", size=4)

# %%
ds3.lon
