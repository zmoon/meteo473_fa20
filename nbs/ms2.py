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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import interact

import data

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
# ## Azimuthal average vertical cross-sections

# %% [markdown]
# > Plot vertical cross sections for the center of the eye out to a radius of 120 grid points of the azimuthally averaged vertical velocity, potential temperature, QRAIN, tangential wind speed (positive cyclonic) and radial wind speed (positive inward). Hint: use 6 km radial bins as a starting point, but tune this number up if your figure is noisy or down if it lacks detail.  Don’t expect smooth results near the eye and eye-wall where you’re averaging only a few points.

# %% [markdown]
# We need to define the center of the eye. Let's try minimum central pressure.

# %%
psfc_min = ds.psfc.isel(
    ds.psfc.argmin(dim=["lat", "lon"])
)  # gridpt-wise min (could interpolate tho?)
psfc_min

# %%
fig, ax = plt.subplots()
ds.psfc.plot(ax=ax)

ax.plot(
    psfc_min.lon,
    psfc_min.lat,
    "r*",
    ms=14,
    label=(
        f"min psfc\nvalue = {psfc_min.values:.0f} hPa\n"
        f"coords = ({psfc_min.lat.values:.5g}{ds.lat.units}, {psfc_min.lon.values:.5g}{ds.lon.units})"
    ),
)

ax.legend()

# %% [markdown]
# Now need to express our coordinates in terms of $r$ (wrt. $r_0$), $\theta$, so that we can do the radial binning.

# %%
# use 3.4 km
d_xy_const_km = 3.4

ds = data.add_xy_coords(ds, dx=d_xy_const_km, dy=d_xy_const_km, units="km")

psfc_min_ds = ds.isel(
    ds.psfc.argmin(dim=["lat", "lon"])
)  # gridpt-wise min (could interpolate tho?)
psfc_min_ds

# %%
# subtract the center
x0 = psfc_min_ds.x.values
y0 = psfc_min_ds.y.values
xrel = ds.x.values - x0 + 1e-13  # hack to avoid zero-division for now
yrel = ds.y.values - y0

# meshgrid
X, Y = np.meshgrid(xrel, yrel)

# calc r and theta
r = np.sqrt(X ** 2 + Y ** 2)
# since we are using gridpt-wise min instead of interpolating,
# we need to ignore that grid point for the theta calc
is_center = (X == 0) & (Y == 0)
# assert np.count_nonzero(is_center) == 1
theta = np.zeros(ds.psfc.shape)
theta[~is_center] = np.arctan2(Y[~is_center], X[~is_center])
# ^ arctan2 chooses correct quadrants for you

# Calculate tangential wind speed (positive cyclonic) and radial windspeed (positive inward)
# The below does work but uses more steps!
# uh = np.sqrt(ds.u**2 + ds.v**2)  # horizontal wind magnitude
# theta_wind = np.arctan2(ds.v.values, ds.u.values)  # wind vector angle (direction)
# urad = -uh*np.cos(theta_wind - theta)  # radial wind
# vtan = uh*np.sin(theta_wind - theta)  # tangential wind

# from Kelly, fewer steps way
vtan = np.cos(theta) * ds.v - np.sin(theta) * ds.u
urad = -(np.cos(theta) * ds.u + np.sin(theta) * ds.v)

# add radial and tangential velocity to the dataset
ds["urad"] = urad
ds.urad.attrs.update({"long_name": "$u$ radial wind (positive inward)", "units": ds.u.units})
ds["vtan"] = vtan
ds.vtan.attrs.update({"long_name": "$v$ tangential wind (positive cyclonic)", "units": ds.u.units})

# define the radius bins
rbins = np.arange(0, d_xy_const_km * (120 + 1), 4)
rbinsc = rbins[:-1] + 0.5 * np.diff(rbins)  # bin centers

# indices of which rbin the r at each x,y position belongs to
# returns index of the right bin edge
rbin_inds = np.digitize(r, rbins) - 1

# add to the dataset
ds["r"] = (("lat", "lon"), r, {"long_name": "radius (from TC center)", "units": "km"})
ds["i_rbin"] = (("lat", "lon"), rbin_inds, {"long_name": "index of radius bin", "units": ""})

# r will be a new coordinate variable / dim
rcoord_tup = (
    "r",
    rbinsc,
    {"long_name": "radial distance from center (radius bin center)", "units": "km"},
)


def vxs_radial(da, rbins=rbins):
    # manual binning (using Boolean indexing)
    # this way we can do it at all heights but not bin by height
    out = np.zeros((da.hgt.size, rbins.size - 1))
    for i, (r1, r2) in enumerate(zip(rbins[:-1], rbins[1:])):
        in_bin = (r >= r1) & (r < r2)  # leaving one side open for now
        out[:, i] = da.values[:, in_bin].mean(axis=1)

    return xr.DataArray(
        dims=["hgt", "r"],
        coords={"hgt": ds.hgt, "r": rcoord_tup},
        data=out,
        name=da.name,
        attrs=da.attrs,
    )


def vxs_radial_ds_groupby(ds, rbins=rbins):
    # This does it for all variables
    # Timing indicates that it is a bit slower than running vxs_radial for each variable
    ds_r = ds.groupby("i_rbin").mean()

    # add r as a coordinate variable
    # unforunately all of the data vars still have i_rbin as a coord
    ds_r = ds_r.assign_coords(coords={"r": rcoord_tup})

    return ds_r


# %% [markdown]
# Compare the times?

# %%
# TODO

# %%
not_too_high = ds.hgt <= 15000  # include 15000 for consistency with MS1

vns = ["w", "theta", "qrain", "urad", "vtan"]
fig, axs = plt.subplots(len(vns), 1, sharex=True, figsize=(8, 16))

for ax, vn in zip(axs.flat, vns):
    vxs_radial(ds[vn]).isel(hgt=not_too_high).plot.contourf(levels=40, ax=ax)
    if ax != axs.flat[-1]:
        ax.set_xlabel("")

# %% [markdown]
# ## Radial profile of OLR

# %% [markdown]
# > Plot azimuthally averaged OLR. Hint: use the same radial bin bin size as for the cross sections above.
#
# > Compare your height radius cross section of QRAIN to your radial profile of OLR.  At what geopotential height does the radial distribution of QRAIN best match the radial distribution of OLR?
#
# 👆 We can do one better than this: compare radial profiles for `qrain` at each level using `interact`.

# %%
def radial_prof(da, rbins):
    assert "hgt" not in da.dims  # this fn is for a surface variable or one level
    out = np.zeros((rbins.size - 1))
    for i, (r1, r2) in enumerate(zip(rbins[:-1], rbins[1:])):
        in_bin = (r >= r1) & (r < r2)  # leaving one side open for now
        out[i] = da.values[in_bin].mean()

    return xr.DataArray(
        dims=["r"],
        coords={"r": rcoord_tup},
        data=out,
        name=da.name,
        attrs=da.attrs,
    )


# interact
c1, c2 = "red", plt.cm.tab10.colors[0]
fig, ax = plt.subplots(figsize=(8, 5))
radial_prof(ds.olr, rbins).plot(ax=ax, c=c1)
ax.set_xlim(xmin=0)
ax2 = ax.twinx()

# ax2.spines["left"].set(color=c1, linewidth=2)
ax.yaxis.label.set_color(c1)
ax.tick_params(axis="y", colors=c1)

# ax2.spines["right"].set(color=c2, linewidth=2)
ax2.yaxis.label.set_color(c2)
ax2.tick_params(axis="y", colors=c2)


def plot(ilev):
    ax2.cla()
    radial_prof(ds.qrain.isel(hgt=ilev).squeeze(), rbins).plot(ax=ax2, c=c2)
    ax.set_title(f"ilev={ilev}, hgt={ds.hgt.values[ilev]} m")


interact(plot, ilev=(0, 30))

# %% [markdown]
# ## Vertical alignment of precip and OLR

# %% [markdown]
# > Use vertically stacked subplots to compare location of precipitation (surface QRAIN and composite QRAIN) and OLR.

# %%
# TODO
