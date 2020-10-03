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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import interact

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
# TODO: move this stuff (copied from ms1) to a module, so can easily select this method or the more accurate one

# use 3.4 km
d_xy_const_km = 3.4

x_const = np.arange(0, 300, dtype=np.float) * d_xy_const_km
x_const -= x_const.mean()
y_const = x_const.copy()

ds2 = ds.assign_coords(
    {
        "x": (
            "lon",
            x_const,
            {"long_name": f"$x$ (const $\Delta x = {d_xy_const_km}$)", "units": "km"},
        ),
        "y": (
            "lat",
            y_const,
            {"long_name": f"$y$ (const $\Delta x = {d_xy_const_km}$)", "units": "km"},
        ),
    }
)

psfc_min_ds2 = ds2.isel(
    ds.psfc.argmin(dim=["lat", "lon"])
)  # gridpt-wise min (could interpolate tho?)
psfc_min_ds2

# %%
# subtract the center
x0 = psfc_min_ds2.x.values
y0 = psfc_min_ds2.y.values
xrel = ds2.x.values - x0 + 1e-13  # hack to avoid zero-division for now
yrel = ds2.y.values - y0

# meshgrid
X, Y = np.meshgrid(xrel, yrel)

# calc r and theta
r = np.sqrt(X ** 2 + Y ** 2)
# since we are using gridpt-wise min instead of interpolating,
# we need to ignore that grid point for the theta calc
is_center = (X == 0) & (Y == 0)
# assert np.count_nonzero(is_center) == 1
theta = np.zeros(ds2.psfc.shape)
theta[~is_center] = np.arctan2(
    Y[~is_center], X[~is_center]
)  # arctan2 chooses correct quadrants for you

# calculate tangential wind speed (positive cyclonic) and radial windspeed (positive inward)
# does work but more steps!
# uh = np.sqrt(ds.u**2 + ds.v**2)  # horizontal wind magnitude
# theta_wind = np.arctan2(ds.v.values, ds.u.values)  # wind vector angle (direction)
# urad = -uh*np.cos(theta_wind - theta)  # radial wind
# vtan = uh*np.sin(theta_wind - theta)  # tangential wind

# from Kelly, fewer steps way
vtan = np.cos(theta) * ds.v - np.sin(theta) * ds.u
urad = -(np.cos(theta) * ds.u + np.sin(theta) * ds.v)

ds["urad"] = urad
ds.urad.attrs.update({"long_name": "$u$ radial wind (positive inward)", "units": ds.u.units})
ds["vtan"] = vtan
ds.vtan.attrs.update({"long_name": "$v$ tangential wind (positive cyclonic)", "units": ds.u.units})


rbins = np.arange(0, d_xy_const_km * (120 + 1), 4)
rbinsc = rbins[:-1] + 0.5 * np.diff(rbins)

# r will be a new coordinate variable / dim
rcoord_tup = ("r", rbinsc, {"long_name": "radial distance from center", "units": "km"})

# not sure yet if can do this with xarray (maybe a groupby?) or a better way...
# so let's just loop over levels for now
def vxs_radial(da, rbins):
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


# check
# plt.figure()
# plt.pcolormesh(ds.lon, ds.lat, theta); plt.colorbar()
# ds.vtan.isel(hgt=2).plot(size=3)
# ds.urad.isel(hgt=2).plot(size=3)

not_too_high = ds.hgt < 15000

vns = ["w", "theta", "qrain", "urad", "vtan"]
fig, axs = plt.subplots(len(vns), 1, sharex=True, figsize=(8, 16))

for ax, vn in zip(axs.flat, vns):
    vxs_radial(ds[vn], rbins).isel(hgt=not_too_high).plot.contourf(levels=40, ax=ax)


# %% [markdown]
# ## Radial average of OLR

# %% [markdown]
# > Plot radially averaged OLR as a function of radius from center of R. Hint: use the same radial bin bin size as for the cross sections above.

# %%
def radial_prof(da, rbins):
    assert "hgt" not in da.dims
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
fig, ax = plt.subplots(figsize=(8, 5))
radial_prof(ds.olr, rbins).plot(ax=ax, c="red")
ax.set_xlim(xmin=0)
ax2 = ax.twinx()
ax2.spines["left"].set_color("red")
ax2.spines["right"].set_color(plt.cm.tab10.colors[0])


def plot(ilev):
    ax2.cla()
    radial_prof(ds.qrain.isel(hgt=ilev).squeeze(), rbins).plot(ax=ax2)
    ax.set_title(f"ilev={ilev}, hgt={ds.hgt.values[ilev]} m")


interact(plot, ilev=(0, 26))

# %% [markdown]
# ## Vertical alignment of precip and OLR?

# %% [markdown]
# > Use vertically stacked subplots to compare location of precipitation (surface QRAIN and composite QRAIN) and OLR.

# %%
