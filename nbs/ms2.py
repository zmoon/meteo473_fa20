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

import matplotlib.colors as colors
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
# ### Finding the center
#
# We need to define the center of the eye. Let's try minimum central pressure.

# %%
psfc_min = ds.psfc.isel(ds.psfc.argmin(dim=["lat", "lon"]))  # gridpt-wise min
psfc_min

# %%
# corresponding indices: ilat, ilon
np.where(ds.lat.values == psfc_min.lat.values)[0][0], np.where(
    ds.lon.values == psfc_min.lon.values
)[0][0]
# ^ probably there is a better way to do this?...

# %% [markdown]
# Let's compare this to the central indices, which is what George was telling them to use in class. Our grid is 300x300, both even, so we can't actually get a central point: but we can average the two sets of equally central points.

# %%
loc_min_by_inds = ds.lon[[149, 150]].values.mean(), ds.lat[[149, 150]].values.mean()
loc_min_by_inds

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
# Let's zoom in more and compare to the index-wise one.

# %%
fig, ax = plt.subplots(figsize=(8, 6))

psfc_zoom = ds.psfc.isel(lat=slice(119, 180), lon=slice(119, 180))

psfc_zoom.plot(ax=ax)

cs = psfc_zoom.plot.contour(levels=20, colors="0.5", linewidths=1, ax=ax)

icontour = 8
Xci = cs.allsegs[icontour][0]
ax.plot(*Xci.T, c="orange", lw=2, label=f"selected contour\nlevel = {cs.levels[icontour]}")


def center_of_mass(X):
    # calculate center of mass of a closed polygon; https://stackoverflow.com/a/48172492
    x = X[:, 0]
    y = X[:, 1]
    g = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * g.sum()
    cx = ((x[:-1] + x[1:]) * g).sum()
    cy = ((y[:-1] + y[1:]) * g).sum()
    return 1.0 / (6 * A) * np.array([cx, cy])


xcm_ci = center_of_mass(Xci)

ax.plot(
    *xcm_ci,
    "*",
    c="orange",
    ms=14,
    label=(
        "Centroid of selected contour\n"
        f"coords = ({xcm_ci[1]:.5g}{ds.lat.units}, {xcm_ci[0]:.5g}{ds.lon.units})"
    ),
)

ax.plot(
    psfc_min.lon,
    psfc_min.lat,
    "r*",
    ms=14,
    label=(
        f"min psfc on the grid\nvalue = {psfc_min.values:.0f} hPa\n"
        f"coords = ({psfc_min.lat.values:.5g}{ds.lat.units}, {psfc_min.lon.values:.5g}{ds.lon.units})"
    ),
)

ax.plot(
    *loc_min_by_inds,
    "m*",
    ms=14,
    label=(
        "Average of the index-wise central lat/lons\n"
        f"coords = ({loc_min_by_inds[1]:.5g}{ds.lat.units}, {loc_min_by_inds[0]:.5g}{ds.lon.units})"
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
# note that cos(theta) is equivalent to x/r and sin(theta) to y/r,
# so we could've gotten away with not calculating theta at all
vtan = np.cos(theta) * ds.v - np.sin(theta) * ds.u
urad = -(np.cos(theta) * ds.u + np.sin(theta) * ds.v)
# note also that if we already had u_h, we only really need to calculate one,
# then we can use Pythag with u_h to get the other one

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


# %%
# bit of a check
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
ds.urad.isel(hgt=0).plot.pcolormesh(ax=ax1)
ds.vtan.isel(hgt=0).plot.pcolormesh(ax=ax2)

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
    qs = [0.25, 0.5, 0.75]
    out_q = np.zeros((rbins.size - 1, len(qs)))
    for i, (r1, r2) in enumerate(zip(rbins[:-1], rbins[1:])):
        in_bin = (r >= r1) & (r < r2)  # leaving one side open for now
        data_in_bin = da.values[in_bin]
        out[i] = data_in_bin.mean()
        out_q[i] = np.quantile(data_in_bin, qs)

    return xr.Dataset(
        coords={
            "r": rcoord_tup,
            "q": ("q", qs, {"long_name": "quantile"}),
        },
        data_vars={f"{da.name}_mean": ("r", out, da.attrs), f"{da.name}_q": (("r", "q"), out_q)},
    )


# interact
c1, c2 = "red", plt.cm.tab10.colors[0]
fig, ax = plt.subplots(figsize=(8, 5))
dsr_olr = radial_prof(ds.olr, rbins)
ax.fill_between(dsr_olr.r, dsr_olr.olr_q.isel(q=0), dsr_olr.olr_q.isel(q=-1), color=c1, alpha=0.4)
dsr_olr.olr_mean.plot(ax=ax, c=c1)
ax.set_xlim(xmin=0)
ax2 = ax.twinx()

# ax2.spines["left"].set(color=c1, linewidth=2)
ax.yaxis.label.set_color(c1)
ax.tick_params(axis="y", colors=c1)

# ax2.spines["right"].set(color=c2, linewidth=2)
ax2.yaxis.label.set_color(c2)
ax2.tick_params(axis="y", colors=c2)


def plot(ilev=8):
    ax2.cla()
    dsr_qrain = radial_prof(ds.qrain.isel(hgt=ilev).squeeze(), rbins)
    ax2.fill_between(
        dsr_qrain.r, dsr_qrain.qrain_q.isel(q=0), dsr_qrain.qrain_q.isel(q=-1), color=c2, alpha=0.4
    )
    dsr_qrain.qrain_q.sel(q=0.5).plot(ax=ax2, c=c2, ls=":")  # median (dotted)
    dsr_qrain.qrain_mean.plot(ax=ax2, c=c2)  # mean (solid)
    ax.set_title(f"ilev={ilev}, hgt={ds.hgt.values[ilev]} m")


interact(plot, ilev=(0, 22))

# %% [markdown]
# 👆 We can see that using the mean for the rain water mixing ratio radial profile is misleading, since it is not symmetric. Usually the mean is pretty far above the median, suggesting that the distribution is skewed right, as we would expect for rainfall climatology. This is not rainfall climatology; rather, we have just one time step, but it is not raining everywhere in the domain.

# %% [markdown]
# ## Vertical alignment of precip and OLR

# %% [markdown]
# > Use vertically stacked subplots to compare location of precipitation (surface QRAIN and composite QRAIN) and OLR.

# %% [markdown]
# ### lat/lon

# %%
fig, axs = plt.subplots(3, 1, figsize=(5, 9), sharex=True)

for ax, vn in zip(axs.flat, ["olr", "qrain_cmp", "qrain_sfc"]):
    cmap = "binary" if vn == "olr" else "ocean"
    norm = colors.LogNorm(vmin=1e-11, vmax=1e-2) if vn[:5] == "qrain" else None
    ds[vn].plot(ax=ax, cmap=cmap, norm=norm)
    if ax is not axs[-1]:
        ax.set_xlabel("")

# %% [markdown]
# ### Radial profile

# %%
radial_prof(ds["olr"], rbins)

# %%
fig, ax1 = plt.subplots(figsize=(9, 5))

ax2 = ax1.twinx()

cs = ["red", "blue", "green"]

# mark left y-ax as OLR
ax1.yaxis.label.set_color(cs[0])
ax1.tick_params(axis="y", colors=cs[0])
ax2.spines["left"].set(color=cs[0])

for c, vn in zip(cs, ["olr", "qrain_cmp", "qrain_sfc"]):
    ax = ax1 if vn == "olr" else ax2
    da = radial_prof(ds[vn], rbins)
    ax.fill_between(da.r, da[f"{vn}_q"].isel(q=0), da[f"{vn}_q"].isel(q=-1), color=c, alpha=0.2)
    # da[f"{vn}_q"].sel(q=0.5).plot(ax=ax, color=c, label=vn, alpha=0.5)
    # ^ plotting the median too makes it a bit messy

# fix labels
ax1.set_title("")
ax2.set_title("")
ax1.set_ylabel(f"OLR [{ds.olr.units}]")
ax2.set_ylabel(f"Rain water mixing ratio [{ds.qrain.units}]")
ax1.set_xlim(xmin=0, xmax=da.r[-1])
ax2.legend(["Composite", "Surface"])
