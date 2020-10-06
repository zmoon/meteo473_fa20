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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
# ## Set $x,y$ coords

# %%
# use 3.4 km and reset the coords of the ds
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

# %%
ds2

# %% [markdown]
# ## Plots

# %% [markdown]
# > Make maps of surface QRAIN, composite QRAIN

# %% [markdown]
# ### Surface QRAIN
#
# Note that qrain at the first level is not equiv to the surface qrain (turns out to be quite different...). We demonstrate that in the below figure.

# %%
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(5, 8.7), sharex=True, sharey=True)

ds2.qrain.isel(hgt=0).plot(x="x", y="y", ax=ax1)
ds2.qrain_sfc.plot(x="x", y="y", ax=ax2)
(ds2.qrain.isel(hgt=0) - ds2.qrain_sfc).plot(x="x", y="y", ax=ax3)

for ax in fig.get_axes():
    ax.set_title("")
    if ax != ax3:
        ax.set_xlabel("")

# %% [markdown]
# More would be revealed in the above plots if we used log color scales.

# %%
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(5, 8.7), sharex=True, sharey=True)

ds2.qrain.isel(hgt=0).plot(x="x", y="y", ax=ax1, norm=mpl.colors.LogNorm(vmin=1e-10, vmax=1e-2))
ds2.qrain_sfc.plot(x="x", y="y", ax=ax2, norm=mpl.colors.LogNorm(vmin=1e-10, vmax=1e-2))
(ds2.qrain.isel(hgt=0) - ds2.qrain_sfc).plot(
    x="x", y="y", ax=ax3, norm=mpl.colors.SymLogNorm(linthresh=1e-5, vmin=-1e-3, vmax=1e-3, base=10)
)

for ax in fig.get_axes():
    ax.set_title("")
    if ax != ax3:
        ax.set_xlabel("")

# %% [markdown]
# ### Composite (maximum in column) QRAIN

# %%
# can't see much here
ds2.qrain_cmp.plot(x="x", y="y", size=3.5, aspect=1.5)

# better with log color scale
plt.figure()
im = ds.qrain_cmp.plot(
    norm=mpl.colors.LogNorm(vmin=1e-8, vmax=0.015),
    vmin=None,
    vmax=None,  # mpl 3.3 gets mad because xarray is still passing these
    cmap="gnuplot",
)

# %% [markdown]
# ### Wind speeds at different levels
#
# > Compute wind speed from U and V.

# %%
ds2["uh"] = np.sqrt(ds.u ** 2 + ds.v ** 2)
ds2.uh.attrs.update({"long_name": "Horizontal wind speed", "units": "m s$^{-1}$"})

# %% [markdown]
# > Use pcolormesh to create maps of wind speed, U and V at 500, 6000 and 15000 geopotential meters
#
# xarray `FacetGrid` can easily create a figure with plots at different levels. But not for different variables.

# %%
levs = [500, 6000, 15000]

# colorbar overlays the plots if autolayout is used
# still can't see the colorbar label
# could set `add_colorbar` to False and add it manually in its own ax
# or pass `cbar_ax`...
# many options here: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.plot.pcolormesh.html
with plt.rc_context({"figure.autolayout": False}):
    ds2.uh.sel(hgt=levs).plot.pcolormesh(
        row="hgt", x="x", y="y", size=2.9, aspect=1.3, cbar_kwargs=dict(shrink=0.5, fraction=0.2)
    )
    # note passing axes not supported with facted plots

# %%
# This is a hacky way to do this, probably there are easier ways, maybe with `mpl_toolkits.axes_grid1`
width_ratios = []
for _ in range(3):
    width_ratios.extend([1, 0.05, 0.11, 0.38])  # plot ax, spacing, cbar_ax, spacing (like ProPlot)

fig, axs = plt.subplots(
    3,
    3 * 4,
    figsize=(11, 7.5),
    gridspec_kw=dict(width_ratios=width_ratios, wspace=0.0, hspace=0.08),
)

gs = axs[0, 0].get_gridspec()

for i, vn in enumerate(["uh", "u", "v"]):

    ds_vn_i = ds2[vn].sel(hgt=levs)
    vmm = max(abs(ds_vn_i.min()), abs(ds_vn_i.max()))  # match colorbars using this

    for j, lev in enumerate(levs):
        ax = axs[j, i * 4]

        if j == 1:  # create cbar ax
            for ax_ in axs[:, i * 4 + 2]:
                ax_.remove()
            cbar_ax = fig.add_subplot(gs[:, i * 4 + 2])

        ds2[vn].sel(hgt=lev).plot.pcolormesh(
            x="x",
            y="y",
            vmax=vmm,
            add_colorbar=True if j == 1 else False,
            cbar_kwargs=dict(fraction=0.7) if j == 1 else None,
            cbar_ax=cbar_ax if j == 1 else None,
            ax=ax,
        )
        # adjustments
        if i != 0:  # not in the first column
            ax.set_ylabel("")
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        if j != len(levs) - 1:  # not in the last row
            ax.set_xlabel("")
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        for ax_ in np.concatenate((axs[:, i * 4 + 1], axs[:, i * 4 + 3])):
            ax_.axis("off")
        if j == 1:
            cb = ax.collections[-1].colorbar
            cb.set_label(
                cb._label.replace("\n", " ")
            )  # no need for line break since we have big space

        # move title to text inside ax to save space
        t = ax.get_title()
        ax.set_title("")
        ax.text(0.03, 0.97, t, transform=ax.transAxes, va="top", ha="left", color="green")

# %% [markdown]
# ### Streamlines overlaid on things
#
# > Overlay streamlines at 500 geopotential meters on a map of surface rain rate (actually rain water mixing ratio). Hints: use streamplot with U and V and pcolormesh with QRAIN

# %%
fig, ax = plt.subplots()

# qrain
ds2.qrain_cmp.plot(ax=ax, x="x", y="y")

# wind streamlines
u = ds2.u.sel(hgt=500).values
v = ds2.v.sel(hgt=500).values
x = ds2.x.values
y = ds2.y.values

# note: ok to pass meshgrid of x and y instead, they just need to be equally spaced
ax.streamplot(x, y, u, v, color="0.6", density=1.2)

# %% [markdown]
# > Overlay streamlines on potential temperature, both at 6000 geopotential meters

# %%
fig, ax = plt.subplots()

# potential temp.
ds2.theta.sel(hgt=6000).plot(ax=ax, x="x", y="y")

# wind streamlines
u = ds2.u.sel(hgt=6000).values
v = ds2.v.sel(hgt=6000).values
x = ds2.x.values
y = ds2.y.values

ax.streamplot(x, y, u, v, color="0.6", density=1.2)

# %% [markdown]
# > Overlay streamlines on pcolormesh plots of OLR, both at 15000 geopotential meters)

# %%
fig, ax = plt.subplots()

# OLR (doesn't vary with height)
ds2.olr.plot(ax=ax, x="x", y="y")

# wind streamlines
u = ds2.u.sel(hgt=15000).values
v = ds2.v.sel(hgt=15000).values
x = ds2.x.values
y = ds2.y.values

ax.streamplot(x, y, u, v, color="0.6", density=1.5)

# %% [markdown]
# What if we wanted to do this plot with lat/lon? We have to interpolate $u$ and $v$ to a lat/lon grid with constant spacing (dlon is already constant, but dlat is not) because `streamplot` only accepts evenly spaced grids.

# %%
from scipy.interpolate import griddata

u = ds2.u.sel(hgt=15000).values
v = ds2.v.sel(hgt=15000).values

lat = ds.lat.values
lon = ds.lon.values
Lon, Lat = np.meshgrid(lon, lat)

lat_new = np.linspace(lat[0], lat[-1], 200)
lon_new = np.linspace(lon[0], lon[-1], 200)

Lon_new, Lat_new = np.meshgrid(lon_new, lat_new)

# this is kind of awkward since have to pass coordinate pairs
# probably is a better way somewhere...
coords = (Lon.flatten(), Lat.flatten())

# does seem to work though, but takes a while
u_new = griddata(coords, u.flatten(), (Lon_new, Lat_new))
v_new = griddata(coords, v.flatten(), (Lon_new, Lat_new))

fig, ax = plt.subplots()

ds2.olr.plot(ax=ax)

ax.streamplot(lon_new, lat_new, u_new, v_new, color="0.6", density=1.5)

# %% [markdown]
# Looks pretty much the same to my eye.

# %% [markdown]
# ### Warm core
#
# This isn't one of the MS tasks, but one of the discussion questions asks about it.

# %%
iloc_pmin = ds.psfc.argmin(dim=("lat", "lon"))  # returns indices, not lat/lon values

levs = ds.hgt <= 18000

t = ds.isel(hgt=levs, lat=iloc_pmin["lat"]).theta

tanom = t - t.isel(lon=0)  # kind of arbitrary. really should average over far-field area

tanom.sel(lon=slice(130.6, 131.7)).plot.contourf(
    levels=50,
    size=5,
    cbar_kwargs={"label": rf"$\theta$ anom. (wrt. far-field env.) [°C]"},
)
# TODO: estimate tropopause height and plot its line

# %% [markdown]
# ## Questions
#
# Discussion questions to answer:
#
# > The latitude and longitude of the storm center (accurate to within 0.05 degrees) and whether the storm is warm or cold core at each of these levels.
#
# > Over what range of geopotential heights is the storm warm core?
#
# > Are the typhoon’s rainbands and eye wall easier to make out in surface QRAIN or composite QRAIN? Opinions may differ on this, so explain your logic.
#
# > What is the relative alignment of the Typhoon’s spiral rainbands and its low-level winds?
#
# > Where is the warm air relative to the mid-tropospheric circulation?
#
# > What is the relative alignment of the Typhoon’s cirrus cloud spirals and the outflow level winds?
#
# > What pressure level(s) does the storm inflow occur at? Outflow?  Which levels are mostly non-divergent?
#
