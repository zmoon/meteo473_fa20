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
import statsmodels.api as sm
import xarray as xr

# from ipywidgets import interact

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
# ## Scatter plots
#
# Examining relationships using scatter plots and linear fits.

# %% [markdown]
# ### vs. wind speed
#
# (at the lowest level)
#
# > Do scatter plots of sensible and latent heat flux vs wind speed.
#
# This is supposed to be at the surface, based on the first bullet.
#
# `hfx` and `lh` are surface variables, but the wind speed is not.

# %%
# first compute wind speed
ds["uh"] = np.sqrt(ds.u ** 2 + ds.v ** 2)
ds.uh.attrs.update({"long_name": "Horizontal wind speed", "units": ds.u.units})

ds_sfc = ds.isel(hgt=0)


def format_num(num):
    s0 = f"{num:.3g}"
    if "e" in s0:
        b, e = s0.split("e")
        e = int(e)
        return rf"{b} \times 10^{{{e}}}"
    else:
        return s0


def linear_fit(da_y, da_x):
    """Use statsmodels to fit linear model."""
    y = da_y.values.flatten()
    x = da_x.values.flatten()
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    p0, p1 = res.params
    sp0, sp1 = format_num(p0), format_num(p1)
    return {
        "line": (x, res.predict(X)),
        "text": f"${sp1} \, x + {sp0}$\n$r^2 = {res.rsquared:.3f}$",
    }


def plot_linear_fit(y_vn, x_vn, ax=None):
    """Plot the scatter and OLS linear fit for variable y vs variable x."""
    if ax is None:
        _, ax = plt.subplots()

    xr.plot.scatter(ds_sfc, x_vn, y_vn, ax=ax, marker=".", alpha=0.5, edgecolors="none")
    # ^ both this and the linear fit flatten the arrays, probably wasteful to do twice

    fit_res = linear_fit(ds_sfc[y_vn], ds_sfc[x_vn])
    ax.plot(*fit_res["line"], label=fit_res["text"], c="orangered", lw=2)
    ax.legend()


fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.5, 3.8))

plot_linear_fit("hfx", "uh", ax=ax1)

plot_linear_fit("lh", "uh", ax=ax2)

# %% [markdown]
# ### vs. PBL height
#
# > Do scatter plots of surface temperature and humidity vs PBL height.
#
# Note that surface temperature and near-surface potential temperature should be close to equivalent.

# %%
# TODO: compute relative humidity??

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.5, 3.8))

plot_linear_fit("theta", "pblh", ax=ax1)

plot_linear_fit("qvapor", "pblh", ax=ax2)

# %% [markdown]
# ## Surface precip. -- spatial relationships
#
# > Compare precipitation and spatial patterns in PBL height, surface temperature and surface mixing ratio to determine the origin of the weird patterns in the latter two variables.  Do this either with side by side subplots of precipitation and the other variables or by overlaying two variables on one plot using contour over pcolormesh.
#
# Note that `qrain_sfc` is *not* equivalent to `qrain` at the lowest level. About 1/2 of the points are the same though.
#
# Recall that what we are plotting for surface precip. is really rain water mixing ratio at the surface.

# %%
# plot precip so we know what the contours correspond to
plt.figure()
im = ds.qrain_sfc.plot(
    norm=mpl.colors.LogNorm(vmin=1e-8, vmax=1e-2),
    vmin=None,
    vmax=None,  # mpl 3.3 gets mad because xarray is still passing these
    cmap="gnuplot",
)

# %%
fig, axs = plt.subplots(3, 1, figsize=(4.7, 8.5), sharex=True, sharey=True)

for vn, ax in zip(["pblh", "theta", "qvapor"], axs.flat):
    ds_sfc[vn].plot(ax=ax)
    # plot a few precip contours
    ds_sfc.qrain_sfc.plot.contour(ax=ax, levels=[1e-4], colors=["0.8"], linewidths=0.5)
    # remove labels we don't need that xarray added
    ax.set_title("")
    if ax != axs.flat[-1]:
        ax.set_xlabel("")
