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
import statsmodels.formula.api as smf
import xarray as xr
from utils import add121

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
# ## Surface temp. vs pot. temp.

# %% [markdown]
# We can compute surface air temperature from the potential temperature since we have the surface pressure. We assume that the reference pressure $p_0$ is 1000 hPa.
# $$
# \theta = T \left(\frac{p}{p_0}\right)^{-\kappa} \to T = \theta \left(\frac{p}{p_0}\right)^{+\kappa}
# $$
# where $\kappa = R/c_p \approx 0.286$ for dry air.
#
# ❗ Note that we would really want to have pressure of the first level! Our first level may not be fully consistent with the surface in the original simulation that the surface pressure corresponds to.

# %%
p_0 = 1000 * 100  # Pa
p = ds.psfc
theta0 = ds.theta.isel(hgt=0)

ds["ta_sfc"] = theta0 * (p / p_0) ** 0.286
ds.ta_sfc.attrs.update({"units": ds.theta.units, "long_name": "Surface air temperature"})

assert np.allclose(
    theta0.values, (ds.ta_sfc * (p / p_0) ** (-0.286)).values
), "Should be able to recover potential temperature at first level"
# ^ `xarray.testing.assert_allclose` didn't work for some reason, maybe the attrs

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 4.5))

cbar_kwargs = dict(orientation="horizontal")

ds.ta_sfc.plot(ax=ax1, cbar_kwargs=cbar_kwargs)
ax1.set(title=r"$T_{a, \mathrm{sfc}}$")

theta0.plot(ax=ax2, cbar_kwargs=cbar_kwargs)
ax2.set(ylabel="", title=r"$\theta_0$ (lowest height level)")

(ds.ta_sfc - theta0).plot(
    ax=ax3, cbar_kwargs=dict(orientation="horizontal", label="Temperature difference\n[°C] or [K]")
)
ax3.set(ylabel="", title=r"$T_{a, \mathrm{sfc}} - \theta_0$")

# %% [markdown]
# 👆 In the vicinity of the eye, the potential temperature is high because it is warm but the pressure is low due to the cyclone. We see that actual temperature is also higher in that region, but the difference wrt. the surrounding environment is less striking.

# %% [markdown]
# ## Maps

# %% [markdown]
# > Map surface pressure, surface temperature, surface water vapor mixing ratio, surface wind speed, surface sensible heat flux, surface latent heat flux, precipitation, and PBL height.

# %%
# TODO: ?

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

# TODO: compute actual surface temperature using pot. T and surface pressure

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.5, 3.8))

plot_linear_fit("theta", "pblh", ax=ax1)

plot_linear_fit("qvapor", "pblh", ax=ax2)

# %% [markdown]
# ## Surface precip. -- spatial relationships
#
# > Compare precipitation and spatial patterns in PBL height, surface temperature and surface mixing ratio to determine the origin of the weird patterns in the latter two variables.  Do this either with side by side subplots of precipitation and the other variables or by overlaying two variables on one plot using contour over pcolormesh.
#
# Note that `qrain_sfc` is *not* equivalent to `qrain` at the lowest level (though after the correction they are much more similar).
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

# %% [markdown]
# ## Linear model of surface sensible heat flux
#
# > In theory, the surface sensible heat flux can be computed using the formula: HFX = C*WindSpeed*(SST-Tsfc), where C is constant. Plot HFX vs WindSpeed*(SST-Tsfc) to see how well this theory holds in a typhoon.  Use linear regression to estimate the value of C.
#
# The constant C should encompass $\rho \, c_p$.

# %%
# TODO: could do this earlier
df = ds.isel(hgt=0).to_dataframe()

df["delta_t"] = df["sst"] - df["theta"]


def fit_hfx_formula_and_plot(formula, *, print_summary=True, limits="max"):
    # construct linear model and fit
    mod = smf.ols(formula=formula, data=df)
    res = mod.fit()

    if print_summary:
        print(res.summary())

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # plot modeled vs actual
    ax.scatter(df["hfx"], res.predict(df), marker="o", s=15, alpha=0.5, linewidths=0)
    ax.set(xlabel="actual hfx", ylabel="predicted hfx")

    # plot 1-1 line
    add121(ax, limits=limits)

    # show coeffs in the title
    p_str = "\n+ ".join(f"{c:.4g}$\,${name}" for name, c in res.params.items())
    # ax.set_title(f"hfx = {p_str}", fontsize=10)
    ax.text(1.05, 0.9, f"hfx = {p_str}", ha="left", va="top", transform=ax.transAxes, fontsize=10)

    # show R^2
    ax.text(0.02, 0.98, f"$R^2 = {res.rsquared:.3f}$", va="top", ha="left", transform=ax.transAxes)


# `-1` - don't use intercept (constant term) in the model (the directions suggest that we shouldn't include one)
# `:`  - only include the multiplied term, not the individuals as well (`*`)
# note that the fit is much worse (according to R^2) with intercept included
# scatters look identical tho, just different y values (and different coeffs)
fit_hfx_formula_and_plot("hfx ~ uh : delta_t -1")

# %%
fit_hfx_formula_and_plot("hfx ~ uh : delta_t", print_summary=False)  # intercept included

# %% [markdown]
# 👆 With intercept included, the computer $R^2$ value is much lower. The scatter looks identical (with auto plot settings, not square), except that the range of $y$ values is different.
#
# A much better model (it seems) includes $\delta T$ as well as the individual temperature terms
# ($\times \, U_h$) in a multiple linear regression.

# %%
fit_hfx_formula_and_plot("hfx ~ uh : (delta_t + sst + theta) -1")
# this time is only a bit better without intercept (0.042 increase)

# %% [markdown]
# 👆 We can see that there is a pretty strong linear relationship between the predicted and actual `hfx`. However, the values are underpredicted on average.
#
# Even better if the individual temperature terms are included as well, in addition to their
# multiplication with $U_h$.

# %%
fit_hfx_formula_and_plot("hfx ~ uh * (delta_t + sst + theta) -1")
# 0.061 increase in R^2 without intercept
# scatters look identical tho, just different y values

# %%
fit_hfx_formula_and_plot(
    "hfx ~ uh * (delta_t + sst + theta)", print_summary=False
)  # intercept included

# %% [markdown]
# 👆 We noticed that---according to `statsmodels`---the $R^2$ values are better (closer to 1) when an intercept is not included in the model. This seems to be because `statsmodels` uses the uncentered $R^2$ formula in this case. The scatter looks essentially identical in both cases; unlike the first, simplest model, there is not a discernible difference in the range of $y$ values, although we can see that the coefficients have changed.
