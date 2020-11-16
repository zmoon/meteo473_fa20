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
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import interact
from scipy.stats import binned_statistic_2d

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
# ## Vertical velocity maps
#
# Vertical velocity associated with convection

# %% [markdown]
# > Use pcolormesh to generate maps of vertical velocity. Create one map every 1000 geopotential meters from 1000 up to 29000.  Use these maps to inform and illustrate your answers to the following questions. [10 pts]
# >
# > * How high does the convection go (in kilometers).  [5 pts]
# > * Across what range of altitudes (in kilometers) can you pick out the gravity waves?  [5pts]
# > * What shape are the gravity wave patterns on these maps and does this shape evolve with height? If so, how? [5pts]

# %% [markdown]
# We can build an interactive selector like we have done before.

# %%
fig = plt.figure(figsize=(8, 6))


def plot_w_hgt(hgt=17000, symlog=False, symlog_linthresh=1.0, contourf=False, nlevs=60):
    da = ds.w.sel(hgt=hgt)
    fig.clf()
    ax = fig.add_subplot()
    norm = None if not symlog else mpl.colors.SymLogNorm(linthresh=symlog_linthresh, base=10)
    fn = da.plot if not contourf else partial(da.plot.contourf, levels=nlevs)
    # note nlevs doesn't seem to work properly with SymLogNorm (actual # of levels come out less)
    fn(ax=ax, norm=norm)


interact(plot_w_hgt, hgt=(1000, 29000, 500), symlog_linthresh=(0.1, 2.0, 0.05), nlevs=(2, 200, 1))

# %% [markdown]
# Vertical cross-sections could help us see this in one figure, albeit for a limited part of the domain. That's kind of what we are doing in the next part though.

# %%
is_hgt_range = (ds.hgt >= 1000) & (ds.hgt <= 29000)

# recall 144 is the latitude of the gridpt-wise min in surface pressure
# from `ds.psfc.argmin(dim=["lat", "lon"])`
ds.w.isel(lat=144, hgt=is_hgt_range).plot.contourf(levels=60, size=4, aspect=1.8)

# %% [markdown]
# ## Convective gravity waves

# %% [markdown]
# > Generate southwest-to-northeast vertical cross sections of vertical velocity, potential temperature, and perturbation potential temperature (i.e. potential temperature minus its height-dependent average) through one of the strong horse-shoe shaped convectively generated gravity waves northwest of the eye. [20 pts]
# >
# > If these waves tilt, they are vertically propagating.  If they don’t, they are horizontally propagating.  Which is the case?  [5 pts]
#
# There is one horseshoe with center around 128.4 degLon, 12.2 degLat in the UT/LS region.

# %%
# Add dtheta to ds
ds["dtheta"] = ds["theta"] - ds.theta.mean(dim=["lat", "lon"])
ds.dtheta.attrs.update(
    long_name=r"Potential temperature difference wrt. height level mean $\theta'$", units="K"
)

# %% [markdown]
# ### Gridpt-wise SW→NE VXS

# %%
# Note NumPy can do vectorized indexing easily (in xarray we will see it is less straightforward)
a = np.arange(49).reshape((7, 7))
print(a)
i_a = np.arange(4, -1, -1)
j_a = np.arange(0, 5)
print(a[i_a, j_a])  # arrays, lists, and tuples of indices all work
print(a[i_a.tolist(), j_a.tolist()])
print(a[tuple(i_a.tolist()), tuple(j_a.tolist())])

# %%
# First method - going by gridpoints
lat_hsc, lon_hsc = 12.2, 128.4
ds_hsc = ds.sel(lon=lon_hsc, lat=lat_hsc, method="nearest")
ilat_hsc, ilon_hsc = (
    np.where(ds.lat.values == ds_hsc.lat.values)[0][0],
    np.where(ds.lon.values == ds_hsc.lon.values)[0][0],
)
print(ds_hsc.lat.values, ds_hsc.lon.values)

# Extend SW to NE
n_ph = 20  # number of points to either side of center point
ilat_hs = ilat_hsc + np.arange(-n_ph, n_ph + 1)
ilon_hs = ilon_hsc + np.arange(-n_ph, n_ph + 1)
lat_hs1 = ds.lat.isel(lat=ilat_hs)
lon_hs1 = ds.lon.isel(lon=ilon_hs)
print(lat_hs1.values)
print(lon_hs1.values, end="\n\n")

# Here are many attempts to do the vectorized indexing that don't work
# print(ds.hfx[xr.DataArray(ilat_hs, dims="lat"), xr.DataArray(ilon_hs, dims="lon")])
# ds.hfx[xr.DataArray(ilat_hs, dims="lat"), xr.DataArray(ilat_hs, dims="lat")]
# ds.hfx[xr.DataArray(ilat_hs, dims="lon"), xr.DataArray(ilat_hs, dims="lon")]
# print(ds.hfx.isel(lon=xr.DataArray(ilon_hs, dims="lon"), lat=xr.DataArray(ilon_hs, dims="lon")))
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilat_hs, dims="lat")))

# This one does work!
# xarray used to have `.sel_points` and `isel_points` that made this more straightforward and less weird, but no more.
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilon_hs, dims="lat")))
# Or this. Just have to keep the `dims` the same it seems.
# print(ds.hfx.isel(lat=xr.DataArray(ilat_hs, dims="lon"), lon=xr.DataArray(ilon_hs, dims="lon")))

ds_hs1 = ds.isel(lat=xr.DataArray(ilat_hs, dims="lat"), lon=xr.DataArray(ilon_hs, dims="lat"))
assert np.all(ds_hs1.lat.values == lat_hs1.values) and np.all(ds_hs1.lon.values == lon_hs1.values)

# %%
is_hgt_range = (ds.hgt > 13000) & (ds.hgt < 28000)

to_plot = ["w", "theta", "dtheta"]

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)

for vn, ax in zip(to_plot, axs.flat):
    da = ds_hs1[vn].isel(hgt=is_hgt_range)
    if vn == "theta":
        da.plot.contour(levels=20, ax=ax, add_colorbar=True)
    else:
        da.plot.contourf(levels=30, ax=ax)

# Prepare to label cross section end points
coords_a = ds_hs1.lat.values[0], ds_hs1.lon.values[0]
s_coords_a = f"({coords_a[0]:.2f}°N, {coords_a[1]:.2f}°E)"
coords_b = ds_hs1.lat.values[-1], ds_hs1.lon.values[-1]
s_coords_b = f"({coords_b[0]:.2f}°N, {coords_b[1]:.2f}°E)"

# Adjustments
for i, ax in enumerate(axs.flat):
    if i >= 1:
        ax.set_title(s_coords_a, loc="left", fontsize=10)
        ax.set_title(s_coords_b, loc="right", fontsize=10)
    if i < len(axs) - 1:
        ax.set_xlabel("")
    else:
        ax.set_xlabel(rf"Latitude of SW$\to$NE gridpt-wise cross section [{ds.lat.units}]")

fig.set_tight_layout(dict(h_pad=0.3))

# %% [markdown]
# 👆 We can see some waviness in the standard $\theta$ plot (I feel it is easier to see with contour lines instead of fills). It becomes much more apparent when subtracting the mean of each level ($\theta'$). Looking at the temperature perturbation/anomaly structures ($\theta'$) and $w$, the waves do seem to be somewhat tilting with height.

# %% [markdown]
# ### Interactive interpolating VXS

# %%
# Second method - interpolation, choosing A and B points and interpolating on line connecting them

fig = plt.figure(figsize=(4.5, 3.2))
fig2 = plt.figure(figsize=(7, 3.5))


def vxs_interp(
    lat_a=11.65,
    lon_a=127.9,
    lat_b=13.1,
    lon_b=129.1,
    n_points=20,
    variable="w",
    contourf=True,
    interp_method="linear",
):
    fig.clf()
    ax = fig.add_subplot()
    fig2.clf()
    ax2 = fig2.add_subplot()

    # Plot reference
    # plot_w_hgt(hgt=23000, symlog=True, symlog_linthresh=0.5, contourf=True, nlevs=60)
    ds.w.sel(hgt=23000).plot(ax=ax)

    # Plot chosen end-points
    ax.plot([lon_a, lon_b], [lat_a, lat_b], "g.-", lw=2)
    ax.annotate("A", (lon_a, lat_a), xytext=(-10, -10), textcoords="offset points", color="g")
    ax.annotate("B", (lon_b, lat_b), xytext=(2, 2), textcoords="offset points", color="g")

    # Interpolate and plot vertical cross-section
    slope = (lat_b - lat_a) / (lon_b - lon_a)
    lon_line = np.linspace(lon_a, lon_b, n_points)
    lat_line = lat_a + slope * (lon_line - lon_a)
    assert np.isclose(lat_b, lat_line[-1])
    dims = "lat" if not lat_b == lat_a else "lon"  # note one dim must be doubled, like before
    da = (
        ds[variable]
        .interp(
            lat=xr.DataArray(lat_line, dims=dims),
            lon=xr.DataArray(lon_line, dims=dims),
            method=interp_method,
        )
        .isel(hgt=is_hgt_range)
    )
    if contourf:
        da.plot.contourf(levels=30, ax=ax2)
    else:
        da.plot(ax=ax2)
    if dims == "lat":
        ax2.set_xlabel(f"Latitude of cross section [{ds.lat.units}]")
    elif dims == "lon":
        ax2.set_xlabel(f"Longitude of cross section [{ds.lon.units}]")
    # TODO: 2nd x-ax to show A->B distance in km


lat_range = (ds.lat.values[0], ds.lat.values[-1], 0.05)
lon_range = (ds.lon.values[0], ds.lon.values[-1], 0.05)
locs = dict(lat_a=lat_range, lon_a=lon_range, lat_b=lat_range, lon_b=lon_range)
interact(
    vxs_interp,
    **locs,
    n_points=(3, 400),
    variable=["w", "dtheta", "theta", "u"],
    interp_method=["linear", "nearest"],
)

# %% [markdown]
# ## Momentum flux

# %% [markdown]
# > Select an area 100x100 grid points centered on one (for example the one at 70 EW and 205 NS) of these convective cells. Compute the momentum flux components (W\*U and W\*V) for all levels in this area.  Average over each level and plot the profiles of the U and V components of momentum flux as a function of height from 1 to 29 km of geopotential height.  Also plot the profiles of the direction and magnitude of the flux. [20 pts]
# >
# > Is the convective transport of momentum upward or downward (explain your reasoning)? Hint: think about the wind direction in your flux calculation box.  If U and V are of the same sign as Uflux and Vflux then the momentum transport is upward.  Note: you may find it helpful to plot U and V profiles as well before trying to work this out.  At what level is the convective momentum transport most extreme?  How closely does the top of the convective transport of momentum compare with the top of convection that you worked out above? [5 pts]

# %% [markdown]
# > Is the gravity wave transport of momentum upward or downward? Is it more extreme than the convective transport or less so?  [5 pts]

# %%
ihgt_mf = (ds.hgt >= 1000) & (ds.hgt <= 29000)

fig = plt.figure(figsize=(4, 3))
fig2 = plt.figure(figsize=(10.5, 4.8))


def mom(ilat_mfc=205, ilon_mfc=70, nbox_half=50):  # 202, 60 instead?
    fig.clf()
    fig2.clf()

    # Orient ourselves
    ax = fig.add_subplot()
    ds.w.sel(hgt=23000).plot(ax=ax)
    ax.plot(ds.lon[ilon_mfc], ds.lat[ilat_mfc], "g*", ms=10, mfc="none")

    # Box (101x101 but whatevs)
    ilat_mf = ilat_mfc + np.arange(-nbox_half, nbox_half + 1)
    ilon_mf = ilon_mfc + np.arange(-nbox_half, nbox_half + 1)

    # TODO: Plot box

    # Selection
    ds_mf = ds.isel(hgt=ihgt_mf, lat=ilat_mf, lon=ilon_mf)

    # Calculate the momentum flux variables
    ds_mf["wu"] = ds.w * ds.u
    ds_mf.wu.attrs.update(long_name="Vertical flux of x-momentum $w u$", units="m$^2$ s$^{-2}$")
    ds_mf["wv"] = ds.w * ds.v
    ds_mf.wv.attrs.update(long_name="Vertical flux of y-momentum $w v$", units="m$^2$ s$^{-2}$")
    ds_mf["mf_mag"] = np.sqrt(ds_mf.wu ** 2 + ds_mf.wv ** 2)
    ds_mf.mf_mag.attrs.update(
        long_name="Magnitude of vertical flux of horizontal momentum", units="m$^2$ s$^{-2}$"
    )
    ds_mf["mf_dir"] = np.arctan2(ds_mf.wv, ds_mf.wu)
    ds_mf.mf_dir.attrs.update(
        long_name="Direction of vertical flux of horizontal momentum", units="radians"
    )

    # Plot level-average profiles
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, , sharey=True)
    ax1 = fig2.add_subplot(141)
    ax2 = fig2.add_subplot(142, sharey=ax1)
    ax3 = fig2.add_subplot(143, sharey=ax1)
    ax4 = fig2.add_subplot(144, sharey=ax1)

    ax1.axvline(x=0, ls=":", c="0.7")
    ds_mf.wu.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", ax=ax1, label="$w u$")
    ds_mf.wv.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", ax=ax1, label="$w v$")
    ax1.set_xlabel(f"Momentum flux components [{ds_mf.wu.units}]")
    ax1.legend()
    ax1.autoscale(enable=True, axis="y", tight=True)

    ds_mf.mf_mag.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", c="forestgreen", ax=ax2)

    ds_mf.mf_dir.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", c="crimson", ax=ax3)
    ax3_xlim = ax3.get_xlim()
    for label, rad in {"E": 0, "N": np.pi / 2, "W": np.pi, "S": 3 * np.pi / 2}.items():
        if (rad > ax3_xlim[0]) & (rad < ax3_xlim[-1]):
            ax3.text(rad, 1000, label, va="bottom", ha="center")

    ax4.axvline(x=0, ls=":", c="gold")
    ds_mf.w.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", ax=ax4, c="gold")

    ax5 = ax4.twiny()
    ax5.axvline(x=0, ls=":", c="0.7")
    ds_mf.u.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", ax=ax5, label="$u$")
    ds_mf.v.mean(dim=("lat", "lon"), keep_attrs=True).plot(y="hgt", ax=ax5, label="$v$")
    np.sqrt(ds_mf.u ** 2 + ds_mf.v ** 2).mean(dim=("lat", "lon"), keep_attrs=True).plot(
        y="hgt", ax=ax5, label="$u_h$"
    )
    ax5.set_xlabel("Horizontal winds [m s$^{-1}$]")
    ax5.legend()

    for ax in fig2.axes:
        ax.label_outer()


interact(mom, ilat_mfc=(50, 249), ilon_mfc=(50, 249), nbox_half=(2, 50))

# %% [markdown]
# 👆 At the 70 EW and 205 NS point: Above the tropopause level, the magnitudes of the individual components are generally similar and smaller than in the troposphere. But below, the $x$ component ($w u$) is considerably stronger in the upper troposphere. In the low–mid troposphere, the component magnitudes are comparable, but $w v$ is a bit stronger. The signs of the components are consistent with upward transport of winds in the direction of the storms cyclonic rotation (here negative $u$ and $v$). Above the tropopause, in the lower LS layer there is non-zero flux corresponding to the gravity wave layer.

# %% [markdown]
# ## Shear

# %% [markdown]
# > Plot, for the entire domain, the shear vectors between 9000 geopotential meters and 20000 geopotential meters over vertical velocity at 20000 geopotential meters.  [10 pts]
# >
# > Use this figure to illustrate the discussion in your report about why gravity waves in the stratosphere over the typhoon’s convection are horseshoe shaped with orientation changing with azimuth from the storm center.  Also discuss why the stratospheric gravity waves are much more circular in the northeast and southeast parts of your domain.  [10 pts]

# %%
fig = plt.figure(figsize=(9.5, 7))


def plot_shear(*, nxy=50, scale=500, stream=False, quiver=True, h1=9000, h2=20000, swap_uv=False):
    fig.clf()
    ax = fig.add_subplot()

    # Horseshoes at 20km for reference
    im = ds.w.sel(hgt=20000).plot(ax=ax, norm=mpl.colors.SymLogNorm(linthresh=0.35, base=10))
    cb = im.colorbar

    # Compute shear as difference in winds
    hgts = [h1, h2]
    u1, u2 = ds.u.sel(hgt=hgts).values  # unpack along first dim
    v1, v2 = ds.v.sel(hgt=hgts).values
    du0 = u2 - u1
    dv0 = v2 - v1
    x0 = ds.lon.values
    y0 = ds.lat.values
    X0, Y0 = np.meshgrid(x0, y0)

    # Take shear to lower res (down-sample/up-scale)
    res = binned_statistic_2d(X0.flatten(), Y0.flatten(), [du0.flatten(), dv0.flatten()], bins=nxy)
    # res.statistic results are (nx, ny) but we need (ny, nx) to match what quiver expects (meshgrid-style)
    du = res.statistic[0].T
    dv = res.statistic[1].T
    if swap_uv:
        dv, du = du, dv
    xe = res.x_edge
    ye = res.y_edge
    x = (xe[:-1] + xe[1:]) / 2
    y = (ye[:-1] + ye[1:]) / 2

    # Plot streamlines (x and y have to be equally spaced!)
    if stream:
        ax.streamplot(x, y, du, dv, density=2.0, color="0.5")

    # Plot shear vectors
    if quiver:
        # ax.quiver(x0, y0, du0, dv0)  # too many arrows if plot all points!
        q = ax.quiver(x, y, du, dv, scale=scale, scale_units="width", color="0.2", alpha=0.9)
        # q = ax.quiver(x0, y0, u1, v1, scale=scale, scale_units="width", color="0.2", alpha=0.9)  # test wind at one lev (lev1)
        ax.set_title(f"Full plot width : {scale} m/s ", loc="left", fontsize=10)
        ax.quiverkey(q, X=0.19, Y=1.045, U=20, label="20 m/s reference:", labelpos="W")

    # Labels
    ax.set_title(f"Shear between:\n{h1/1000:g} and {h2/1000:g} km [m/s]", loc="right", fontsize=10)
    ax.set_title("")
    cb.set_label(f"{cb._label} at 20 km")


h_range = (0, int(ds.hgt.values[-1]), 500)
interact(plot_shear, nxy=(10, 200), scale=(100, 1000, 10), h1=h_range, h2=h_range)

# %%
# confirm that v and u are not switched in the dataset
ds.v.sel(hgt=1000).plot(size=5)
