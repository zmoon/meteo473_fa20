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
#     display_name: Python [conda env:meteo473_fa20]
#     language: python
#     name: conda-env-meteo473_fa20-py
# ---
# %%
import ipywidgets
import matplotlib.pyplot as plt
import xarray as xr

# #%matplotlib widget
# ^ not working yet
# #%matplotlib notebook

# %% [markdown]
# Load data.
# %%
ds = xr.open_dataset("../data/data.nc")
ds

# %% [markdown]
# Pretty-print the list of variables.

# %%
from IPython.display import Markdown

lines = ["name | long_name | units", ":--- |:--- | --- "]
for name, datavar in ds.data_vars.items():
    long_name = datavar.attrs["long_name"]
    units = datavar.attrs.get("units", "missing")
    lines.append(f"`{name}` | {long_name} | [{units}]")

Markdown("\n".join(lines))

# %% [markdown]
# ## PSFC

# %% [markdown]
# Plot sea-level (surface) pressure (`PSFC`), convering units Pa to hPa.

# %%
ds["psfc_hPa"] = ds.psfc / 100
ds.psfc_hPa.attrs.update(
    {"units": "hPa", "long_name": "Sea-level pressure"}
)  # note: original `long_name` doesn't survive
ds.psfc_hPa.plot(size=8, cmap="gnuplot_r")

# %% [markdown]
# Zoom in a bit and do contours.

# %%
cs = ds.psfc_hPa.sel(lon=slice(129.5, 132.5), lat=slice(8.5, 12.5)).plot.contour(
    size=7, levels=20, aspect=1.1
)
plt.gca().clabel(cs, inline=True, fontsize=10)

# %% [markdown]
# ## OLR

# %% [markdown]
# Plot OLR (outgoing longwave radiation).

# %%
ds.olr.plot(size=8, cmap="Spectral_r")


# %% [markdown]
# > Experiment around with contour, contour and pcolormesh and chose the one that looks best for each of the variables.
#
# I think that pcolormesh is good for OLR and contour for PSFC. The below widget setup allows for easy comparison.

# %%
def plot(name, plot_type):
    da = ds[name]
    shared_kw = dict(size=8)
    if plot_type == "contour":
        cs = da.plot.contour(levels=20, **shared_kw)
        plt.gca().clabel(cs, inline=True, fontsize=10)
    elif plot_type == "contourf":
        da.plot.contourf(levels=20, **shared_kw)
    elif plot_type == "pcolormesh":
        da.plot.pcolormesh(**shared_kw)
    else:
        raise ValueError(f"`plot_type` {plot_type!r} no good")


ipywidgets.interact(
    plot, name=["psfc_hPa", "olr"], plot_type=["contour", "contourf", "pcolormesh"],
)
