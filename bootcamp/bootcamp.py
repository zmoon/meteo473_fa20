# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python [conda env:.conda-meteo473_fa20]
#     language: python
#     name: conda-env-.conda-meteo473_fa20-py
# ---
# %% [markdown]
# Load data
# %%
import matplotlib.pyplot as plt
import xarray as xr

# #%matplotlib widget  # not working yet

# %%
ds = xr.open_dataset("../data/data.nc")
ds

# %% [markdown]
# Print variables list

# %%
# TODO: pretty colors, maybe with IPython.Display Markdown or HTML
for name, datavar in ds.data_vars.items():
    long_name = datavar.attrs["long_name"]
    units = datavar.attrs.get("units", "missing")
    print(f"{name} | {long_name} [{units}]")

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
ds.olr.plot(size=8)

# %% [markdown]
# > Experiment around with contour, contour and pcolormesh and chose the one that looks best for each of the variables.
