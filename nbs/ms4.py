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
from utils import add121

# import statsmodels.api as sm
# import statsmodels.formula.api as smf

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
# ## Vertical velocity associated with convection

# %% [markdown]
# > Use pcolormesh to generate maps of vertical velocity. Create one map every 1000 geopotential meters from 1000 up to 29000.  Use these maps to inform and illustrate your answers to the following questions. [10 pts]
# >
# > * How high does the convection go (in kilometers).  [5 pts]
# > * Across what range of altitudes (in kilometers) can you pick out the gravity waves?  [5pts]
# > * What shape are the gravity wave patterns on these maps and does this shape evolve with height? If so, how? [5pts]

# %% [markdown]
# ## Convective gravity waves

# %% [markdown]
# > Generate southwest-to-northeast vertical cross sections of vertical velocity, potential temperature, and perturbation potential temperature (i.e. potential temperature minus its height-dependent average) through one of the strong horse-shoe shaped convectively generated gravity waves northwest of the eye. [20 pts]
# >
# > If these waves tilt, they are vertically propagating.  If they don’t, they are horizontally propagating.  Which is the case?  [5 pts]

# %% [markdown]
# ## Momentum flux

# %% [markdown]
# > Select an area 100x100 grid points centered on one (for example the one at 70 EW and 205 NS) of these convective cells. Compute the momentum flux components (W*U and W*V) for all levels in this area.  Average over each level and plot the profiles of the U and V components of momentum flux as a function of height from 1 to 29 km of geopotential height.  Also plot the profiles of the direction and magnitude of the flux. [20 pts]
# >
# > Is the convective transport of momentum upward or downward (explain your reasoning)? Hint: think about the wind direction in your flux calculation box.  If U and V are of the same sign as Uflux and Vflux then the momentum transport is upward.  Note: you may find it helpful to plot U and V profiles as well before trying to work this out.  At what level is the convective momentum transport most extreme?  How closely does the top of the convective transport of momentum compare with the top of convection that you worked out above? [5 pts]

# %% [markdown]
# > Is the gravity wave transport of momentum upward or downward? Is it more extreme than the convective transport or less so?  [5 pts]

# %% [markdown]
# ## Shear

# %% [markdown]
# > Plot, for the entire domain, the shear vectors between 9000 geopotential meters and 20000 geopotential meters over vertical velocity at 20000 geopotential meters.  [10 pts]
# >
# > Use this figure to illustrate the discussion in your report about why gravity waves in the stratosphere over the typhoon’s convection are horseshoe shaped with orientation changing with azimuth from the storm center.  Also discuss why the stratospheric gravity waves are much more circular in the northeast and southeast parts of your domain.  [10 pts]
