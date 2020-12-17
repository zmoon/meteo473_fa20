My Meteo 473 (TA) solutions for Fall 2020

## Browse on Binder

| nb | &nbsp; |
| -- | ----- |
| Bootcamp | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/meteo473_fa20/HEAD?urlpath=lab%2Ftree%2Fnbs%2Fbootcamp.ipynb) |
| Milestone 1 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/meteo473_fa20/HEAD?urlpath=lab%2Ftree%2Fnbs%2Fms1.ipynb) |
| Milestone 2 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/meteo473_fa20/HEAD?urlpath=lab%2Ftree%2Fnbs%2Fms2.ipynb) |
| Milestone 3 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/meteo473_fa20/HEAD?urlpath=lab%2Ftree%2Fnbs%2Fms3.ipynb) |
| Milestone 4 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/meteo473_fa20/HEAD?urlpath=lab%2Ftree%2Fnbs%2Fms4.ipynb) |

Note: it can take several minutes to start if a built image is not found.


## Work with the repo locally

Getting set up:

1. Create Conda env:  
   ```
   conda env create -f environment.yml
   ```

2. Run `write_nc()` from `data/create_nc.py` to create the netCDF file
   from `data.zip`, which is included in the repo via Git LFS.

   For example:
   ```
   # at repo root
   python -c 'import data; data.write_nc()'
   ```

3. Install the pre-commit Git hooks with `pre-commit install --install-hooks` (optional)

### JupyterLab

* :eyes: To open a Jupytext py:percent `.py`, right-click → "Open With" → "Notebook".

* Install extensions:
  ```
  jupyter labextension install @jupyterlab/toc @jupyter-widgets/jupyterlab-manager
  jupyter lab build
  ```
  :point_up: `ipympl` won't work in JupyterLab [unless you install the widget manager](https://github.com/matplotlib/ipympl#install-the-jupyterlab-extension) (`jupyerlab-manager`):exclamation:.
