# meteo473_fa20
My Meteo 473 (TA) solutions for Fall 2020

## Browse on Binder

| nb | &nbsp; |
| -- | ----- |
| Bootcamp | https://mybinder.org/v2/gh/zmoon/meteo473_fa20/binder?urlpath=lab/tree/nbs/bootcamp.ipynb |
| Milestone 1 | |
| Milestone 2 | |
| Milestone 3 | |
| Milestone 4 | |
| Milestone 5 | |


## Working with the repo locally

Current steps for setting up after a fresh clone of the repo.

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

3. Install the pre-commit Git hooks with `pre-commit install --install-hooks`

### JupyterLab extensions

```
jupyter labextension install @jupyterlab/toc @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

:point_up: `ipympl` won't work in JupyterLab [unless you install the widget manager](https://github.com/matplotlib/ipympl#install-the-jupyterlab-extension) (`jupyerlab-manager`).
