# meteo473_fa20
My Meteo 473 TA workspace for Fall 2020

## Getting started

Current steps for setting up after a fresh clone of the repo.

1. Create Conda env:  
   ```
   conda env create -f environment.yml
   ```
2. Place the original data files (`*.npy`) in [`data/orig`](./data/orig).
   Alternatively, place `data.zip` containing the `*.npy` files in `data/orig` and they will be loaded from there.
3. Run `write_nc()` from `data/create_nc.py` to create the netCDF file.
4. Install the pre-commit Git hooks with `pre-commit install`

JupyterLab extensions
```
jupyter labextension install @jupyterlab/toc
```
