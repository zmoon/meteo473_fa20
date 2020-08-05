"""
Create a NetCDF file from the original .npy files using xarray
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


DATA_ORIG_PATH = "./orig"


def check_dupes(l):
    """Check an iterable for duplicates and raise error if there are any."""
    from collections import Counter

    c = Counter(l)
    for key, count in c.items():
        if count > 1:
            raise Exception(f"{key} counted {count} times (should be 1)")


def load_metadata():
    with open("data.yml", "r") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    # check names for duplicates
    check_dupes(f["name"] for f in metadata["files"].values())

    return metadata


def _remove_extraneous_dim(data, dim):
    """
    dim : int
        the one to remove
    """
    # for the lat/lon arrays, which are 2-d but needn't (unless they do vary in both x and y)
    # hack for now
    if dim == 0:
        assert np.all(data[:, 0] == data[0, 0])  # check could be better...
        return data[0, :].copy()
    elif dim == 1:
        assert np.all(data[0, :] == data[0, 0])
        return data[:, 0].copy()
    else:
        raise NotImplementedError(f"dim={dim!r}")


def _data_var(fpath, metadata):
    """Use the data from the YAML file to make a xr.Dataset data_vars item"""
    name = metadata["name"]
    dims = tuple(metadata["dims"])
    data = np.load(fpath)

    # corrections for coords
    if name == "lat":
        data = _remove_extraneous_dim(data, 1)
    elif name == "lon":
        data = _remove_extraneous_dim(data, 0)

    attrs = {k: v for k, v in metadata.items() if k in ("units", "long_name")}
    return {name: (dims, data, attrs)}


def _data_vars(files_data):
    """Create xr.Dataset dat_vars items for all of the data we have."""
    dvs = {}
    for fname, meta_dict in files_data.items():
        fpath = f"{DATA_ORIG_PATH}/{fname}"
        dv = _data_var(fpath, meta_dict)
        dvs.update(dv)
    return dvs


def create_ds():
    """Create the xr.Dataset using the .npy data and YAML metadata description."""

    md = load_metadata()

    dvs_all = _data_vars(md["files"])

    # construct height manually
    shape3d = np.load(f"{DATA_ORIG_PATH}/tInterp.npy").shape
    n_z, n_y, n_x = shape3d
    dz = 500.0  # m
    z = np.arange(0, n_z) * dz
    dv_z = {"height": ("height", z, {"units": "m", "long_name": "Height above the surface plane"},)}

    dvs_all.update(dv_z)

    coord_names = ["height", "lat", "lon"]
    coords = {name: dvs_all[name] for name in coord_names}

    data_vars = {name: dv for name, dv in dvs_all.items() if name not in coord_names}

    ds = xr.Dataset(coords=coords, data_vars=data_vars)

    return ds


def nsb(nsd):
    """Number of significant bits (nsb) from number of significant digits (nsd).

    nsd is like sig figs, not dsd (decimal significant digits)
    so the numbers we use this on need to be between -1.0 and 1.0 (divided by their 10 base)

    ref: https://github.com/fraserwg/elePyant/issues/1#issue-656925070
    """
    ...


def _rebase(x):
    """Convert to decimal and 10 base"""
    if isinstance(x, xr.DataArray):
        x = x.values
    base10 = np.ceil(np.log10(np.abs(x)))
    tens = 10 ** base10
    # correction for log10(0)
    # would be nice to be able to avoid the `RuntimeWarning: divide by zero encountered in log10` somehow though
    tens[x == 0] = 1.0
    return x / tens, tens


def round_arr(x, decimal_places=6):
    x1, tens = _rebase(x)
    return np.around(x1, decimals=decimal_places) * tens


def write_datasets():

    ds = create_ds()

    # try different output engine options
    ds.to_netcdf("test_default.nc")
    ds.to_netcdf("test_h5netcdf.nc", engine="h5netcdf")

    # comp without modifying
    comp = {"zlib": True, "complevel": 9}
    encoding = {vn: comp for vn in ds.data_vars}
    ds.to_netcdf("test_h5netcdf_comp9.nc", engine="h5netcdf", encoding=encoding)

    # comp after rounding in base 10
    # like elepyant: https://github.com/fraserwg/elePyant/blob/master/elePyant/core.py
    # but preserving sig figs instead of decimal places
    decimal_places_try = [3, 4, 6]
    for nsd in decimal_places_try:
        ds2 = ds.copy(deep=True)
        for vn in ds2.data_vars:
            ds2[vn][:] = round_arr(ds2[vn], decimal_places=nsd)

        ds2.to_netcdf(f"test_h5netcdf_comp9_nsd{nsd}.nc", engine="h5netcdf", encoding=encoding)


def convert_bytes(num):
    # ref: https://stackoverflow.com/a/39988702
    # but modified to use MB, not MiB, etc.
    for metric in ["B", "kB", "GB", "TB"]:  # etc.
        if num < 1000:
            return f"{num:3.2f} {metric}"
        num /= 1000


def file_size(path):
    """
    path : pathlib.Path
    """
    bytes_ = path.stat().st_size
    return convert_bytes(bytes_)


def compare_test_ncs():
    """Compare the different NetCDF files.

    zlib compression is lossless, but how much is lost by the rounding scheme?
    """

    ctl_path = Path("test_default.nc")
    test_nc_paths = [p for p in Path("./").glob("test_*.nc") if p != ctl_path]

    ctl = xr.open_dataset(ctl_path)

    data_ctl = {
        "name": f"ctl ({ctl_path.name})",
        "size": file_size(ctl_path),
        "sst_mean": float(ctl.sst.mean()),  # use `float()` to get a single number out
        "sst_mean_anom": 0,
        "sst_std": float(ctl.sst.std()),
        "t_mean": float(ctl.theta.mean()),
        "t_mean_anom": 0,
        "t_std": float(ctl.theta.std()),
    }
    # TODO: don't need to have this ^ twice

    data_others = []
    for test_nc_path in test_nc_paths:
        ds = xr.open_dataset(test_nc_path)

        data_others.append(
            {
                "name": f"{test_nc_path.name}",
                "size": file_size(test_nc_path),
                "sst_mean": float(ds.sst.mean()),
                "sst_mean_anom": float(ds.sst.mean() - ctl.sst.mean()),
                "sst_std": float(ds.sst.std()),
                "t_mean": float(ds.theta.mean()),
                "t_mean_anom": float(ds.theta.mean() - ctl.theta.mean()),
                "t_std": float(ds.theta.std()),
            }
        )

    df = pd.DataFrame(data=[data_ctl] + data_others)

    # print(df.to_string(float_format="%.5g"))

    # print markdown table
    # give backticks to file name column
    df2 = df.copy()
    df2.name = df.name.apply(lambda x: f"`{x}`")
    print(df2.to_markdown())

    # TODO: check mean/std for all variables, create separate dfs for each nsd nc file


if __name__ == "__main__":

    # write_datasets()

    compare_test_ncs()
