"""
Create a NetCDF file from the original .npy files using xarray
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from rounding import round_array  # noreorder (doesn't seem to be recognized as 1st-party)


DATA_ORIG_PATH = "./orig"


# TODO: check for missing units, etc. in meta data file


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
    # TODO: could create `keep` tuples and use those for indexing
    if data.ndim != 2:
        raise NotImplementedError(f"data.ndim={data.ndim!r}")
    if dim == 0:
        # assert np.all(data[:, 0] == data[0, 0])
        assert np.all(data - data[0, :][np.newaxis, :] == 0)
        return data[0, :].copy()
    elif dim == 1:
        # assert np.all(data[0, :] == data[0, 0])
        assert np.all(data - data[:, 0][:, np.newaxis] == 0)
        return data[:, 0].copy()
    else:
        raise ValueError(f"dim={dim!r}")


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
    dv_z = {
        "height": (
            "height",
            z,
            {"units": "m", "long_name": "Geopotential height above the surface plane"},
        )
    }
    # MS1 notes suggest this should be geopotential height

    dvs_all.update(dv_z)

    coord_names = ["height", "lat", "lon"]
    coords = {name: dvs_all[name] for name in coord_names}

    data_vars = {name: dv for name, dv in dvs_all.items() if name not in coord_names}

    ds = xr.Dataset(coords=coords, data_vars=data_vars)

    return ds


def write_test_ncs():
    """Create the NetCDF files for the comparison tests."""
    ds = create_ds()

    # try different output engine options
    ds.to_netcdf("test_default.nc")
    ds.to_netcdf("test_h5netcdf.nc", engine="h5netcdf")

    # comp without modifying
    comp = {"zlib": True, "complevel": 9}  # 9 is the max. 6 is zlib's default
    encoding = {vn: comp for vn in ds.data_vars}
    ds.to_netcdf("test_h5netcdf_comp9.nc", engine="h5netcdf", encoding=encoding)

    # comp after rounding
    decimal_places_try = [3, 4, 6]
    for nsd in decimal_places_try:
        ds2 = ds.copy(deep=True)
        for vn in ds2.data_vars:
            ds2[vn][:] = round_array(ds2[vn].values, nsd=nsd, in_binary=True)

        # ds2.to_netcdf(f"test_h5netcdf_comp9_nsd{nsd}.nc", engine="h5netcdf", encoding=encoding)
        ds2.to_netcdf(f"test_h5netcdf_comp9_nsd={nsd}_bin.nc", engine="h5netcdf", encoding=encoding)


def write_nc():
    """Write the nc that we will/may actually use."""
    ds = create_ds()
    comp = {"zlib": True, "complevel": 9}
    encoding = {vn: comp for vn in ds.data_vars}
    ds.to_netcdf("data.nc", engine="h5netcdf", encoding=encoding)


def convert_bytes(num):
    # ref: https://stackoverflow.com/a/39988702
    # but modified to use MB, not MiB, etc.
    for metric in ["B", "kB", "MB", "GB", "TB"]:  # etc.
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

    # write_test_ncs()
    # compare_test_ncs()

    # write_nc()

    pass
