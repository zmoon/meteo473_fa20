"""
Create a NetCDF file from the original .npy files using xarray
"""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .rounding import round_array  # noreorder (doesn't seem to be recognized as 1st-party)


DATA_BASE_PATH = Path(__file__).parent
DATA_ORIG_PATH = DATA_BASE_PATH / "orig"


# TODO: check for missing units, etc. in meta data file


def cf_units_to_tex(s: str):
    """Convert CF-style units string to TeX-like.
    (In order to get exponents in plot labels, etc.)
    """
    # note pyparsing is a dependency of matplotlib
    # import pyparsing as pp
    #
    # unit_exp = pp.pyparsing_common.signed_integer
    import re

    def expify(match):
        m = match.group(0)
        # if m == "1":
        #     return m
        # else:
        return f"$^{{{m}}}$"

    # this regex matches integers with optional negative sign (hyphen)
    # TODO: more careful match only to the right of a base unit (one with no exp)
    s_new = re.sub(r"-?\d", expify, s)

    return s_new


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


def _data_var(fpath, metadata, *, from_zip=False):
    """Use the data from the YAML file to make a ``xr.Dataset()`` ``data_vars`` item.

    Optional parameters
    -------------------
    from_zip : bool or ZipFile
        False: assume the ``.npy`` file has already been extracted and read it
        ZipFile: extract the file from the zip into memory and read it

    Returns
    -------
    dict
        a ``data_vars`` entry, for creating an `xr.Dataset`
    """
    name = metadata["name"]  # we fail here if no `name`
    dims = tuple(metadata["dims"])

    # load the actual data (a single `np.ndarray` stored in a `.npy` file)
    if not from_zip:
        data = np.load(fpath)
    elif isinstance(from_zip, ZipFile):
        zf = from_zip
        fname = fpath.name
        data = np.load(BytesIO(zf.read(fname)))
    else:
        raise TypeError(f"`from_zip` should be False or type ZipFile.")

    # corrections for coords
    if name == "lat":
        data = _remove_extraneous_dim(data, 1)
    elif name == "lon":
        data = _remove_extraneous_dim(data, 0)

    # check for required attrs
    required_attrs = ("units", "name", "long_name")
    if any(attr not in metadata for attr in required_attrs):
        raise Exception(
            "'name', 'units', 'long_name' must be specified. "
            f"variable `{name}` is missing one of the latter two."
        )

    # create new unit strings
    units = metadata.get("units")
    long_units = metadata.get("long_units")
    units_tex = cf_units_to_tex(units)
    if long_units:
        long_units_tex = cf_units_to_tex(long_units)

    # orig attrs
    allowed_attrs = ("units", "long_name", "cf_standard_name", "cf_units")
    attrs = {k: v for k, v in metadata.items() if k in allowed_attrs}

    # add new attrs (new unit strings)
    attrs["units"], attrs["_units_orig"] = units_tex, attrs["units"]
    if long_units:
        attrs["long_units"], attrs["_long_units_orig"] = long_units_tex, long_units

    return {name: (dims, data, attrs)}


def _data_vars(files_data, *, from_zip=False):
    """Create xr.Dataset dat_vars items for all of the data we have."""
    if from_zip:
        zf = ZipFile(DATA_ORIG_PATH / "data.zip")
    else:
        zf = False

    dvs = {}
    for fname, meta_dict in files_data.items():
        fpath = DATA_ORIG_PATH / fname
        dv = _data_var(fpath, meta_dict, from_zip=zf)
        dvs.update(dv)
    return dvs


def create_ds(*, from_zip=False):
    """Create the xr.Dataset using the .npy data and YAML metadata description."""

    md = load_metadata()

    dvs_all = _data_vars(md["files"], from_zip=from_zip)

    # construct height manually based on the MS1 notes
    shape3d = dvs_all["theta"][1].shape  # 2nd element of the tuple is the actual data
    n_z, n_y, n_x = shape3d
    dz = 500.0  # m
    z = np.arange(0, n_z) * dz
    dv_z = {
        "hgt": (  # 3-letter fits better style of `lat`, `lon`
            "hgt",  # must match name for it to be a coordinate variable
            z,
            {
                "units": "m",
                "long_name": "Geopotential height above the surface plane",
                "cf_standard_name": "geopotential_height",  # AMIP name `zg`
            },
        )
    }
    # MS1 notes suggest this it geopotential height

    dvs_all.update(dv_z)

    coord_names = ["hgt", "lat", "lon"]
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


def write_nc(*, from_zip=False):
    """Write the nc that we will/may actually use."""
    ds = create_ds(from_zip=from_zip)
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
