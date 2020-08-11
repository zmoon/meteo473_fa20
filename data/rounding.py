"""
Some of the rounding functions from my elePyant fork
"""
import numpy as np
import xarray as xr


def _round_dsd_base10(x, dsd):
    """Round to preserve decimal places (in base 10).

    Parameters
    ----------
    x : numpy.ndarray
    dsd : int
        decimal significant digits
        positive to the right of decimal point, negative to the left

    Returns
    -------
    numpy.array
        rounded array
    """
    # original elePyant method
    return np.around(x, decimals=dsd)


def _round_nsd_base10(x, nsd):
    """Round to preserve sig figs (in base 10).

    Parameters
    ----------
    x : numpy.ndarray
    nsd : int
        number of significant digits (sig figs)

    Returns
    -------
    numpy.array
        rounded array
    """
    take_log = (x != 0) & np.isfinite(x)  # avoid taking log10 of 0 and inf/inf
    tens = np.ones_like(x)
    tens[take_log] = 10 ** np.ceil(np.log10(np.abs(x[take_log].astype(np.float64))))
    # note: could instead pass `where=where_zero` to np.abs or np.log etc.

    # "true normalized" significand
    # https://en.wikipedia.org/wiki/Significand
    x_sig = x / tens

    return np.around(x_sig, decimals=nsd) * tens
    # or could pass to _round_dsd_base10


def _nsb_from_nsd(nsd):
    """Number of sigificant bits from number of sigificant digits (sig figs)."""
    return int(np.ceil(np.log(10) / np.log(2) * nsd))


def _round_binary(x, nsd, *, inplace=False):
    """Round in binary, preserving significant bits based on desired `nsd`.

    Parameters
    ----------
    x : numpy.ndarray
    nsd : int
        number of significant digits (sig figs)
    inplace : bool
        whether to round in place and modify `x`
        default False

    Returns
    -------
    numpy.array
        rounded array

    Notes
    -----
    Following:
    - https://github.com/esowc/Elefridge.jl/blob/master/src/bitrounding.jl
    - https://github.com/esowc/Elefridge.jl/blob/master/src/bitgrooming.jl
    """
    nsb = _nsb_from_nsd(nsd)
    dtype = x.dtype

    if dtype == np.float32:
        ui_type = np.uint32  # was using "<i4" (not unsigned, but integer)
        shift = 23 - nsb
        setmask = 0x003F_FFFF >> nsb
        shavemask = ~np.array(2 ** (23 - nsb) - 1, dtype=ui_type)
        and1 = 0x0000_0001

    elif dtype == np.float64:
        ui_type = np.uint64  # was using "<i8"
        shift = 52 - nsb
        setmask = 0x0007_FFFF_FFFF_FFFF >> nsb
        shavemask = ~np.array(2 ** (52 - nsb) - 1, dtype=ui_type)
        and1 = 0x0000_0000_0000_0001

    else:
        raise TypeError(f"x.dtype={dtype!r} not supported.")

    ui = x.view(ui_type)
    if not inplace:
        ui = ui.copy()

    ui += setmask + ((ui >> shift) & and1)

    return (ui & shavemask).view(dtype)


def round_array(x, nsd=None, dsd=None, in_binary=False):
    """Round a NumPy array using selected method.

    Args:
        x (numpy.ndarray)
        nsd (int)
        dsd (int)
        in_binary (bool)

    """
    if nsd and dsd:
        raise Exception(f"Must set either `nsd` or `dsd`, not both.")

    if in_binary and not nsd:
        raise Exception("Must set `nsd` to use option `in_binary`.")

    if nsd:
        if in_binary:
            return _round_binary(x, nsd)  # note not passing inplace here
        else:
            return _round_nsd_base10(x, nsd)

    elif dsd:
        return _round_dsd_base10(x, dsd)

    else:
        raise Exception("Must set either `nsd` or `dsd`.")
