

## Xarray NetCDF compression comparison

|    | name                          | size      |   sst_mean |   sst_mean_anom |   sst_std |   t_mean |   t_mean_anom |   t_std |
|---:|:------------------------------|:----------|-----------:|----------------:|----------:|---------:|--------------:|--------:|
|  0 | `ctl (test_default.nc)`       | 311.07 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  1 | `test_h5netcdf.nc`            | 311.06 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  2 | `test_h5netcdf_comp9.nc`      | 205.38 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  3 | `test_h5netcdf_comp9_nsd3.nc` | 52.55 MB  |    302.45  |     0.0292358   |  0.501643 |  442.598 |  -0.000711053 | 145.823 |
|  4 | `test_h5netcdf_comp9_nsd4.nc` | 88.98 MB  |    302.421 |    -0.00012207  |  0.226201 |  442.599 |  -7.62013e-07 | 145.822 |
|  5 | `test_h5netcdf_comp9_nsd6.nc` | 160.77 MB |    302.421 |    -3.05176e-05 |  0.224718 |  442.599 |   1.48006e-07 | 145.822 |

* `sst`: sea surface temperature (K)
* `t`: air potential temperature (K)
* `nsd`: number of significant digits ("sig figs") rounded to
* anomalies are wrt. `ctl`


Round 2: (now rounding in binary but still to preserve sig figs)

|    | name                               | size      |   sst_mean |   sst_mean_anom |   sst_std |   t_mean |   t_mean_anom |   t_std |
|---:|:-----------------------------------|:----------|-----------:|----------------:|----------:|---------:|--------------:|--------:|
|  0 | `ctl (test_default.nc)`            | 311.07 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  1 | `test_h5netcdf.nc`                 | 311.06 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  2 | `test_h5netcdf_comp9.nc`           | 205.38 MB |    302.421 |     0           |  0.224716 |  442.599 |   0           | 145.822 |
|  3 | `test_h5netcdf_comp9_nsd4.nc`      | 88.98 MB  |    302.421 |    -0.00012207  |  0.226201 |  442.599 |  -7.62013e-07 | 145.822 |
|  4 | `test_h5netcdf_comp9_nsd6.nc`      | 160.77 MB |    302.421 |    -3.05176e-05 |  0.224718 |  442.599 |   1.48006e-07 | 145.822 |
|  5 | `test_h5netcdf_comp9_nsd=3_bin.nc` | 51.79 MB  |    302.407 |    -0.0141602   |  0.226754 |  442.599 |  -1.72921e-05 | 145.822 |
|  6 | `test_h5netcdf_comp9_nsd=4_bin.nc` | 72.95 MB  |    302.421 |    -0.000152588 |  0.224683 |  442.599 |   3.25191e-07 | 145.822 |
|  7 | `test_h5netcdf_comp9_nsd=6_bin.nc` | 95.45 MB  |    302.421 |     0           |  0.224716 |  442.599 |  -9.44607e-09 | 145.822 |
