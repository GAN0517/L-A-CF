# Integrated input contract

`data_model_ready.nc` is daily NetCDF with `time`, `latitude`, `longitude`, and `common_valid_mask(latitude, longitude)`. Time must be unique and strictly increasing. The current model-ready reader expects base fields `AODANA`, `ceres_all_sky_sw_down`, `t2m`, `u10`, `v10`, `blh`, `r`, `sp`, `tp`, `wind_speed_10m`, `aod_advection`, `aod_local_contrast`, `stl1`, `skt`, `sshf`, `slhf`, and `swvl1` on `(time, latitude, longitude)`. Input features are grouped in the YAML templates; 15/30-day anomalies and W m-2 flux features are recomputed by the data module.

Unit conventions: `swvl1` is m3/m3; `t2m`, `skt`, `stl1` are K; `r` is percent; `ceres_all_sky_sw_down` is W m-2. `sshf` and `slhf` are daily accumulated J m-2, divided by 86400 inside the data module to form `sshf_wm2` and `slhf_wm2`. Do not supply already averaged W m-2 under the raw `sshf` or `slhf` names. Positive ET and vegetation transpiration targets are in mm/day.

The provided configs read daily external total ET (`E`) and vegetation transpiration (`Et`) files plus `cvh`, `cvl`, `slt`, `tvh`, and `tvl` static NetCDF files. These must share the model grid and cover all requested dates. Neither these source files nor instructions for constructing them are in this package. Do not turn off external ET targets unless your integrated dataset contains compatible `e` and `evavt` raw fields.

The configured train and validation date boundaries, learning rates, batch sizes, scenario strength, and physical constraint strength are deliberately not the original experiment settings. Choose and document them before running. No released checkpoint or metric in this package claims to reproduce the original reported results.
