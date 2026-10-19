from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


BASE_FIELDS = {
    "AODANA", "ceres_all_sky_sw_down", "t2m", "u10", "v10", "blh",
    "r", "sp", "tp", "wind_speed_10m", "aod_advection",
    "aod_local_contrast", "stl1", "skt", "sshf", "slhf", "swvl1",
}
DAILY_DIMS = ("time", "latitude", "longitude")


def validate(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    data_path = Path(data["path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Integrated input not found: {data_path}")
    with xr.open_dataset(data_path) as ds:
        missing = sorted(BASE_FIELDS - set(ds.data_vars))
        if missing:
            raise ValueError(f"Integrated input is missing: {missing}")
        if "common_valid_mask" not in ds or ds["common_valid_mask"].dims != ("latitude", "longitude"):
            raise ValueError("common_valid_mask must have dimensions (latitude, longitude).")
        wrong_dims = [name for name in sorted(BASE_FIELDS) if ds[name].dims != DAILY_DIMS]
        if wrong_dims:
            raise ValueError(f"Expected (time, latitude, longitude) for: {wrong_dims}")
        for name in ("sshf", "slhf"):
            units = str(ds[name].attrs.get("units", "")).lower().replace(" ", "")
            if "w" in units and ("m-2" in units or "m**-2" in units):
                raise ValueError(f"{name} is labeled W m-2, but the loader expects daily J m-2.")
        times = ds["time"].values.astype("datetime64[D]")
        if times.size < 8 or not np.all(np.diff(times).astype("timedelta64[D]").astype(int) == 1):
            raise ValueError("Expected at least eight consecutive daily timestamps.")
        n_valid = int(ds["common_valid_mask"].sum().item())
        if n_valid == 0:
            raise ValueError("common_valid_mask contains no valid cells.")
        print(f"Integrated input: {data_path} | days={times.size} | valid cells={n_valid}")

    external = data.get("external_et_targets", {})
    if external.get("enabled"):
        for key in ("total_et", "vegetation_transpiration"):
            field_path = Path(external[key]["path"])
            if not field_path.is_file():
                raise FileNotFoundError(f"{key} not found: {field_path}")
            with xr.open_dataset(field_path) as ds:
                variable = external[key]["var"]
                if variable not in ds:
                    raise ValueError(f"{field_path} lacks {variable!r}.")
    static = data.get("static_land_features", {})
    if static.get("enabled"):
        for item in static["variables"]:
            field_path = Path(item["path"])
            if not field_path.is_file():
                raise FileNotFoundError(f"Static feature not found: {field_path}")
            with xr.open_dataset(field_path) as ds:
                if item["var"] not in ds:
                    raise ValueError(f"{field_path} lacks {item['var']!r}.")
    print("File and dimension checks passed. Verify remaining physical units against DATA_CONTRACT.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the integrated L-A model input.")
    parser.add_argument("--config", type=Path, required=True)
    validate(parser.parse_args().config)
