from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr


TIME_DIM = "time"
WATER_M_TO_MM = 1000.0
SPATIAL_DIMS = ("latitude", "longitude")


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class SplitBounds:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


@dataclass
class Normalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    def normalize_inputs(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: (value - self.mean[key]) / self.std[key]
            for key, value in arrays.items()
        }

    def normalize_targets(self, main: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            (main - self.mean["main_targets"]) / self.std["main_targets"],
            (aux - self.mean["aux_targets"]) / self.std["aux_targets"],
        )

    def denormalize_targets(self, main: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            main * self.std["main_targets"] + self.mean["main_targets"],
            aux * self.std["aux_targets"] + self.mean["aux_targets"],
        )


class WindowedPointDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        main_targets: np.ndarray,
        aux_targets: np.ndarray,
        time_indices: np.ndarray,
        *,
        input_window: int,
        mode_window: int | None = None,
        horizon: int,
        output_horizon: int = 1,
    ) -> None:
        self.arrays = arrays
        self.main_targets = main_targets
        self.aux_targets = aux_targets
        self.time_indices = np.asarray(time_indices, dtype=np.int64)
        self.input_window = int(input_window)
        self.mode_window = int(mode_window or input_window)
        self.horizon = int(horizon)
        self.output_horizon = int(output_horizon)
        if self.output_horizon <= 0:
            raise ValueError("output_horizon must be a positive integer.")
        self.n_points = int(main_targets.shape[1])
        self.n_time = int(main_targets.shape[0])
        if self.n_points <= 0:
            raise ValueError("Dataset has zero valid spatial points.")
        if self.time_indices.size <= 0:
            raise ValueError("Dataset has zero valid target times.")

    def __len__(self) -> int:
        return int(self.time_indices.size * self.n_points)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        time_pos = int(index // self.n_points)
        point = int(index % self.n_points)
        target_time = int(self.time_indices[time_pos])
        start = target_time - self.horizon - self.input_window + 1
        end = target_time - self.horizon + 1
        mode_start = target_time - self.horizon - self.mode_window + 1
        mode_end = target_time - self.horizon + 1
        if start < 0 or end > self.n_time:
            raise IndexError(f"Invalid time window for target index {target_time}.")
        if mode_start < 0 or mode_end > self.n_time:
            raise IndexError(f"Invalid mode window for target index {target_time}.")

        item = {}
        for key, value in self.arrays.items():
            key_start, key_end = (mode_start, mode_end) if key == "mode_context" else (start, end)
            item[key] = torch.from_numpy(value[key_start:key_end, point, :].astype(np.float32, copy=False))
        if self.output_horizon == 1:
            item["main_target"] = torch.from_numpy(
                self.main_targets[target_time, point, :].astype(np.float32, copy=False)
            )
            item["aux_target"] = torch.from_numpy(
                self.aux_targets[target_time, point, :].astype(np.float32, copy=False)
            )
        else:
            target_end = target_time + self.output_horizon
            if target_end > self.n_time:
                raise IndexError(f"Invalid output window for target index {target_time}.")
            item["main_target"] = torch.from_numpy(
                self.main_targets[target_time:target_end, point, :].astype(np.float32, copy=False)
            )
            item["aux_target"] = torch.from_numpy(
                self.aux_targets[target_time:target_end, point, :].astype(np.float32, copy=False)
            )
        item["time_index"] = torch.tensor(target_time, dtype=torch.long)
        item["point_index"] = torch.tensor(point, dtype=torch.long)
        return item


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_time_value(value: str) -> np.datetime64:
    return np.datetime64(value)


def _stack_points(da: xr.DataArray, mask_values: np.ndarray) -> np.ndarray:
    stacked = da.stack(point=("latitude", "longitude")).isel(point=mask_values)
    return np.asarray(stacked.values, dtype=np.float32)


def _resolve_existing_path(path_cfg: Any, *, label: str) -> Path:
    candidates: list[Path] = []
    if isinstance(path_cfg, (str, Path)):
        candidates.append(Path(path_cfg))
    elif isinstance(path_cfg, dict):
        if path_cfg.get("path"):
            candidates.append(Path(path_cfg["path"]))
        candidates.extend(Path(path) for path in path_cfg.get("path_candidates", []))
    else:
        raise TypeError(f"{label} must be a path string or mapping with path/path_candidates.")

    for path in candidates:
        if path.exists():
            return path
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried: {rendered}")


def _align_external_daily_field(
    base_ds: xr.Dataset,
    *,
    path_cfg: Any,
    variable: str,
    output_name: str,
    source_name: str,
) -> xr.DataArray:
    path = _resolve_existing_path(path_cfg, label=f"{source_name} {output_name}")
    handle = xr.open_dataset(path, engine="netcdf4")
    try:
        ext_ds = handle
        if TIME_DIM not in ext_ds.dims and "valid_time" in ext_ds.dims:
            ext_ds = ext_ds.rename({"valid_time": TIME_DIM})
        if variable not in ext_ds.data_vars:
            raise KeyError(f"{path} does not contain variable {variable!r}.")
        da = ext_ds[variable]
        if "latitude" not in da.dims or "longitude" not in da.dims or TIME_DIM not in da.dims:
            raise ValueError(
                f"{source_name} {variable} must have dimensions including "
                f"{TIME_DIM}, latitude, and longitude; got {da.dims}."
            )
        if np.any(np.diff(np.asarray(da["latitude"].values, dtype=np.float64)) < 0):
            da = da.sortby("latitude")
        da = da.sel(
            {
                TIME_DIM: base_ds[TIME_DIM].values,
                "latitude": base_ds["latitude"].values,
                "longitude": base_ds["longitude"].values,
            },
            method="nearest",
            tolerance=1e-6,
        )
        da = da.transpose(TIME_DIM, *SPATIAL_DIMS).astype(np.float32)
        da.name = output_name
        da.attrs.update(
            {
                "units": "mm/day",
                "source_dataset": source_name,
                "source_path": str(path),
                "source_variable": variable,
                "description": f"{output_name} replaced by {source_name} {variable}.",
            }
        )
        return da.load()
    finally:
        handle.close()


def _align_external_static_field(
    base_ds: xr.Dataset,
    *,
    path_cfg: Any,
    variable: str,
    output_name: str,
    source_name: str,
) -> xr.DataArray:
    path = _resolve_existing_path(path_cfg, label=f"{source_name} {output_name}")
    handle = xr.open_dataset(path, engine="netcdf4")
    try:
        ext_ds = handle
        if variable not in ext_ds.data_vars:
            raise KeyError(f"{path} does not contain variable {variable!r}.")
        da = ext_ds[variable]
        if TIME_DIM in da.dims:
            if int(da.sizes[TIME_DIM]) != 1:
                raise ValueError(
                    f"Static land variable {variable!r} must have one time step, "
                    f"got {int(da.sizes[TIME_DIM])}."
                )
            da = da.isel({TIME_DIM: 0}, drop=True)
        if "latitude" not in da.dims or "longitude" not in da.dims:
            raise ValueError(
                f"{source_name} {variable} must have latitude and longitude dimensions; "
                f"got {da.dims}."
            )
        if np.any(np.diff(np.asarray(da["latitude"].values, dtype=np.float64)) < 0):
            da = da.sortby("latitude")
        da = da.sel(
            {
                "latitude": base_ds["latitude"].values,
                "longitude": base_ds["longitude"].values,
            },
            method="nearest",
            tolerance=0.06,
        )
        da = da.transpose(*SPATIAL_DIMS).astype(np.float32).load()
        expanded = da.expand_dims({TIME_DIM: base_ds[TIME_DIM].values})
        expanded = expanded.transpose(TIME_DIM, *SPATIAL_DIMS).astype(np.float32)
        expanded.name = output_name
        expanded.attrs.update(
            {
                "units": da.attrs.get("units", ""),
                "long_name": da.attrs.get("long_name", ""),
                "source_dataset": source_name,
                "source_path": str(path),
                "source_variable": variable,
                "description": (
                    f"Static land feature {output_name} aligned to the model grid "
                    "and repeated across all time steps."
                ),
            }
        )
        return expanded
    finally:
        handle.close()


def _static_land_features_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("enabled", False))


def _resolve_static_land_feature_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    variables = config.get("variables", [])
    source_dir = config.get("source_dir")
    specs: list[dict[str, Any]] = []
    for item in variables:
        if isinstance(item, str):
            spec: dict[str, Any] = {"name": item, "var": item}
        elif isinstance(item, dict):
            spec = dict(item)
        else:
            raise TypeError("data.static_land_features.variables entries must be strings or mappings.")
        name = str(spec.get("name") or spec.get("var") or "")
        var = str(spec.get("var") or name)
        if not name or not var:
            raise ValueError("Each static land feature needs a non-empty name and var.")
        if "path" not in spec and "path_candidates" not in spec:
            if not source_dir:
                raise ValueError(
                    f"Static land feature {name!r} needs path/path_candidates or data.static_land_features.source_dir."
                )
            spec["path"] = str(Path(source_dir) / f"{var}.nc")
        spec["name"] = name
        spec["var"] = var
        specs.append(spec)
    return specs


def _add_static_land_features(ds: xr.Dataset, config: dict[str, Any] | None) -> xr.Dataset:
    if not _static_land_features_enabled(config):
        return ds
    cfg = config or {}
    source_name = str(cfg.get("source_name", "static_land_features"))
    out = ds.copy()
    added_names = []
    for spec in _resolve_static_land_feature_specs(cfg):
        name = str(spec["name"])
        out[name] = _align_external_static_field(
            out,
            path_cfg=spec,
            variable=str(spec["var"]),
            output_name=name,
            source_name=source_name,
        )
        added_names.append(name)
    out.attrs["static_land_features"] = ", ".join(added_names)
    return out


def _external_et_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("enabled", False))


def _add_external_et_targets(ds: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    out = ds.copy()
    source_name = str(config.get("source_name", "external_et"))
    total_cfg = config.get("total_et", {})
    transpiration_cfg = config.get("vegetation_transpiration", {})
    bare_soil_cfg = config.get("bare_soil_evaporation")
    out["e_pos"] = _align_external_daily_field(
        out,
        path_cfg=total_cfg,
        variable=str(total_cfg.get("var", "E")),
        output_name="e_pos",
        source_name=source_name,
    )
    out["evavt_pos"] = _align_external_daily_field(
        out,
        path_cfg=transpiration_cfg,
        variable=str(transpiration_cfg.get("var", "Et")),
        output_name="evavt_pos",
        source_name=source_name,
    )
    if bare_soil_cfg:
        out["eb_pos"] = _align_external_daily_field(
            out,
            path_cfg=bare_soil_cfg,
            variable=str(bare_soil_cfg.get("var", "Eb")),
            output_name="eb_pos",
            source_name=source_name,
        )
    out.attrs["external_et_targets"] = (
        f"e_pos, evavt_pos, and optional eb_pos replaced by {source_name}; "
        "swvl1 and all forcing inputs remain from the base model-ready dataset."
    )
    return out


def _add_derived_variables(ds: xr.Dataset, external_et_config: dict[str, Any] | None = None) -> xr.Dataset:
    out = ds.copy()
    if _external_et_enabled(external_et_config):
        out = _add_external_et_targets(out, external_et_config or {})
    else:
        out["e_pos"] = (-out["e"] * WATER_M_TO_MM).astype(np.float32)
        out["e_pos"].attrs.update(
            {
                "units": "mm/day",
                "source_units": out["e"].attrs.get("units", "m of water equivalent"),
                "source_dataset": "ERA5-Land",
                "description": "Positive upward total evapotranspiration from -e converted to mm/day.",
            }
        )
        out["evavt_pos"] = (-out["evavt"] * WATER_M_TO_MM).astype(np.float32)
        out["evavt_pos"].attrs.update(
            {
                "units": "mm/day",
                "source_units": out["evavt"].attrs.get("units", "m of water equivalent"),
                "source_dataset": "ERA5-Land",
                "description": "Positive vegetation transpiration from -evavt converted to mm/day.",
            }
        )
    out["sshf_wm2"] = (out["sshf"] / 86400.0).astype(np.float32)
    out["sshf_wm2"].attrs.update({"units": "W m-2", "description": "Daily mean sensible heat flux from sshf / 86400."})
    out["slhf_wm2"] = (out["slhf"] / 86400.0).astype(np.float32)
    out["slhf_wm2"].attrs.update({"units": "W m-2", "description": "Daily mean latent heat flux from slhf / 86400."})
    return out


def _add_splitwise_anomaly_variables(ds: xr.Dataset, split_bounds: SplitBounds) -> xr.Dataset:
    out = ds.copy()
    for window in (15, 30):
        out[f"swvl1_anom{window}d"] = _splitwise_trailing_anomaly(out["swvl1"], window, split_bounds)
        out[f"e_pos_anom{window}d"] = _splitwise_trailing_anomaly(out["e_pos"], window, split_bounds)
        out[f"evavt_pos_anom{window}d"] = _splitwise_trailing_anomaly(out["evavt_pos"], window, split_bounds)
        if "eb_pos" in out:
            out[f"eb_pos_anom{window}d"] = _splitwise_trailing_anomaly(out["eb_pos"], window, split_bounds)
        out[f"t2m_anom{window}d"] = _splitwise_trailing_anomaly(out["t2m"], window, split_bounds)
        out[f"ceres_all_sky_sw_down_anom{window}d"] = _splitwise_trailing_anomaly(
            out["ceres_all_sky_sw_down"], window, split_bounds
        )
        out[f"sshf_wm2_anom{window}d"] = _splitwise_trailing_anomaly(out["sshf_wm2"], window, split_bounds)
        out[f"slhf_wm2_anom{window}d"] = _splitwise_trailing_anomaly(out["slhf_wm2"], window, split_bounds)
    return out


def _trailing_anomaly(da: xr.DataArray, window: int) -> xr.DataArray:
    """
    Causal rolling anomaly using only the current and previous days.

    The stored NetCDF anomaly variables may have been produced with centred
    rolling windows. Those are useful descriptively but leak future information
    for one-day-ahead prediction, so model training recomputes anomalies here.
    """
    out = da - da.rolling({TIME_DIM: int(window)}, center=False, min_periods=1).mean()
    out = out.astype(np.float32)
    out.attrs.update(
        {
            "source_variable": da.name or "",
            "anomaly_window_days": int(window),
            "description": "Causal trailing rolling anomaly; uses only current and previous days.",
        }
    )
    return out


def _splitwise_trailing_anomaly(
    da: xr.DataArray,
    window: int,
    split_bounds: SplitBounds,
) -> xr.DataArray:
    pieces = []
    for split_name in ("train", "val", "test"):
        start_idx, end_idx = getattr(split_bounds, split_name)
        if end_idx < start_idx:
            continue
        segment = da.isel({TIME_DIM: slice(start_idx, end_idx + 1)})
        piece = _trailing_anomaly(segment, window)
        piece.attrs.update(
            {
                "split": split_name,
                "description": (
                    "Split-local causal trailing rolling anomaly; rolling state is reset at "
                    "train/val/test boundaries to avoid cross-split leakage."
                ),
            }
        )
        pieces.append(piece)
    if not pieces:
        raise ValueError("Cannot build splitwise anomaly: every split is empty.")
    out = xr.concat(pieces, dim=TIME_DIM).sortby(TIME_DIM)
    out.attrs.update(
        {
            "source_variable": da.name or "",
            "anomaly_window_days": int(window),
            "description": (
                "Split-local causal trailing rolling anomaly; uses only current and "
                "previous days within the same train/val/test split."
            ),
        }
    )
    return out.astype(np.float32)


def _build_feature_array(
    ds: xr.Dataset,
    variables: list[str],
    mask_values: np.ndarray,
) -> np.ndarray:
    if not variables:
        n_time = int(ds.sizes[TIME_DIM])
        n_points = int(mask_values.size)
        return np.empty((n_time, n_points, 0), dtype=np.float32)
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Dataset is missing required variables: {missing}")
    parts = [_stack_points(ds[name], mask_values) for name in variables]
    return np.stack(parts, axis=-1).astype(np.float32)


def _build_global_mode_context_array(
    ds: xr.Dataset,
    variables: list[str],
    mask_values: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if not variables:
        n_time = int(ds.sizes[TIME_DIM])
        n_points = int(mask_values.size)
        return np.empty((n_time, n_points, 0), dtype=np.float32), []

    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Dataset is missing required mode variables: {missing}")
    if "common_valid_mask" not in ds:
        raise KeyError("Global mode context requires common_valid_mask.")

    mask_2d = np.asarray(ds["common_valid_mask"].values, dtype=bool)
    lat_weights = np.cos(np.deg2rad(np.asarray(ds["latitude"].values, dtype=np.float64)))
    weight_2d = lat_weights[:, None] * mask_2d.astype(np.float64)
    weight_flat = weight_2d.reshape(-1)
    valid_flat = np.flatnonzero(weight_flat > 0)
    weights = weight_flat[valid_flat]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("Cannot build global mode context: valid mask has zero weighted area.")

    features = []
    feature_names = []
    for name in variables:
        values = np.asarray(ds[name].stack(point=("latitude", "longitude")).isel(point=valid_flat).values, dtype=np.float64)
        finite = np.isfinite(values)
        weighted_valid = np.where(finite, weights[None, :], 0.0)
        denom = np.sum(weighted_valid, axis=1)
        safe_denom = np.where(denom <= 0, np.nan, denom)
        mean = np.nansum(np.where(finite, values * weights[None, :], 0.0), axis=1) / safe_denom
        diff = values - mean[:, None]
        variance = np.nansum(np.where(finite, (diff**2) * weights[None, :], 0.0), axis=1) / safe_denom
        std = np.sqrt(np.maximum(variance, 0.0))
        features.extend([mean.astype(np.float32), std.astype(np.float32)])
        feature_names.extend([f"{name}_global_mean", f"{name}_global_std"])

    mode_time = np.stack(features, axis=-1).astype(np.float32)
    mode_time_points = np.repeat(mode_time[:, None, :], int(mask_values.size), axis=1)
    return mode_time_points, feature_names


def _valid_target_times(
    times: np.ndarray,
    *,
    input_window: int,
    horizon: int,
    output_horizon: int = 1,
    train_end: str,
    val_end: str,
) -> SplitIndices:
    bounds = _split_bounds(times, train_end=train_end, val_end=val_end)
    train = _target_indices_within_bounds(
        bounds.train,
        input_window=input_window,
        horizon=horizon,
        output_horizon=output_horizon,
    )
    val = _target_indices_within_bounds(
        bounds.val,
        input_window=input_window,
        horizon=horizon,
        output_horizon=output_horizon,
    )
    test = _target_indices_within_bounds(
        bounds.test,
        input_window=input_window,
        horizon=horizon,
        output_horizon=output_horizon,
    )
    return SplitIndices(train=train, val=val, test=test)


def _split_bounds(times: np.ndarray, *, train_end: str, val_end: str) -> SplitBounds:
    train_end_value = _as_time_value(train_end)
    val_end_value = _as_time_value(val_end)
    train_end_candidates = np.flatnonzero(times <= train_end_value)
    val_end_candidates = np.flatnonzero(times <= val_end_value)
    if train_end_candidates.size == 0:
        raise ValueError(f"No training dates found up to train_end={train_end}.")
    if val_end_candidates.size == 0:
        raise ValueError(f"No validation dates found up to val_end={val_end}.")
    train_end_idx = int(train_end_candidates[-1])
    val_end_idx = int(val_end_candidates[-1])
    if val_end_idx <= train_end_idx:
        raise ValueError("val_end must be later than train_end.")
    return SplitBounds(
        train=(0, train_end_idx),
        val=(train_end_idx + 1, val_end_idx),
        test=(val_end_idx + 1, len(times) - 1),
    )


def _target_indices_within_bounds(
    bounds: tuple[int, int],
    *,
    input_window: int,
    horizon: int,
    output_horizon: int = 1,
) -> np.ndarray:
    start_idx, end_idx = bounds
    first_target = int(start_idx) + int(input_window) + int(horizon) - 1
    last_target_start = int(end_idx) - int(output_horizon) + 1
    if first_target > last_target_start:
        return np.asarray([], dtype=np.int64)
    return np.arange(first_target, last_target_start + 1, dtype=np.int64)


def _compute_stats(
    arrays: dict[str, np.ndarray],
    main_targets: np.ndarray,
    aux_targets: np.ndarray,
    train_times: np.ndarray,
) -> Normalizer:
    mean: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}

    for key, value in arrays.items():
        train_value = value[train_times, :, :]
        mean[key] = np.nanmean(train_value, axis=(0, 1), dtype=np.float64).astype(np.float32)
        std_value = np.nanstd(train_value, axis=(0, 1), dtype=np.float64).astype(np.float32)
        std[key] = np.where(std_value < 1e-6, 1.0, std_value).astype(np.float32)

    for key, value in (("main_targets", main_targets), ("aux_targets", aux_targets)):
        train_value = value[train_times, :, :]
        mean[key] = np.nanmean(train_value, axis=(0, 1), dtype=np.float64).astype(np.float32)
        std_value = np.nanstd(train_value, axis=(0, 1), dtype=np.float64).astype(np.float32)
        std[key] = np.where(std_value < 1e-6, 1.0, std_value).astype(np.float32)

    return Normalizer(mean=mean, std=std)


def _finite_points(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3:
        raise ValueError(f"Expected array with shape [time, point, feature], got {value.shape}.")
    if value.shape[-1] == 0:
        return np.ones(value.shape[1], dtype=bool)
    return np.isfinite(value).all(axis=(0, 2))


def _filter_finite_points(
    arrays: dict[str, np.ndarray],
    main_targets: np.ndarray,
    aux_targets: np.ndarray,
    valid_point_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    point_mask = np.ones(int(valid_point_indices.size), dtype=bool)
    group_valid_counts: dict[str, int] = {}

    for key, value in arrays.items():
        finite = _finite_points(value)
        group_valid_counts[key] = int(finite.sum())
        point_mask &= finite

    for key, value in (("main_targets", main_targets), ("aux_targets", aux_targets)):
        finite = _finite_points(value)
        group_valid_counts[key] = int(finite.sum())
        point_mask &= finite

    kept = int(point_mask.sum())
    original = int(valid_point_indices.size)
    if kept <= 0:
        raise ValueError(
            "No finite spatial points remain after checking all inputs and targets. "
            f"Per-group finite point counts: {group_valid_counts}"
        )
    if kept == original:
        diagnostics = {
            "enabled": True,
            "original_points": original,
            "kept_points": kept,
            "removed_points": 0,
            "group_valid_counts": group_valid_counts,
        }
        return arrays, main_targets, aux_targets, valid_point_indices, diagnostics

    filtered_arrays = {key: value[:, point_mask, :] for key, value in arrays.items()}
    diagnostics = {
        "enabled": True,
        "original_points": original,
        "kept_points": kept,
        "removed_points": original - kept,
        "group_valid_counts": group_valid_counts,
    }
    return (
        filtered_arrays,
        main_targets[:, point_mask, :],
        aux_targets[:, point_mask, :],
        valid_point_indices[point_mask],
        diagnostics,
    )


def prepare_dataloaders(
    config: dict[str, Any],
    normalizer_override: Normalizer | None = None,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    data_cfg = config["data"]
    vars_cfg = config["variables"]
    input_window = int(data_cfg["input_window"])
    mode_window = int(data_cfg.get("mode_window", input_window))
    horizon = int(data_cfg["horizon"])
    output_horizon = int(data_cfg.get("output_horizon", 1))
    seed = int(data_cfg.get("seed", 42))

    ds_handle = xr.open_dataset(data_cfg["path"], engine="netcdf4")
    ds = _add_derived_variables(ds_handle.load(), data_cfg.get("external_et_targets"))
    ds_handle.close()
    ds = _add_static_land_features(ds, data_cfg.get("static_land_features"))
    split_bounds = _split_bounds(
        ds[TIME_DIM].values,
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )
    ds = _add_splitwise_anomaly_variables(ds, split_bounds)

    if "common_valid_mask" not in ds:
        raise KeyError("data_model_ready.nc must contain common_valid_mask.")
    mask = ds["common_valid_mask"].stack(point=("latitude", "longitude")).values.astype(bool)
    valid_point_indices = np.flatnonzero(mask)
    max_points = data_cfg.get("max_points")
    if max_points is not None:
        rng = np.random.default_rng(seed)
        max_points_int = int(max_points)
        if max_points_int < valid_point_indices.size:
            valid_point_indices = np.sort(rng.choice(valid_point_indices, size=max_points_int, replace=False))

    arrays = {
        "atm_total": _build_feature_array(ds, vars_cfg["atm_total"], valid_point_indices),
        "land_total": _build_feature_array(ds, vars_cfg["land_total"], valid_point_indices),
        "atm_anom": _build_feature_array(ds, vars_cfg["atm_anom"], valid_point_indices),
        "land_anom": _build_feature_array(ds, vars_cfg["land_anom"], valid_point_indices),
    }
    mode_feature_names: list[str] = []
    if "mode_variables" in vars_cfg:
        arrays["mode_context"], mode_feature_names = _build_global_mode_context_array(
            ds,
            vars_cfg.get("mode_variables", []),
            valid_point_indices,
        )
    main_targets = _build_feature_array(ds, vars_cfg["main_targets"], valid_point_indices)
    aux_targets = _build_feature_array(ds, vars_cfg["aux_targets"], valid_point_indices)
    arrays, main_targets, aux_targets, valid_point_indices, finite_point_filter = _filter_finite_points(
        arrays,
        main_targets,
        aux_targets,
        valid_point_indices,
    )

    split_indices = _valid_target_times(
        ds[TIME_DIM].values,
        input_window=max(input_window, mode_window),
        horizon=horizon,
        output_horizon=output_horizon,
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )
    stats_times = np.arange(0, int(split_bounds.train[1]) + 1, dtype=np.int64)
    normalizer = normalizer_override or _compute_stats(arrays, main_targets, aux_targets, stats_times)
    arrays_norm = normalizer.normalize_inputs(arrays)
    main_norm, aux_norm = normalizer.normalize_targets(main_targets, aux_targets)

    datasets = {
        "train": WindowedPointDataset(
            arrays_norm,
            main_norm,
            aux_norm,
            split_indices.train,
            input_window=input_window,
            mode_window=mode_window,
            horizon=horizon,
            output_horizon=output_horizon,
        ),
        "val": WindowedPointDataset(
            arrays_norm,
            main_norm,
            aux_norm,
            split_indices.val,
            input_window=input_window,
            mode_window=mode_window,
            horizon=horizon,
            output_horizon=output_horizon,
        ),
        "test": WindowedPointDataset(
            arrays_norm,
            main_norm,
            aux_norm,
            split_indices.test,
            input_window=input_window,
            mode_window=mode_window,
            horizon=horizon,
            output_horizon=output_horizon,
        ),
    }

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }

    metadata = {
        "normalizer": normalizer,
        "valid_point_indices": valid_point_indices,
        "latitudes": ds["latitude"].values,
        "longitudes": ds["longitude"].values,
        "times": ds[TIME_DIM].values,
        "split_indices": split_indices,
        "split_bounds": split_bounds,
        "input_dims": {key: arrays[key].shape[-1] for key in arrays},
        "input_window": input_window,
        "mode_window": mode_window,
        "output_horizon": output_horizon,
        "main_targets": list(vars_cfg["main_targets"]),
        "aux_targets": list(vars_cfg["aux_targets"]),
        "main_target_units": [ds[name].attrs.get("units", "") for name in vars_cfg["main_targets"]],
        "aux_target_units": [ds[name].attrs.get("units", "") for name in vars_cfg["aux_targets"]],
        "feature_groups": {key: list(vars_cfg[key]) for key in ("atm_total", "land_total", "atm_anom", "land_anom")},
        "mode_variables": list(vars_cfg.get("mode_variables", [])),
        "mode_features": mode_feature_names,
        "unit_convention": "swvl1=m3/m3; e_pos/evavt_pos/eb_pos=mm/day; heat_flux=W m-2",
        "external_et_targets": data_cfg.get("external_et_targets", {"enabled": False}),
        "static_land_features": data_cfg.get("static_land_features", {"enabled": False}),
        "finite_point_filter": finite_point_filter,
    }
    return loaders, metadata
