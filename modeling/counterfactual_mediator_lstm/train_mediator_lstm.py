from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from data_module import (  # noqa: E402
    TIME_DIM,
    Normalizer,
    _add_derived_variables,
    _add_splitwise_anomaly_variables,
    _add_static_land_features,
    _build_feature_array,
    _compute_stats,
    _filter_finite_points,
    _split_bounds,
    _valid_target_times,
    load_yaml_config,
)
from metrics import regression_metrics  # noqa: E402
from model_mediator_lstm import CounterfactualMediatorLSTM  # noqa: E402


DEFAULT_MEDIATORS = ("r", "t2m", "skt", "sshf_wm2", "slhf_wm2")
DEFAULT_MAIN_TARGETS = ("swvl1", "e_pos", "evavt_pos")
DEFAULT_DROPPED_INPUTS: tuple[str, ...] = ()
MEDIATOR_WRITEBACK_RULES = {
    "r": {"dataset_variable": "r", "scale_factor_to_dataset_units": 1.0},
    "t2m": {"dataset_variable": "t2m", "scale_factor_to_dataset_units": 1.0},
    "skt": {"dataset_variable": "skt", "scale_factor_to_dataset_units": 1.0},
    "sshf_wm2": {"dataset_variable": "sshf", "scale_factor_to_dataset_units": 86400.0},
    "slhf_wm2": {"dataset_variable": "slhf", "scale_factor_to_dataset_units": 86400.0},
}
INPUT_GROUPS = ("atm_total", "land_total", "atm_anom", "land_anom")
DERIVED_SUFFIXES = ("_anom15d", "_anom30d")


@dataclass
class MediatorData:
    loaders: dict[str, DataLoader]
    metadata: dict[str, Any]


class MediatorWindowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        mediator_targets: np.ndarray,
        time_indices: np.ndarray,
        *,
        input_window: int,
        horizon: int,
        output_horizon: int,
    ) -> None:
        self.arrays = arrays
        self.mediator_targets = mediator_targets
        self.time_indices = np.asarray(time_indices, dtype=np.int64)
        self.input_window = int(input_window)
        self.horizon = int(horizon)
        self.output_horizon = int(output_horizon)
        self.n_time = int(mediator_targets.shape[0])
        self.n_points = int(mediator_targets.shape[1])
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
        if start < 0 or end > self.n_time:
            raise IndexError(f"Invalid input window for target index {target_time}.")
        target_end = target_time + self.output_horizon
        if target_end > self.n_time:
            raise IndexError(f"Invalid output window for target index {target_time}.")

        item: dict[str, torch.Tensor] = {}
        for key, value in self.arrays.items():
            item[key] = torch.from_numpy(value[start:end, point, :].astype(np.float32, copy=False))

        base_time = target_time - self.horizon
        target_abs = self.mediator_targets[target_time:target_end, point, :].astype(np.float32, copy=False)
        residual_base = self.mediator_targets[base_time, point, :].astype(np.float32, copy=False)
        target_delta = target_abs - residual_base[None, :]
        item["target_abs"] = torch.from_numpy(target_abs)
        item["target_delta"] = torch.from_numpy(target_delta)
        item["residual_base"] = torch.from_numpy(residual_base)
        item["time_index"] = torch.tensor(target_time, dtype=torch.long)
        item["point_index"] = torch.tensor(point, dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train counterfactual mediator LSTM models. Two modes are supported: "
            "absolute predicts normalized mediator values with residual base; "
            "delta predicts normalized next-day changes from the residual base."
        )
    )
    parser.add_argument("--config", type=Path, default=PARENT_DIR / "config_v4_single_swvl1.yaml")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mediator-targets", nargs="+", default=list(DEFAULT_MEDIATORS))
    parser.add_argument("--exclude-input-vars", nargs="+", default=None)
    parser.add_argument("--model-types", nargs="+", choices=("absolute", "delta"), default=["delta"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-horizon", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--h-sign-constraint-weight",
        type=float,
        help=(
            "Weight for the SKT-T2M-SSHF soft sign constraint. "
            "The current sshf_wm2 convention is ERA5 downward-positive, so the "
            "constraint penalizes positive delta_sshf_wm2 * (delta_skt - delta_t2m)."
        ),
    )
    parser.add_argument("--h-sign-flux-scale", type=float, help="Scale H_s for sshf_wm2 delta, in W m-2.")
    parser.add_argument("--h-sign-temp-scale", type=float, help="Scale T_s for SKT-T2M delta, in K.")
    parser.add_argument("--skip-test-eval", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.data_path is not None:
        cfg["data"]["path"] = str(args.data_path)
    if args.output_root is not None:
        cfg["data"]["output_dir"] = str(args.output_root)
    else:
        cfg["data"]["output_dir"] = str(PARENT_DIR.parent / "outputs" / "mediator_lstm")
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.output_horizon is not None:
        cfg["data"]["output_horizon"] = int(args.output_horizon)
    else:
        cfg["data"]["output_horizon"] = 1
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    if args.val_interval is not None:
        cfg["training"]["val_interval"] = int(args.val_interval)
    if args.patience is not None:
        cfg["training"]["patience"] = int(args.patience)
    if args.seed is not None:
        cfg["data"]["seed"] = int(args.seed)
    if args.device is not None and args.gpu_id is not None:
        raise ValueError("Use only one of --device or --gpu-id.")
    if args.gpu_id is not None:
        cfg["training"]["device"] = f"cuda:{int(args.gpu_id)}"
    elif args.device is not None:
        cfg["training"]["device"] = args.device
    cfg["mediator_model"] = {
        "hidden_dim": int(args.hidden_dim if args.hidden_dim is not None else cfg.get("model", {}).get("hidden_dim", 128)),
        "dropout": float(args.dropout if args.dropout is not None else cfg.get("model", {}).get("dropout", 0.1)),
    }
    physical_constraints = dict(cfg.get("physical_constraints", {}))
    physical_constraints.setdefault("h_sign_constraint_weight", 0.0)
    physical_constraints.setdefault("h_sign_flux_scale", 10.0)
    physical_constraints.setdefault("h_sign_temp_scale", 1.0)
    physical_constraints["h_sign_convention"] = (
        "sshf_wm2 is ERA5 downward-positive. Upward sensible heat delta is "
        "-delta_sshf_wm2, so the sign constraint penalizes "
        "positive delta_sshf_wm2 * (delta_skt - delta_t2m)."
    )
    if args.h_sign_constraint_weight is not None:
        physical_constraints["h_sign_constraint_weight"] = float(args.h_sign_constraint_weight)
    if args.h_sign_flux_scale is not None:
        physical_constraints["h_sign_flux_scale"] = float(args.h_sign_flux_scale)
    if args.h_sign_temp_scale is not None:
        physical_constraints["h_sign_temp_scale"] = float(args.h_sign_temp_scale)
    cfg["physical_constraints"] = physical_constraints
    return cfg


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=to_jsonable)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_arrays(arrays: dict[str, np.ndarray], normalizer: Normalizer) -> dict[str, np.ndarray]:
    return {key: (value - normalizer.mean[key]) / normalizer.std[key] for key, value in arrays.items()}


def expand_default_exclusions(mediator_targets: list[str]) -> list[str]:
    excluded = list(DEFAULT_MAIN_TARGETS) + list(DEFAULT_DROPPED_INPUTS) + list(mediator_targets)
    for name in mediator_targets:
        excluded.extend(f"{name}{suffix}" for suffix in DERIVED_SUFFIXES)
    return list(dict.fromkeys(excluded))


def prepare_mediator_dataloaders(
    config: dict[str, Any],
    mediator_targets: list[str],
    normalizer_override: Normalizer | None = None,
) -> MediatorData:
    data_cfg = config["data"]
    vars_cfg = config["variables"]
    input_window = int(data_cfg["input_window"])
    horizon = int(data_cfg["horizon"])
    output_horizon = int(data_cfg.get("output_horizon", 1))
    seed = int(data_cfg.get("seed", 42))

    import xarray as xr

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
        max_points_int = int(max_points)
        if max_points_int < valid_point_indices.size:
            rng = np.random.default_rng(seed)
            valid_point_indices = np.sort(rng.choice(valid_point_indices, size=max_points_int, replace=False))

    exclude_vars = set(config["mediator_setup"]["exclude_input_vars"])
    input_variables: dict[str, list[str]] = {}
    excluded_by_group: dict[str, list[str]] = {}
    for group in INPUT_GROUPS:
        group_vars = list(vars_cfg.get(group, []))
        input_variables[group] = [name for name in group_vars if name not in exclude_vars]
        excluded_by_group[group] = [name for name in group_vars if name in exclude_vars]
        if not input_variables[group]:
            input_variables[group] = []

    arrays = {
        group: _build_feature_array(ds, input_variables[group], valid_point_indices)
        for group in INPUT_GROUPS
    }
    mediator_raw = _build_feature_array(ds, mediator_targets, valid_point_indices)
    aux_empty = np.empty((mediator_raw.shape[0], mediator_raw.shape[1], 0), dtype=np.float32)
    arrays, mediator_raw, aux_empty, _, valid_point_indices, finite_point_filter = _filter_finite_points(
        arrays,
        mediator_raw,
        aux_empty,
        None,
        valid_point_indices,
    )

    split_indices = _valid_target_times(
        ds[TIME_DIM].values,
        input_window=input_window,
        horizon=horizon,
        output_horizon=output_horizon,
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )
    stats_times = np.arange(0, int(split_bounds.train[1]) + 1, dtype=np.int64)
    normalizer = normalizer_override or _compute_stats(arrays, mediator_raw, aux_empty, None, stats_times)
    arrays_norm = normalize_arrays(arrays, normalizer)
    mediator_norm = (mediator_raw - normalizer.mean["main_targets"]) / normalizer.std["main_targets"]

    datasets = {
        "train": MediatorWindowDataset(
            arrays_norm,
            mediator_norm,
            split_indices.train,
            input_window=input_window,
            horizon=horizon,
            output_horizon=output_horizon,
        ),
        "val": MediatorWindowDataset(
            arrays_norm,
            mediator_norm,
            split_indices.val,
            input_window=input_window,
            horizon=horizon,
            output_horizon=output_horizon,
        ),
        "test": MediatorWindowDataset(
            arrays_norm,
            mediator_norm,
            split_indices.test,
            input_window=input_window,
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
        "input_window": input_window,
        "horizon": horizon,
        "output_horizon": output_horizon,
        "input_dims": {key: arrays_norm[key].shape[-1] for key in INPUT_GROUPS},
        "input_variables": input_variables,
        "excluded_input_vars_by_group": excluded_by_group,
        "mediator_targets": mediator_targets,
        "mediator_target_units": [ds[name].attrs.get("units", "") for name in mediator_targets],
        "finite_point_filter": finite_point_filter,
        "static_land_features": data_cfg.get("static_land_features", {"enabled": False}),
        "external_et_targets": data_cfg.get("external_et_targets", {"enabled": False}),
    }
    return MediatorData(loaders=loaders, metadata=metadata)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in {"time_index", "point_index"}
    }


def h_sign_constraint_metrics(
    outputs: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
    device: torch.device,
    flux_scale: float,
    temp_scale: float,
) -> dict[str, torch.Tensor]:
    names = list(metadata["mediator_targets"])
    required = ("sshf_wm2", "skt", "t2m")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            "SKT-T2M-SSHF sign constraint requires mediator targets "
            f"{required}, missing {missing}."
        )
    if flux_scale <= 0 or temp_scale <= 0:
        raise ValueError("h_sign_flux_scale and h_sign_temp_scale must be positive.")

    idx = {name: names.index(name) for name in required}
    std = torch.as_tensor(
        np.asarray(metadata["normalizer"].std["main_targets"], dtype=np.float32),
        device=device,
        dtype=outputs["delta"].dtype,
    ).view(1, 1, -1)
    delta_phys = outputs["delta"] * std
    delta_sshf = delta_phys[..., idx["sshf_wm2"]]
    delta_temp_gradient = delta_phys[..., idx["skt"]] - delta_phys[..., idx["t2m"]]
    # sshf_wm2 keeps ERA5 downward-positive sign. Upward H is -sshf_wm2,
    # so a positive product below violates delta_H_up * delta(SKT-T2M) >= 0.
    sign_product = (delta_sshf / float(flux_scale)) * (delta_temp_gradient / float(temp_scale))
    violation = torch.relu(sign_product)
    return {
        "loss": violation.mean(),
        "violation_rate": (sign_product > 0).to(outputs["delta"].dtype).mean(),
        "mean_product": sign_product.mean(),
        "mean_delta_sshf_wm2": delta_sshf.mean(),
        "mean_delta_skt_minus_t2m": delta_temp_gradient.mean(),
    }


def run_epoch(
    model: CounterfactualMediatorLSTM,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    prediction_mode: str,
    metadata: dict[str, Any],
    physical_constraints: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    h_sign_weight = float(physical_constraints.get("h_sign_constraint_weight", 0.0))
    h_flux_scale = float(physical_constraints.get("h_sign_flux_scale", 10.0))
    h_temp_scale = float(physical_constraints.get("h_sign_temp_scale", 1.0))
    h_required_targets = {"sshf_wm2", "skt", "t2m"}
    has_h_sign_targets = h_required_targets.issubset(set(metadata["mediator_targets"]))
    if h_sign_weight > 0 and not has_h_sign_targets:
        missing = sorted(h_required_targets - set(metadata["mediator_targets"]))
        raise ValueError(
            "h_sign_constraint_weight > 0 requires mediator targets "
            f"{sorted(h_required_targets)}, missing {missing}."
        )
    totals = {
        "loss": 0.0,
        "base_loss": 0.0,
        "abs_loss": 0.0,
        "delta_loss": 0.0,
        "h_sign_loss": 0.0,
        "h_sign_violation_rate": 0.0,
        "h_sign_mean_product": 0.0,
        "mean_delta_sshf_wm2": 0.0,
        "mean_delta_skt_minus_t2m": 0.0,
        "n": 0,
    }
    for batch_idx, batch_raw in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch(batch_raw, device)
        target_abs = batch["target_abs"]
        target_delta = batch["target_delta"]
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            abs_loss = criterion(outputs["prediction"], target_abs)
            delta_loss = criterion(outputs["delta"], target_delta)
            base_loss = abs_loss if prediction_mode == "absolute" else delta_loss
            if has_h_sign_targets:
                h_metrics = h_sign_constraint_metrics(
                    outputs,
                    metadata=metadata,
                    device=device,
                    flux_scale=h_flux_scale,
                    temp_scale=h_temp_scale,
                )
            else:
                zero = outputs["delta"].new_tensor(0.0)
                h_metrics = {
                    "loss": zero,
                    "violation_rate": zero,
                    "mean_product": zero,
                    "mean_delta_sshf_wm2": zero,
                    "mean_delta_skt_minus_t2m": zero,
                }
            h_sign_loss = h_metrics["loss"] if h_sign_weight > 0 else outputs["delta"].new_tensor(0.0)
            loss = base_loss + h_sign_weight * h_sign_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

        n = int(target_abs.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["base_loss"] += float(base_loss.detach().cpu()) * n
        totals["abs_loss"] += float(abs_loss.detach().cpu()) * n
        totals["delta_loss"] += float(delta_loss.detach().cpu()) * n
        totals["h_sign_loss"] += float(h_metrics["loss"].detach().cpu()) * n
        totals["h_sign_violation_rate"] += float(h_metrics["violation_rate"].detach().cpu()) * n
        totals["h_sign_mean_product"] += float(h_metrics["mean_product"].detach().cpu()) * n
        totals["mean_delta_sshf_wm2"] += float(h_metrics["mean_delta_sshf_wm2"].detach().cpu()) * n
        totals["mean_delta_skt_minus_t2m"] += float(h_metrics["mean_delta_skt_minus_t2m"].detach().cpu()) * n
        totals["n"] += n
    if totals["n"] == 0:
        raise RuntimeError("No batches were processed.")
    return {key: value / totals["n"] for key, value in totals.items() if key != "n"}


def denormalize_targets(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    normalizer: Normalizer = metadata["normalizer"]
    mean = np.asarray(normalizer.mean["main_targets"], dtype=np.float32)
    std = np.asarray(normalizer.std["main_targets"], dtype=np.float32)
    return values * std.reshape((1,) * (values.ndim - 1) + (-1,)) + mean.reshape((1,) * (values.ndim - 1) + (-1,))


def delta_to_physical(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    normalizer: Normalizer = metadata["normalizer"]
    std = np.asarray(normalizer.std["main_targets"], dtype=np.float32)
    return values * std.reshape((1,) * (values.ndim - 1) + (-1,))


def h_sign_summary_from_physical_delta(
    delta_phys: np.ndarray,
    *,
    names: list[str],
    flux_scale: float,
    temp_scale: float,
    label: str,
) -> dict[str, Any] | None:
    required = ("sshf_wm2", "skt", "t2m")
    if any(name not in names for name in required):
        return None
    idx = {name: names.index(name) for name in required}
    delta_sshf = delta_phys[..., idx["sshf_wm2"]]
    delta_temp_gradient = delta_phys[..., idx["skt"]] - delta_phys[..., idx["t2m"]]
    product = (delta_sshf / float(flux_scale)) * (delta_temp_gradient / float(temp_scale))
    product = product[np.isfinite(product)]
    if product.size == 0:
        return {
            "label": label,
            "n": 0,
            "h_sign_loss": np.nan,
            "h_sign_violation_rate": np.nan,
            "h_sign_mean_product": np.nan,
            "mean_delta_sshf_wm2": np.nan,
            "mean_delta_skt_minus_t2m": np.nan,
        }
    return {
        "label": label,
        "n": int(product.size),
        "h_sign_loss": float(np.maximum(product, 0.0).mean()),
        "h_sign_violation_rate": float((product > 0.0).mean()),
        "h_sign_mean_product": float(product.mean()),
        "mean_delta_sshf_wm2": float(np.nanmean(delta_sshf)),
        "mean_delta_skt_minus_t2m": float(np.nanmean(delta_temp_gradient)),
    }


def evaluate_model(
    model: CounterfactualMediatorLSTM,
    loader: DataLoader,
    *,
    device: torch.device,
    metadata: dict[str, Any],
    physical_constraints: dict[str, Any],
    split: str,
    output_dir: Path,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    pred_abs_parts: list[np.ndarray] = []
    target_abs_parts: list[np.ndarray] = []
    pred_delta_parts: list[np.ndarray] = []
    target_delta_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch_raw in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_batch(batch_raw, device)
            outputs = model(batch)
            pred_abs_parts.append(outputs["prediction"].detach().cpu().numpy().astype(np.float32))
            target_abs_parts.append(batch["target_abs"].detach().cpu().numpy().astype(np.float32))
            pred_delta_parts.append(outputs["delta"].detach().cpu().numpy().astype(np.float32))
            target_delta_parts.append(batch["target_delta"].detach().cpu().numpy().astype(np.float32))

    if not pred_abs_parts:
        raise RuntimeError(f"No {split} batches were evaluated.")
    pred_abs_norm = np.concatenate(pred_abs_parts, axis=0)
    target_abs_norm = np.concatenate(target_abs_parts, axis=0)
    pred_delta_norm = np.concatenate(pred_delta_parts, axis=0)
    target_delta_norm = np.concatenate(target_delta_parts, axis=0)

    pred_abs_phys = denormalize_targets(pred_abs_norm, metadata)
    target_abs_phys = denormalize_targets(target_abs_norm, metadata)
    pred_delta_phys = delta_to_physical(pred_delta_norm, metadata)
    target_delta_phys = delta_to_physical(target_delta_norm, metadata)
    residual_abs_phys = target_abs_phys - target_delta_phys

    names = list(metadata["mediator_targets"])
    rows_abs = []
    rows_delta = []
    rows_persistence = []
    for lead_idx in range(pred_abs_phys.shape[1]):
        lead = lead_idx + 1
        abs_df = regression_metrics(target_abs_phys[:, lead_idx, :], pred_abs_phys[:, lead_idx, :], names)
        abs_df.insert(0, "lead", lead)
        rows_abs.append(abs_df)

        delta_df = regression_metrics(target_delta_phys[:, lead_idx, :], pred_delta_phys[:, lead_idx, :], names)
        delta_df.insert(0, "lead", lead)
        rows_delta.append(delta_df)

        persistence_df = regression_metrics(
            target_abs_phys[:, lead_idx, :],
            residual_abs_phys[:, lead_idx, :],
            names,
        )
        persistence_df.insert(0, "lead", lead)
        rows_persistence.append(persistence_df)

    abs_by_lead = pd.concat(rows_abs, ignore_index=True)
    delta_by_lead = pd.concat(rows_delta, ignore_index=True)
    persistence_by_lead = pd.concat(rows_persistence, ignore_index=True)
    abs_all = regression_metrics(
        target_abs_phys.reshape(-1, target_abs_phys.shape[-1]),
        pred_abs_phys.reshape(-1, pred_abs_phys.shape[-1]),
        names,
    )
    delta_all = regression_metrics(
        target_delta_phys.reshape(-1, target_delta_phys.shape[-1]),
        pred_delta_phys.reshape(-1, pred_delta_phys.shape[-1]),
        names,
    )
    persistence_all = regression_metrics(
        target_abs_phys.reshape(-1, target_abs_phys.shape[-1]),
        residual_abs_phys.reshape(-1, residual_abs_phys.shape[-1]),
        names,
    )
    h_sign_summaries = [
        h_sign_summary_from_physical_delta(
            pred_delta_phys,
            names=names,
            flux_scale=float(physical_constraints.get("h_sign_flux_scale", 10.0)),
            temp_scale=float(physical_constraints.get("h_sign_temp_scale", 1.0)),
            label="pred_delta",
        ),
        h_sign_summary_from_physical_delta(
            target_delta_phys,
            names=names,
            flux_scale=float(physical_constraints.get("h_sign_flux_scale", 10.0)),
            temp_scale=float(physical_constraints.get("h_sign_temp_scale", 1.0)),
            label="target_delta",
        ),
    ]
    h_sign_summaries = [row for row in h_sign_summaries if row is not None]
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_by_lead.to_csv(output_dir / f"metrics_abs_{split}_by_lead.csv", index=False)
    delta_by_lead.to_csv(output_dir / f"metrics_delta_{split}_by_lead.csv", index=False)
    persistence_by_lead.to_csv(output_dir / f"metrics_persistence_{split}_by_lead.csv", index=False)
    abs_all.to_csv(output_dir / f"metrics_abs_{split}_all_leads.csv", index=False)
    delta_all.to_csv(output_dir / f"metrics_delta_{split}_all_leads.csv", index=False)
    persistence_all.to_csv(output_dir / f"metrics_persistence_{split}_all_leads.csv", index=False)
    if h_sign_summaries:
        pd.DataFrame(h_sign_summaries).to_csv(
            output_dir / f"metrics_h_sign_constraint_{split}.csv",
            index=False,
        )
    np.savez_compressed(
        output_dir / f"predictions_{split}.npz",
        pred_abs_norm=pred_abs_norm,
        target_abs_norm=target_abs_norm,
        pred_delta_norm=pred_delta_norm,
        target_delta_norm=target_delta_norm,
        pred_abs_phys=pred_abs_phys,
        target_abs_phys=target_abs_phys,
        pred_delta_phys=pred_delta_phys,
        target_delta_phys=target_delta_phys,
    )
    return {
        "abs_all": abs_all.to_dict(orient="records"),
        "delta_all": delta_all.to_dict(orient="records"),
        "persistence_all": persistence_all.to_dict(orient="records"),
        "h_sign_constraint": h_sign_summaries,
    }


def plot_loss_curve(log_path: Path, output_dir: Path) -> Path | None:
    if not log_path.exists():
        return None
    rows = list(csv.DictReader(open(log_path, "r", encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows if row.get("val_loss", "")]
    train = [float(row["train_loss"]) for row in rows if row.get("val_loss", "")]
    val = [float(row["val_loss"]) for row in rows if row.get("val_loss", "")]
    if not epochs:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "loss_curve_plot_error.txt").write_text(str(exc), encoding="utf-8")
        return None
    path = output_dir / "train_val_loss.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, train, marker="o", label="train loss")
    ax.plot(epochs, val, marker="s", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Counterfactual Mediator LSTM Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def train_one_model(
    *,
    prediction_mode: str,
    config: dict[str, Any],
    data: MediatorData,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = data.metadata
    output_dir = Path(config["data"]["output_dir"]) / prediction_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    model = CounterfactualMediatorLSTM.from_config(
        config,
        input_dims=metadata["input_dims"],
        mediator_dim=len(metadata["mediator_targets"]),
        prediction_mode=prediction_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
    epochs = int(config["training"]["epochs"])
    val_interval = int(config["training"].get("val_interval", 5))
    patience = int(config["training"].get("patience", 5))
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    checkpoint_path = output_dir / "best_model.pt"
    log_path = output_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()

    print("=" * 100)
    print(f"Training counterfactual mediator LSTM: mode={prediction_mode}")
    print("Mediator targets:", metadata["mediator_targets"])
    print("Input dims:", metadata["input_dims"])
    print("Excluded input vars:", metadata["excluded_input_vars_by_group"])
    print("Physical constraints:", config.get("physical_constraints", {}))
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            data.loaders["train"],
            device=device,
            criterion=criterion,
            prediction_mode=prediction_mode,
            metadata=metadata,
            physical_constraints=config["physical_constraints"],
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            max_batches=args.max_train_batches,
        )
        val_metrics = None
        if epoch % val_interval == 0 or epoch == epochs:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    data.loaders["val"],
                    device=device,
                    criterion=criterion,
                    prediction_mode=prediction_mode,
                    metadata=metadata,
                    physical_constraints=config["physical_constraints"],
                    optimizer=None,
                    max_batches=args.max_eval_batches,
                )
            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = float(val_metrics["loss"])
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "prediction_mode": prediction_mode,
                        "config": config,
                        "input_dims": metadata["input_dims"],
                        "input_variables": metadata["input_variables"],
                        "excluded_input_vars_by_group": metadata["excluded_input_vars_by_group"],
                        "mediator_targets": metadata["mediator_targets"],
                        "mediator_target_units": metadata["mediator_target_units"],
                        "mediator_writeback_rules": {
                            name: MEDIATOR_WRITEBACK_RULES[name]
                            for name in metadata["mediator_targets"]
                            if name in MEDIATOR_WRITEBACK_RULES
                        },
                        "physical_constraints": config.get("physical_constraints", {}),
                        "normalizer_mean": metadata["normalizer"].mean,
                        "normalizer_std": metadata["normalizer"].std,
                        "input_window": metadata["input_window"],
                        "horizon": metadata["horizon"],
                        "output_horizon": metadata["output_horizon"],
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"train_base={train_metrics['base_loss']:.6f} "
                f"train_abs={train_metrics['abs_loss']:.6f} "
                f"train_delta={train_metrics['delta_loss']:.6f} "
                f"train_h_sign={train_metrics['h_sign_loss']:.6f} "
                f"train_h_viol={train_metrics['h_sign_violation_rate']:.4f} "
                f"val={val_metrics['loss']:.6f} "
                f"val_base={val_metrics['base_loss']:.6f} "
                f"val_abs={val_metrics['abs_loss']:.6f} "
                f"val_delta={val_metrics['delta_loss']:.6f} "
                f"val_h_sign={val_metrics['h_sign_loss']:.6f} "
                f"val_h_viol={val_metrics['h_sign_violation_rate']:.4f} "
                f"best={best_val:.6f} best_epoch={best_epoch} stale={stale} "
                f"time={time.perf_counter() - epoch_start:.2f}s"
            )
            append_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_base_loss": train_metrics["base_loss"],
                    "train_abs_loss": train_metrics["abs_loss"],
                    "train_delta_loss": train_metrics["delta_loss"],
                    "train_h_sign_loss": train_metrics["h_sign_loss"],
                    "train_h_sign_violation_rate": train_metrics["h_sign_violation_rate"],
                    "train_h_sign_mean_product": train_metrics["h_sign_mean_product"],
                    "train_mean_delta_sshf_wm2": train_metrics["mean_delta_sshf_wm2"],
                    "train_mean_delta_skt_minus_t2m": train_metrics["mean_delta_skt_minus_t2m"],
                    "val_loss": val_metrics["loss"],
                    "val_base_loss": val_metrics["base_loss"],
                    "val_abs_loss": val_metrics["abs_loss"],
                    "val_delta_loss": val_metrics["delta_loss"],
                    "val_h_sign_loss": val_metrics["h_sign_loss"],
                    "val_h_sign_violation_rate": val_metrics["h_sign_violation_rate"],
                    "val_h_sign_mean_product": val_metrics["h_sign_mean_product"],
                    "val_mean_delta_sshf_wm2": val_metrics["mean_delta_sshf_wm2"],
                    "val_mean_delta_skt_minus_t2m": val_metrics["mean_delta_skt_minus_t2m"],
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "stale_val_checks": stale,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )
            if stale >= patience:
                print(
                    f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                    f"val={best_val:.6f}"
                )
                break
        else:
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"train_base={train_metrics['base_loss']:.6f} "
                f"train_h_sign={train_metrics['h_sign_loss']:.6f} "
                f"train_h_viol={train_metrics['h_sign_violation_rate']:.4f} "
                f"val=skipped time={time.perf_counter() - epoch_start:.2f}s"
            )
            append_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_base_loss": train_metrics["base_loss"],
                    "train_abs_loss": train_metrics["abs_loss"],
                    "train_delta_loss": train_metrics["delta_loss"],
                    "train_h_sign_loss": train_metrics["h_sign_loss"],
                    "train_h_sign_violation_rate": train_metrics["h_sign_violation_rate"],
                    "train_h_sign_mean_product": train_metrics["h_sign_mean_product"],
                    "train_mean_delta_sshf_wm2": train_metrics["mean_delta_sshf_wm2"],
                    "train_mean_delta_skt_minus_t2m": train_metrics["mean_delta_skt_minus_t2m"],
                    "val_loss": "",
                    "val_base_loss": "",
                    "val_abs_loss": "",
                    "val_delta_loss": "",
                    "val_h_sign_loss": "",
                    "val_h_sign_violation_rate": "",
                    "val_h_sign_mean_product": "",
                    "val_mean_delta_sshf_wm2": "",
                    "val_mean_delta_skt_minus_t2m": "",
                    "best_val_loss": "" if best_epoch < 0 else best_val,
                    "best_epoch": "" if best_epoch < 0 else best_epoch,
                    "stale_val_checks": stale,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )

    if best_epoch < 0:
        raise RuntimeError(f"No checkpoint was saved for mode={prediction_mode}.")
    plot_path = plot_loss_curve(log_path, output_dir)
    eval_summary = None
    if not args.skip_test_eval:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        eval_summary = evaluate_model(
            model,
            data.loaders["test"],
            device=device,
            metadata=metadata,
            physical_constraints=config["physical_constraints"],
            split="test",
            output_dir=output_dir,
            max_batches=args.max_eval_batches,
        )
        save_json(output_dir / "evaluation_summary_test.json", eval_summary)

    summary = {
        "prediction_mode": prediction_mode,
        "best_checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_ran": epoch,
        "total_seconds": time.perf_counter() - start_time,
        "loss_curve_path": str(plot_path) if plot_path is not None else None,
        "physical_constraints": config.get("physical_constraints", {}),
        "test_evaluation": eval_summary,
    }
    save_json(output_dir / "training_summary.json", summary)
    print(f"Best checkpoint ({prediction_mode}): {checkpoint_path}")
    return summary


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    config = apply_overrides(load_yaml_config(args.config), args)
    mediator_targets = list(dict.fromkeys(args.mediator_targets))
    default_exclude = expand_default_exclusions(mediator_targets)
    exclude_input_vars = list(dict.fromkeys(args.exclude_input_vars or default_exclude))
    config["mediator_setup"] = {
        "mediator_targets": mediator_targets,
        "main_targets_excluded": list(DEFAULT_MAIN_TARGETS),
        "exclude_input_vars": exclude_input_vars,
        "mediator_writeback_rules": {
            name: MEDIATOR_WRITEBACK_RULES[name]
            for name in mediator_targets
            if name in MEDIATOR_WRITEBACK_RULES
        },
        "target_definition": (
            "absolute mode trains on normalized mediator value; "
            "delta mode trains on normalized value minus normalized t-horizon value. "
            "Both modes pass the t-horizon mediator value directly to the final head as a shortcut. "
            "Predictions are inverse-normalized before any NetCDF write-back."
        ),
    }
    output_root = Path(config["data"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "resolved_config.json", config)

    seed = int(config["data"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(config)
    print(f"Using device: {device}")
    print("Preparing mediator dataloaders...")
    prep_start = time.perf_counter()
    data = prepare_mediator_dataloaders(config, mediator_targets)
    print(
        "Dataset sizes:",
        {name: len(loader.dataset) for name, loader in data.loaders.items()},
        "input_dims:",
        data.metadata["input_dims"],
    )
    print(f"Data preparation time: {time.perf_counter() - prep_start:.2f}s")
    print("Finite point filter:", data.metadata["finite_point_filter"])
    save_json(
        output_root / "mediator_metadata.json",
        {
            key: value
            for key, value in data.metadata.items()
            if key not in {"normalizer", "split_indices", "split_bounds"}
        }
        | {
            "normalizer_mean": data.metadata["normalizer"].mean,
            "normalizer_std": data.metadata["normalizer"].std,
        },
    )

    summaries = {}
    for prediction_mode in args.model_types:
        summaries[prediction_mode] = train_one_model(
            prediction_mode=prediction_mode,
            config=config,
            data=data,
            device=device,
            args=args,
        )
    save_json(output_root / "training_summary_all_models.json", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
