from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PARENT_DIR))

from data_module import Normalizer, prepare_dataloaders  # noqa: E402
from evaluate_arse_mlp import validate_unit_convention  # noqa: E402
from metrics import regression_metrics  # noqa: E402
from model_arse_mlp import ARSEMLPCoupledModel  # noqa: E402
from run_counterfactual_direct import (  # noqa: E402
    apply_scenario,
    assign_date_range_targets,
    load_yaml,
    predict_mediators,
    requested_device,
    resolve_device,
    save_dataset,
    save_json,
)
from train_mediator_lstm import MEDIATOR_WRITEBACK_RULES  # noqa: E402


DEFAULT_MAIN_TARGETS = ("swvl1", "e_pos", "evavt_pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run date-range counterfactual inference with one multi-task mediator "
            "LSTM and three V4 single-task main checkpoints. Mediators are written "
            "as original + (counterfactual mediator prediction - no-op mediator prediction)."
        )
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=CURRENT_DIR / "configs" / "aod_ceres_scenario.yaml",
    )
    parser.add_argument("--mediator-checkpoint", type=Path, required=True)
    parser.add_argument("--swvl1-checkpoint", type=Path, required=True)
    parser.add_argument("--e-pos-checkpoint", type=Path, required=True)
    parser.add_argument("--evavt-pos-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--start-date", required=True, help="Inclusive target date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive target date, YYYY-MM-DD.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-main-batches-per-split", type=int)
    parser.add_argument("--max-mediator-batches", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_nc(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return xr.open_dataset(path, engine="h5netcdf")


def safe_stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "mean_abs": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "min": float(np.min(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


DISPLAY_UNIT_RULES = {
    "AODANA": {"display_variable": "AODANA", "display_unit": "1", "scale": 1.0},
    "ceres_all_sky_sw_down": {
        "display_variable": "ceres_all_sky_sw_down",
        "display_unit": "W m-2",
        "scale": 1.0,
    },
    "r": {"display_variable": "r", "display_unit": "%", "scale": 1.0},
    "t2m": {"display_variable": "t2m", "display_unit": "K", "scale": 1.0},
    "skt": {"display_variable": "skt", "display_unit": "K", "scale": 1.0},
    "stl1": {"display_variable": "stl1", "display_unit": "K", "scale": 1.0},
    "sshf": {"display_variable": "sshf_wm2", "display_unit": "W m-2", "scale": 1.0 / 86400.0},
    "slhf": {"display_variable": "slhf_wm2", "display_unit": "W m-2", "scale": 1.0 / 86400.0},
    "sshf_wm2": {"display_variable": "sshf_wm2", "display_unit": "W m-2", "scale": 1.0},
    "slhf_wm2": {"display_variable": "slhf_wm2", "display_unit": "W m-2", "scale": 1.0},
}


def add_display_stats(row: dict[str, Any], variable: str, values: np.ndarray) -> dict[str, Any]:
    rule = DISPLAY_UNIT_RULES.get(
        variable,
        {"display_variable": variable, "display_unit": "", "scale": 1.0},
    )
    scale = float(rule["scale"])
    row["display_variable"] = str(rule["display_variable"])
    row["display_unit"] = str(rule["display_unit"])
    row["display_scale_from_stored_values"] = scale
    row.update(safe_stats(np.asarray(values) * scale))
    return row


def apply_main_overrides(config: dict[str, Any], args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["data"]["path"] = str(data_path)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    return cfg


def selected_splits(split: str) -> tuple[str, ...]:
    return ("train", "val", "test") if split == "all" else (split,)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in {"time_index", "point_index"}
    }


def encode_split_labels(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(str)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[values == "train"] = 0
    out[values == "val"] = 1
    out[values == "test"] = 2
    if np.any(out < 0):
        raise ValueError(f"Unknown split labels: {np.unique(values[out < 0]).tolist()}")
    return out


def decode_split_codes(codes: np.ndarray) -> np.ndarray:
    names = np.asarray(["train", "val", "test"])
    return names[codes.astype(np.int64)]


def structured_keys(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    split_codes = encode_split_labels(np.asarray(result["split_labels"]))
    flat_points = np.asarray(result["flat_point_indices"], dtype=np.int64)
    order = np.lexsort(
        (
            flat_points,
            np.asarray(result["time_indices"], dtype=np.int64),
            split_codes.astype(np.int64),
        )
    )
    keys = np.empty(
        order.shape[0],
        dtype=[("split", np.int8), ("time", np.int64), ("flat_point", np.int64)],
    )
    keys["split"] = split_codes[order]
    keys["time"] = np.asarray(result["time_indices"], dtype=np.int64)[order]
    keys["flat_point"] = flat_points[order]
    return order, keys


def mediator_structured_keys(time_indices: np.ndarray, flat_point_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(
        (
            np.asarray(flat_point_indices, dtype=np.int64),
            np.asarray(time_indices, dtype=np.int64),
        )
    )
    keys = np.empty(
        order.shape[0],
        dtype=[("time", np.int64), ("flat_point", np.int64)],
    )
    keys["time"] = np.asarray(time_indices, dtype=np.int64)[order]
    keys["flat_point"] = np.asarray(flat_point_indices, dtype=np.int64)[order]
    return order, keys


def align_mediator_delta(
    *,
    noop_pred_phys: np.ndarray,
    noop_time_indices: np.ndarray,
    noop_flat_point_indices: np.ndarray,
    cf_pred_phys: np.ndarray,
    cf_time_indices: np.ndarray,
    cf_flat_point_indices: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    noop_order, noop_keys = mediator_structured_keys(noop_time_indices, noop_flat_point_indices)
    cf_order, cf_keys = mediator_structured_keys(cf_time_indices, cf_flat_point_indices)
    alignment = {
        "noop_samples": int(noop_keys.shape[0]),
        "counterfactual_samples": int(cf_keys.shape[0]),
        "sample_order_identical_before_sort": bool(
            np.array_equal(noop_time_indices, cf_time_indices)
            and np.array_equal(noop_flat_point_indices, cf_flat_point_indices)
        ),
        "sample_keys_identical_after_sort": False,
        "common_samples": int(noop_keys.shape[0]),
    }
    if noop_keys.shape == cf_keys.shape and np.array_equal(noop_keys, cf_keys):
        alignment["sample_keys_identical_after_sort"] = True
    else:
        common_keys, noop_common, cf_common = np.intersect1d(
            noop_keys,
            cf_keys,
            assume_unique=True,
            return_indices=True,
        )
        if common_keys.size == 0:
            raise ValueError("No-op and counterfactual mediator predictions share no sample keys.")
        noop_order = noop_order[noop_common]
        cf_order = cf_order[cf_common]
        noop_keys = common_keys
        alignment["common_samples"] = int(common_keys.shape[0])

    delta_phys = cf_pred_phys[cf_order] - noop_pred_phys[noop_order]
    save_json(output_dir / "mediator_prediction_alignment_summary.json", alignment)
    return delta_phys, noop_keys["time"], noop_keys["flat_point"], alignment


def write_delta_mediator_predictions(
    original_ds: xr.Dataset,
    scenario_ds: xr.Dataset,
    *,
    mediator_targets: list[str],
    delta_phys: np.ndarray,
    time_indices: np.ndarray,
    flat_point_indices: np.ndarray,
) -> tuple[xr.Dataset, list[dict[str, Any]], pd.DataFrame]:
    if delta_phys.ndim != 3 or delta_phys.shape[1] != 1:
        raise ValueError(f"Expected mediator delta as [sample, 1, variable], got {delta_phys.shape}.")
    out = scenario_ds.copy(deep=True)
    n_time = int(out.sizes["time"])
    n_lat = int(out.sizes["latitude"])
    n_lon = int(out.sizes["longitude"])
    delta_values = delta_phys[:, 0, :]
    records: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for target_idx, target_name in enumerate(mediator_targets):
        if target_name not in MEDIATOR_WRITEBACK_RULES:
            raise ValueError(f"No write-back rule is defined for mediator target {target_name!r}.")
        rule = MEDIATOR_WRITEBACK_RULES[target_name]
        dataset_var = str(rule["dataset_variable"])
        factor = float(rule["scale_factor_to_dataset_units"])
        if dataset_var not in original_ds or dataset_var not in out:
            raise KeyError(f"Dataset variable {dataset_var!r} is required for mediator {target_name!r}.")

        original_da = original_ds[dataset_var].transpose("time", "latitude", "longitude")
        arr = np.asarray(original_da.values, dtype=np.float32).copy()
        flat = arr.reshape(n_time, n_lat * n_lon)
        write_delta = (delta_values[:, target_idx] * factor).astype(np.float32)
        flat[time_indices, flat_point_indices] = flat[time_indices, flat_point_indices] + write_delta
        updated = flat.reshape(n_time, n_lat, n_lon)
        attrs = dict(original_ds[dataset_var].attrs)
        attrs["counterfactual_mediator_source"] = target_name
        attrs["counterfactual_write_rule"] = "original + (counterfactual mediator prediction - no-op mediator prediction)"
        out[dataset_var] = xr.DataArray(
            updated,
            coords={
                "time": out["time"],
                "latitude": out["latitude"],
                "longitude": out["longitude"],
            },
            dims=("time", "latitude", "longitude"),
            attrs=attrs,
        )
        records.append(
            {
                "mediator_target": target_name,
                "dataset_variable": dataset_var,
                "n_written": int(time_indices.size),
                "scale_factor_to_dataset_units": factor,
                "write_rule": "original_plus_counterfactual_minus_noop",
            }
        )
        mediator_row = {
            "kind": "mediator_delta_training_units",
            "variable": target_name,
            "dataset_variable": dataset_var,
            "writeback_scale_factor_to_dataset_units": factor,
        }
        add_display_stats(mediator_row, target_name, delta_values[:, target_idx])
        stats_rows.append(mediator_row)

    out.attrs["counterfactual_mediators"] = ", ".join(mediator_targets)
    out.attrs["counterfactual_mediator_write_rule"] = (
        "original + (counterfactual mediator prediction - no-op mediator prediction)"
    )
    return out, records, pd.DataFrame(stats_rows)


def load_single_task_checkpoint(path: Path, device: torch.device, expected_target: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Main checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    targets = list(checkpoint.get("main_targets", []))
    if targets != [expected_target]:
        raise ValueError(
            f"Expected {path} to be a single-task checkpoint for {expected_target!r}, "
            f"got main_targets={targets}."
        )
    return checkpoint


def run_single_task_inference(
    *,
    checkpoint: dict[str, Any],
    data_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    normalizer = Normalizer(
        mean=checkpoint["normalizer_mean"],
        std=checkpoint["normalizer_std"],
    )
    config = apply_main_overrides(checkpoint["config"], args, data_path)
    loaders, metadata = prepare_dataloaders(config, normalizer_override=normalizer)
    times = np.asarray(metadata["times"]).astype("datetime64[D]")
    assign_date_range_targets(
        loaders,
        metadata,
        args.start_date,
        args.end_date,
        preserve_existing=False,
    )
    validate_unit_convention(checkpoint, metadata)

    model = ARSEMLPCoupledModel.from_config(config, metadata["input_dims"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    main_pred_norm_parts: list[np.ndarray] = []
    aux_pred_norm_parts: list[np.ndarray] = []
    main_true_norm_parts: list[np.ndarray] = []
    aux_true_norm_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    local_point_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []

    with torch.no_grad():
        for split_name in selected_splits(args.split):
            if len(loaders[split_name].dataset) == 0:
                continue
            for batch_idx, raw_batch in enumerate(loaders[split_name]):
                if (
                    args.max_main_batches_per_split is not None
                    and batch_idx >= int(args.max_main_batches_per_split)
                ):
                    break
                batch = move_batch(raw_batch, device)
                outputs = model(batch)
                n = int(raw_batch["time_index"].shape[0])
                main_pred_norm_parts.append(outputs["main"].detach().cpu().numpy().astype(np.float32))
                aux_pred_norm_parts.append(outputs["aux"].detach().cpu().numpy().astype(np.float32))
                main_true_norm_parts.append(raw_batch["main_target"].numpy().astype(np.float32))
                aux_true_norm_parts.append(raw_batch["aux_target"].numpy().astype(np.float32))
                time_parts.append(raw_batch["time_index"].numpy().astype(np.int64))
                local_point_parts.append(raw_batch["point_index"].numpy().astype(np.int64))
                split_parts.append(np.asarray([split_name] * n))

    if not time_parts:
        raise ValueError(
            f"No valid target dates in {args.start_date} through {args.end_date} "
            f"for split={args.split!r}."
        )
    main_pred_norm = np.concatenate(main_pred_norm_parts, axis=0)
    aux_pred_norm = np.concatenate(aux_pred_norm_parts, axis=0)
    main_true_norm = np.concatenate(main_true_norm_parts, axis=0)
    aux_true_norm = np.concatenate(aux_true_norm_parts, axis=0)
    main_pred, aux_pred = normalizer.denormalize_targets(main_pred_norm, aux_pred_norm)
    main_true, aux_true = normalizer.denormalize_targets(main_true_norm, aux_true_norm)
    time_indices = np.concatenate(time_parts, axis=0)
    local_point_indices = np.concatenate(local_point_parts, axis=0)
    split_labels = np.concatenate(split_parts, axis=0)
    expected_time_indices = np.flatnonzero(
        (times >= args.start_date) & (times <= args.end_date)
    )
    actual_time_indices = np.unique(time_indices)
    if args.split == "all" and not np.array_equal(actual_time_indices, expected_time_indices):
        missing = np.setdiff1d(expected_time_indices, actual_time_indices)
        missing_dates = times[missing].astype(str).tolist()
        raise RuntimeError(f"Incomplete continuous prediction window; missing dates: {missing_dates}")
    valid_point_indices = np.asarray(metadata["valid_point_indices"], dtype=np.int64)
    flat_point_indices = valid_point_indices[local_point_indices]

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    main_targets = list(checkpoint["main_targets"])
    aux_targets = list(checkpoint["aux_targets"])
    main_metrics = regression_metrics(
        main_true.reshape(-1, main_true.shape[-1]),
        main_pred.reshape(-1, main_pred.shape[-1]),
        main_targets,
    )
    aux_metrics = regression_metrics(
        aux_true.reshape(-1, aux_true.shape[-1]),
        aux_pred.reshape(-1, aux_pred.shape[-1]),
        aux_targets,
    )
    suffix = args.split
    main_metrics.to_csv(label_dir / f"metrics_main_{suffix}.csv", index=False)
    aux_metrics.to_csv(label_dir / f"metrics_aux_{suffix}.csv", index=False)
    np.savez_compressed(
        label_dir / f"predictions_{suffix}.npz",
        main_pred=main_pred,
        main_true=main_true,
        aux_pred=aux_pred,
        aux_true=aux_true,
        time_indices=time_indices,
        local_point_indices=local_point_indices,
        flat_point_indices=flat_point_indices,
        split_labels=split_labels,
        main_targets=np.asarray(main_targets),
        aux_targets=np.asarray(aux_targets),
        valid_point_indices=valid_point_indices,
        times=np.asarray(metadata["times"]),
        latitudes=np.asarray(metadata["latitudes"]),
        longitudes=np.asarray(metadata["longitudes"]),
    )
    return {
        "label": label,
        "main_pred": main_pred,
        "main_true": main_true,
        "aux_pred": aux_pred,
        "aux_true": aux_true,
        "time_indices": time_indices,
        "local_point_indices": local_point_indices,
        "flat_point_indices": flat_point_indices,
        "split_labels": split_labels,
        "main_targets": main_targets,
        "aux_targets": aux_targets,
        "main_metrics": main_metrics,
        "aux_metrics": aux_metrics,
        "valid_point_indices": valid_point_indices,
        "dataset_size": int(main_pred.shape[0]),
    }


def summarize_target_effect(
    *,
    target: str,
    baseline: dict[str, Any],
    counterfactual: dict[str, Any],
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    baseline_order, baseline_keys = structured_keys(baseline)
    counterfactual_order, counterfactual_keys = structured_keys(counterfactual)
    alignment = {
        "target": target,
        "baseline_samples": int(baseline_keys.shape[0]),
        "counterfactual_samples": int(counterfactual_keys.shape[0]),
        "sample_order_identical_before_sort": bool(
            np.array_equal(baseline["time_indices"], counterfactual["time_indices"])
            and np.array_equal(baseline["flat_point_indices"], counterfactual["flat_point_indices"])
            and np.array_equal(baseline["split_labels"], counterfactual["split_labels"])
        ),
        "sample_keys_identical_after_sort": False,
        "common_samples": int(baseline_keys.shape[0]),
    }
    if baseline_keys.shape == counterfactual_keys.shape and np.array_equal(baseline_keys, counterfactual_keys):
        alignment["sample_keys_identical_after_sort"] = True
    else:
        common_keys, baseline_common, counterfactual_common = np.intersect1d(
            baseline_keys,
            counterfactual_keys,
            assume_unique=True,
            return_indices=True,
        )
        if common_keys.size == 0:
            raise ValueError(f"{target}: baseline and counterfactual predictions share no sample keys.")
        baseline_order = baseline_order[baseline_common]
        counterfactual_order = counterfactual_order[counterfactual_common]
        baseline_keys = common_keys
        alignment["common_samples"] = int(common_keys.shape[0])

    delta = counterfactual["main_pred"][counterfactual_order] - baseline["main_pred"][baseline_order]
    target_dir = output_dir / "target_effects" / target
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_dir / f"effect_{split}.npz",
        counterfactual_minus_baseline=delta,
        baseline_pred=baseline["main_pred"][baseline_order],
        counterfactual_pred=counterfactual["main_pred"][counterfactual_order],
        true=baseline["main_true"][baseline_order],
        main_targets=np.asarray([target]),
        time_indices=baseline_keys["time"],
        flat_point_indices=baseline_keys["flat_point"],
        split_labels=decode_split_codes(baseline_keys["split"]),
    )
    save_json(target_dir / f"alignment_{split}.json", alignment)
    row = {
        "target": target,
        "effect": "counterfactual_minus_baseline",
        "split": split,
    }
    row.update(safe_stats(delta))
    row.update(
        {
            "baseline_samples": alignment["baseline_samples"],
            "counterfactual_samples": alignment["counterfactual_samples"],
            "common_samples": alignment["common_samples"],
            "sample_keys_identical_after_sort": alignment["sample_keys_identical_after_sort"],
        }
    )
    return row


def summarize_input_changes(
    original: xr.Dataset,
    direct: xr.Dataset,
    mediated: xr.Dataset,
    *,
    scenario_records: list[dict[str, Any]],
    mediator_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in scenario_records:
        name = str(record["variable"])
        if name in original and name in direct:
            row = {"kind": "driver", "variable": name, "source": "scenario_direct_minus_original"}
            add_display_stats(row, name, (direct[name] - original[name]).values)
            rows.append(row)
    for record in mediator_records:
        name = str(record["dataset_variable"])
        target = str(record["mediator_target"])
        if name in direct and name in mediated:
            row = {
                "kind": "mediator",
                "variable": name,
                "mediator_target": target,
                "source": "scenario_mediated_minus_direct",
                "scale_factor_to_dataset_units": float(record["scale_factor_to_dataset_units"]),
            }
            add_display_stats(row, name, (mediated[name] - direct[name]).values)
            rows.append(row)
    return pd.DataFrame(rows)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    args.start_date = np.datetime64(args.start_date, "D")
    args.end_date = np.datetime64(args.end_date, "D")
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be on or before --end-date.")
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device(args))

    checkpoint_paths = {
        "swvl1": args.swvl1_checkpoint,
        "e_pos": args.e_pos_checkpoint,
        "evavt_pos": args.evavt_pos_checkpoint,
    }
    main_checkpoints = {
        target: load_single_task_checkpoint(path, device, target)
        for target, path in checkpoint_paths.items()
    }
    mediator_checkpoint = torch.load(args.mediator_checkpoint, map_location=device, weights_only=False)
    missing_mediators = [name for name in ("r", "t2m", "skt", "sshf_wm2", "slhf_wm2") if name not in mediator_checkpoint["mediator_targets"]]
    if missing_mediators:
        raise ValueError(f"Mediator checkpoint is missing expected targets: {missing_mediators}")

    base_data_path = args.data_path or Path(main_checkpoints["swvl1"]["config"]["data"]["path"])
    scenario = load_yaml(args.scenario_config)
    scenario_direct_path = args.output_dir / "scenario_direct_input.nc"
    scenario_mediated_path = args.output_dir / "scenario_mediated_input.nc"
    if (scenario_direct_path.exists() or scenario_mediated_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{scenario_direct_path} or {scenario_mediated_path} already exists. "
            "Use --overwrite to replace them."
        )

    print(f"Using device: {device}")
    print(f"Base data: {base_data_path}")
    print(f"Scenario config: {args.scenario_config}")
    print(f"Mediator checkpoint: {args.mediator_checkpoint}")
    for target, path in checkpoint_paths.items():
        print(f"Main checkpoint {target}: {path}")

    with open_nc(base_data_path) as ds_handle:
        original_ds = ds_handle.load()
    dataset_times = np.asarray(original_ds["time"].values).astype("datetime64[D]")
    requested_time_indices = np.flatnonzero(
        (dataset_times >= args.start_date) & (dataset_times <= args.end_date)
    )
    if requested_time_indices.size == 0:
        raise ValueError("Requested prediction dates are outside the input dataset.")
    scenario_ds, scenario_records = apply_scenario(original_ds, scenario)
    save_dataset(scenario_ds, scenario_direct_path)
    print(f"Wrote direct scenario input: {scenario_direct_path}")

    noop_pred_phys, noop_time_indices, noop_flat_point_indices, noop_mediator_metadata = predict_mediators(
        checkpoint=mediator_checkpoint,
        data_path=base_data_path,
        args=args,
        device=device,
        target_date_range=(args.start_date, args.end_date),
    )
    cf_pred_phys, cf_time_indices, cf_flat_point_indices, mediator_metadata = predict_mediators(
        checkpoint=mediator_checkpoint,
        data_path=scenario_direct_path,
        args=args,
        device=device,
        target_date_range=(args.start_date, args.end_date),
    )
    for label, predicted_indices in (
        ("no-op mediators", noop_time_indices),
        ("counterfactual mediators", cf_time_indices),
    ):
        missing_indices = np.setdiff1d(requested_time_indices, np.unique(predicted_indices))
        if missing_indices.size:
            missing_dates = dataset_times[missing_indices].astype(str).tolist()
            raise RuntimeError(f"{label} are missing requested dates: {missing_dates}")
    mediator_delta_phys, mediator_time_indices, mediator_flat_point_indices, mediator_alignment = align_mediator_delta(
        noop_pred_phys=noop_pred_phys,
        noop_time_indices=noop_time_indices,
        noop_flat_point_indices=noop_flat_point_indices,
        cf_pred_phys=cf_pred_phys,
        cf_time_indices=cf_time_indices,
        cf_flat_point_indices=cf_flat_point_indices,
        output_dir=args.output_dir,
    )
    mediated_ds, mediator_records, mediator_delta_summary = write_delta_mediator_predictions(
        original_ds,
        scenario_ds,
        mediator_targets=list(mediator_checkpoint["mediator_targets"]),
        delta_phys=mediator_delta_phys,
        time_indices=mediator_time_indices,
        flat_point_indices=mediator_flat_point_indices,
    )
    save_dataset(mediated_ds, scenario_mediated_path)
    mediator_delta_summary.to_csv(args.output_dir / "mediator_delta_prediction_summary.csv", index=False)
    print(f"Wrote bias-corrected mediated scenario input: {scenario_mediated_path}")
    print("Mediator delta prediction summary")
    print(mediator_delta_summary.to_string(index=False))

    input_change_summary = summarize_input_changes(
        original_ds,
        scenario_ds,
        mediated_ds,
        scenario_records=scenario_records,
        mediator_records=mediator_records,
    )
    input_change_summary.to_csv(args.output_dir / "counterfactual_input_change_summary.csv", index=False)
    print("Counterfactual input change summary")
    print(input_change_summary.to_string(index=False))

    effect_rows: list[dict[str, Any]] = []
    for target in DEFAULT_MAIN_TARGETS:
        target_dir = args.output_dir / "single_task_predictions" / target
        baseline = run_single_task_inference(
            checkpoint=main_checkpoints[target],
            data_path=base_data_path,
            args=args,
            device=device,
            output_dir=target_dir,
            label=f"baseline_observed_{args.split}",
        )
        counterfactual = run_single_task_inference(
            checkpoint=main_checkpoints[target],
            data_path=scenario_mediated_path,
            args=args,
            device=device,
            output_dir=target_dir,
            label=f"counterfactual_mediated_{args.split}",
        )
        effect_rows.append(
            summarize_target_effect(
                target=target,
                baseline=baseline,
                counterfactual=counterfactual,
                output_dir=args.output_dir,
                split=args.split,
            )
        )
        print(f"Finished target {target}: {effect_rows[-1]}")

    effect_summary = pd.DataFrame(effect_rows)
    effect_summary.to_csv(args.output_dir / f"v4_single_task_counterfactual_effect_summary_{args.split}.csv", index=False)
    save_json(
        args.output_dir / "v4_single_task_counterfactual_run_summary.json",
        {
            "scenario_name": scenario.get("scenario_name"),
            "scenario_config": str(args.scenario_config),
            "scenario_operations": scenario_records,
            "mediator_checkpoint": str(args.mediator_checkpoint),
            "mediator_targets": list(mediator_checkpoint["mediator_targets"]),
            "mediator_prediction_mode": str(mediator_checkpoint["prediction_mode"]),
            "noop_mediator_input_dims": noop_mediator_metadata["input_dims"],
            "mediator_input_dims": mediator_metadata["input_dims"],
            "mediator_excluded_input_vars_by_group": mediator_metadata["excluded_input_vars_by_group"],
            "mediator_alignment": mediator_alignment,
            "mediator_prediction_date_coverage": {
                "expected_days": int(requested_time_indices.size),
                "noop_predicted_days": int(np.intersect1d(requested_time_indices, np.unique(noop_time_indices)).size),
                "counterfactual_predicted_days": int(
                    np.intersect1d(requested_time_indices, np.unique(cf_time_indices)).size
                ),
            },
            "main_checkpoints": {target: str(path) for target, path in checkpoint_paths.items()},
            "main_targets": list(DEFAULT_MAIN_TARGETS),
            "base_data_path": str(base_data_path),
            "scenario_direct_path": str(scenario_direct_path),
            "scenario_mediated_path": str(scenario_mediated_path),
            "prediction_date_range": {
                "start_inclusive": str(args.start_date),
                "end_inclusive": str(args.end_date),
                "expected_days": int(requested_time_indices.size),
                "predicted_days": int(
                    np.intersect1d(requested_time_indices, np.unique(baseline["time_indices"])).size
                ),
            },
            "split": args.split,
            "mediator_writeback": mediator_records,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    print("V4 single-task counterfactual effect summary")
    print(effect_summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
