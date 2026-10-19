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
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PARENT_DIR))

from data_module import (  # noqa: E402
    SPATIAL_DIMS,
    TIME_DIM,
    Normalizer,
    prepare_dataloaders,
)
from evaluate_arse_mlp import validate_unit_convention  # noqa: E402
from metrics import regression_metrics  # noqa: E402
from model_arse_mlp import ARSEMLPCoupledModel  # noqa: E402
from model_mediator_lstm import CounterfactualMediatorLSTM  # noqa: E402
from train_mediator_lstm import (  # noqa: E402
    MEDIATOR_WRITEBACK_RULES,
    denormalize_targets as denormalize_mediators,
    move_batch as move_mediator_batch,
    prepare_mediator_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct, non-recursive counterfactual inference. The scenario "
            "first edits AOD/shortwave radiation, then the mediator LSTM predicts "
            "counterfactual mediator variables, and finally a V4 single-task model "
            "is evaluated on baseline, direct-scenario, and mediated-scenario inputs."
        )
    )
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--mediator-checkpoint", type=Path, required=True)
    parser.add_argument("--main-checkpoint", "--v2-checkpoint", dest="main_checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-main-batches", type=int)
    parser.add_argument("--max-mediator-batches", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def requested_device(args: argparse.Namespace) -> str:
    if args.device != "auto" and args.gpu_id is not None:
        raise ValueError("Use only one of --device or --gpu-id.")
    if args.gpu_id is not None:
        return f"cuda:{int(args.gpu_id)}"
    return str(args.device)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario config must be a mapping: {path}")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def apply_one_operation(
    values: xr.DataArray,
    operation: dict[str, Any],
    *,
    source_delta: xr.DataArray | None = None,
) -> xr.DataArray:
    mode = str(operation.get("mode", "")).lower()
    if mode == "multiply":
        out = values * float(operation["value"])
    elif mode == "add":
        out = values + float(operation["value"])
    elif mode == "set":
        out = xr.zeros_like(values) + float(operation["value"])
    elif mode == "add_scaled_delta_from":
        if source_delta is None:
            raise ValueError("mode='add_scaled_delta_from' requires a source_delta.")
        out = values + float(operation["value"]) * source_delta
    else:
        raise ValueError(
            f"Unsupported scenario mode={mode!r}; "
            "expected multiply/add/set/add_scaled_delta_from."
        )

    if operation.get("clip_min") is not None:
        out = out.clip(min=float(operation["clip_min"]))
    if operation.get("clip_max") is not None:
        out = out.clip(max=float(operation["clip_max"]))
    return out.astype(np.float32)


def apply_scenario(ds: xr.Dataset, scenario: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    out = ds.copy(deep=True)
    records: list[dict[str, Any]] = []
    variables = scenario.get("variables") or {}
    if not isinstance(variables, dict):
        raise ValueError("scenario.variables must be a mapping.")

    for name, raw_op in variables.items():
        op = dict(raw_op or {})
        if not bool(op.get("enabled", True)):
            continue
        if name not in out:
            raise KeyError(f"Scenario variable {name!r} is not present in the dataset.")
        da = out[name]
        mode = str(op.get("mode", "")).lower()
        source_name = op.get("source_variable")
        if mode == "add_scaled_delta_from":
            if not source_name:
                raise ValueError(f"Scenario variable {name!r} requires source_variable.")
            if source_name not in out or source_name not in ds:
                raise KeyError(f"source_variable {source_name!r} is not present in the dataset.")
        if TIME_DIM in da.dims and (op.get("start") or op.get("end")):
            start = op.get("start")
            end = op.get("end")
            selector = {TIME_DIM: slice(start, end)}
            edited = da.copy(deep=True)
            source_delta = None
            if mode == "add_scaled_delta_from":
                source_delta = out[str(source_name)].sel(selector) - ds[str(source_name)].sel(selector)
            edited.loc[selector] = apply_one_operation(
                da.sel(selector),
                op,
                source_delta=source_delta,
            )
            out[name] = edited.astype(np.float32)
        else:
            source_delta = None
            if mode == "add_scaled_delta_from":
                source_delta = out[str(source_name)] - ds[str(source_name)]
            out[name] = apply_one_operation(da, op, source_delta=source_delta)
        out[name].attrs.update(da.attrs)
        out[name].attrs["counterfactual_scenario"] = str(scenario.get("scenario_name", "unnamed"))
        records.append(
            {
                "variable": str(name),
                "mode": str(op.get("mode")),
                "value": op.get("value"),
                "start": op.get("start"),
                "end": op.get("end"),
                "clip_min": op.get("clip_min"),
                "clip_max": op.get("clip_max"),
                "source_variable": source_name,
            }
        )
    out.attrs["counterfactual_scenario"] = str(scenario.get("scenario_name", "unnamed"))
    return out, records


def save_dataset(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 1}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.number)
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def apply_main_overrides(config: dict[str, Any], args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["data"]["path"] = str(data_path)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    return cfg


def apply_mediator_overrides(config: dict[str, Any], args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["data"]["path"] = str(data_path)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    return cfg


def assign_date_range_targets(
    loaders: dict[str, Any],
    metadata: dict[str, Any],
    start_date: np.datetime64,
    end_date: np.datetime64,
    *,
    preserve_existing: bool,
) -> None:
    """Include calendar targets even when their input windows cross split boundaries."""
    times = np.asarray(metadata["times"]).astype("datetime64[D]")
    in_range = np.flatnonzero(
        (times >= np.datetime64(start_date, "D")) & (times <= np.datetime64(end_date, "D"))
    )
    split_bounds = metadata["split_bounds"]
    for split_name, loader in loaders.items():
        start_idx, end_idx = getattr(split_bounds, split_name)
        requested = in_range[(in_range >= start_idx) & (in_range <= end_idx)]
        current = loader.dataset.time_indices if preserve_existing else np.empty(0, dtype=np.int64)
        loader.dataset.time_indices = np.unique(
            np.concatenate((np.asarray(current, dtype=np.int64), requested))
        )


def predict_mediators(
    *,
    checkpoint: dict[str, Any],
    data_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    target_date_range: tuple[np.datetime64, np.datetime64] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mediator_targets = list(checkpoint["mediator_targets"])
    normalizer = Normalizer(
        mean=checkpoint["normalizer_mean"],
        std=checkpoint["normalizer_std"],
    )
    config = apply_mediator_overrides(checkpoint["config"], args, data_path)
    data = prepare_mediator_dataloaders(
        config,
        mediator_targets,
        normalizer_override=normalizer,
    )
    if target_date_range is not None:
        assign_date_range_targets(
            data.loaders,
            data.metadata,
            *target_date_range,
            preserve_existing=False,
        )
    model = CounterfactualMediatorLSTM.from_config(
        config,
        input_dims=data.metadata["input_dims"],
        mediator_dim=len(mediator_targets),
        prediction_mode=str(checkpoint["prediction_mode"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_norm_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            for batch_idx, raw_batch in enumerate(data.loaders[split_name]):
                if args.max_mediator_batches is not None and batch_idx >= int(args.max_mediator_batches):
                    break
                batch = move_mediator_batch(raw_batch, device)
                outputs = model(batch)
                pred_norm_parts.append(outputs["prediction"].detach().cpu().numpy().astype(np.float32))
                time_parts.append(raw_batch["time_index"].numpy())
                point_parts.append(raw_batch["point_index"].numpy())

    pred_norm = np.concatenate(pred_norm_parts, axis=0)
    pred_phys = denormalize_mediators(pred_norm, data.metadata)
    time_indices = np.concatenate(time_parts, axis=0).astype(np.int64)
    local_point_indices = np.concatenate(point_parts, axis=0).astype(np.int64)
    flat_point_indices = np.asarray(data.metadata["valid_point_indices"], dtype=np.int64)[local_point_indices]
    return pred_phys, time_indices, flat_point_indices, data.metadata


def _ensure_time_lat_lon(da: xr.DataArray, name: str) -> xr.DataArray:
    required = (TIME_DIM, *SPATIAL_DIMS)
    missing = [dim for dim in required if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} must have dimensions {required}; missing {missing}.")
    return da.transpose(*required)


def write_mediator_predictions(
    ds: xr.Dataset,
    *,
    mediator_targets: list[str],
    pred_phys: np.ndarray,
    time_indices: np.ndarray,
    flat_point_indices: np.ndarray,
) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    out = ds.copy(deep=True)
    if pred_phys.ndim != 3 or pred_phys.shape[1] != 1:
        raise ValueError(f"Expected mediator predictions as [sample, 1, variable], got {pred_phys.shape}.")
    n_time = int(out.sizes[TIME_DIM])
    n_lat = int(out.sizes[SPATIAL_DIMS[0]])
    n_lon = int(out.sizes[SPATIAL_DIMS[1]])
    write_values = pred_phys[:, 0, :]
    records: list[dict[str, Any]] = []

    for target_idx, target_name in enumerate(mediator_targets):
        if target_name not in MEDIATOR_WRITEBACK_RULES:
            raise ValueError(f"No write-back rule is defined for mediator target {target_name!r}.")
        rule = MEDIATOR_WRITEBACK_RULES[target_name]
        dataset_var = str(rule["dataset_variable"])
        factor = float(rule["scale_factor_to_dataset_units"])
        if dataset_var not in out:
            raise KeyError(f"Dataset variable {dataset_var!r} is required to write mediator {target_name!r}.")
        da = _ensure_time_lat_lon(out[dataset_var], dataset_var)
        arr = np.asarray(da.values, dtype=np.float32).copy()
        flat = arr.reshape(n_time, n_lat * n_lon)
        flat[time_indices, flat_point_indices] = (write_values[:, target_idx] * float(factor)).astype(np.float32)
        updated = flat.reshape(n_time, n_lat, n_lon)
        out[dataset_var] = xr.DataArray(
            updated,
            coords={TIME_DIM: out[TIME_DIM], SPATIAL_DIMS[0]: out[SPATIAL_DIMS[0]], SPATIAL_DIMS[1]: out[SPATIAL_DIMS[1]]},
            dims=(TIME_DIM, *SPATIAL_DIMS),
            attrs=dict(out[dataset_var].attrs),
        )
        out[dataset_var].attrs["counterfactual_mediator_source"] = target_name
        records.append(
            {
                "mediator_target": target_name,
                "dataset_variable": dataset_var,
                "n_written": int(time_indices.size),
                "scale_factor_to_dataset_units": float(factor),
            }
        )
    out.attrs["counterfactual_mediators"] = ", ".join(mediator_targets)
    return out, records


def run_main_inference(
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
    validate_unit_convention(checkpoint, metadata)
    model = ARSEMLPCoupledModel.from_config(config, metadata["input_dims"]).to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the main-task checkpoint with the base ARSEMLPCoupledModel. "
            "Use a V4 single-task checkpoint generated by train_arse_mlp.py."
        ) from exc
    model.eval()

    main_pred_norm_parts: list[np.ndarray] = []
    aux_pred_norm_parts: list[np.ndarray] = []
    main_true_norm_parts: list[np.ndarray] = []
    aux_true_norm_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loaders[args.split]):
            if args.max_main_batches is not None and batch_idx >= int(args.max_main_batches):
                break
            batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if torch.is_tensor(value) and key not in {"time_index", "point_index"}
            }
            outputs = model(batch)
            main_pred_norm_parts.append(outputs["main"].detach().cpu().numpy().astype(np.float32))
            aux_pred_norm_parts.append(outputs["aux"].detach().cpu().numpy().astype(np.float32))
            main_true_norm_parts.append(raw_batch["main_target"].numpy().astype(np.float32))
            aux_true_norm_parts.append(raw_batch["aux_target"].numpy().astype(np.float32))
            time_parts.append(raw_batch["time_index"].numpy())
            point_parts.append(raw_batch["point_index"].numpy())

    main_pred_norm = np.concatenate(main_pred_norm_parts, axis=0)
    aux_pred_norm = np.concatenate(aux_pred_norm_parts, axis=0)
    main_true_norm = np.concatenate(main_true_norm_parts, axis=0)
    aux_true_norm = np.concatenate(aux_true_norm_parts, axis=0)
    main_pred, aux_pred = normalizer.denormalize_targets(main_pred_norm, aux_pred_norm)
    main_true, aux_true = normalizer.denormalize_targets(main_true_norm, aux_true_norm)
    time_indices = np.concatenate(time_parts, axis=0).astype(np.int64)
    point_indices = np.concatenate(point_parts, axis=0).astype(np.int64)

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
    main_metrics.to_csv(label_dir / f"metrics_main_{args.split}.csv", index=False)
    aux_metrics.to_csv(label_dir / f"metrics_aux_{args.split}.csv", index=False)
    np.savez_compressed(
        label_dir / f"predictions_{args.split}.npz",
        main_pred=main_pred,
        main_true=main_true,
        aux_pred=aux_pred,
        aux_true=aux_true,
        time_indices=time_indices,
        point_indices=point_indices,
        main_targets=np.asarray(main_targets),
        aux_targets=np.asarray(aux_targets),
    )
    return {
        "label": label,
        "main_pred": main_pred,
        "main_true": main_true,
        "aux_pred": aux_pred,
        "aux_true": aux_true,
        "time_indices": time_indices,
        "point_indices": point_indices,
        "main_targets": main_targets,
        "aux_targets": aux_targets,
        "main_metrics": main_metrics,
        "aux_metrics": aux_metrics,
    }


def effect_summary(
    baseline: dict[str, Any],
    direct: dict[str, Any],
    mediated: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    if list(baseline["main_targets"]) != list(direct["main_targets"]) or list(baseline["main_targets"]) != list(mediated["main_targets"]):
        raise ValueError("Main target lists differ across inference branches.")
    for key in ("time_indices", "point_indices"):
        if not np.array_equal(baseline[key], direct[key]) or not np.array_equal(baseline[key], mediated[key]):
            raise ValueError(f"{key} differ across inference branches; cannot summarize effects safely.")

    direct_effect = direct["main_pred"] - baseline["main_pred"]
    mediated_effect = mediated["main_pred"] - baseline["main_pred"]
    mediator_extra = mediated["main_pred"] - direct["main_pred"]
    rows = []
    for idx, name in enumerate(baseline["main_targets"]):
        for effect_name, values in (
            ("direct_minus_baseline", direct_effect[..., idx]),
            ("mediated_minus_baseline", mediated_effect[..., idx]),
            ("mediated_minus_direct", mediator_extra[..., idx]),
        ):
            flat = values.reshape(-1).astype(np.float64)
            rows.append(
                {
                    "target": name,
                    "effect": effect_name,
                    "n": int(flat.size),
                    "mean": float(np.mean(flat)),
                    "std": float(np.std(flat)),
                    "mean_abs": float(np.mean(np.abs(flat))),
                    "p05": float(np.percentile(flat, 5)),
                    "p50": float(np.percentile(flat, 50)),
                    "p95": float(np.percentile(flat, 95)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "counterfactual_effect_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "counterfactual_effects.npz",
        direct_minus_baseline=direct_effect,
        mediated_minus_baseline=mediated_effect,
        mediated_minus_direct=mediator_extra,
        main_targets=np.asarray(baseline["main_targets"]),
        time_indices=baseline["time_indices"],
        point_indices=baseline["point_indices"],
    )
    return df


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device(args))

    scenario = load_yaml(args.scenario_config)
    mediator_checkpoint = torch.load(args.mediator_checkpoint, map_location=device, weights_only=False)
    main_checkpoint = torch.load(args.main_checkpoint, map_location=device, weights_only=False)
    base_data_path = args.data_path or Path(main_checkpoint["config"]["data"]["path"])

    scenario_direct_path = args.output_dir / "scenario_direct_input.nc"
    scenario_mediated_path = args.output_dir / "scenario_mediated_input.nc"
    if (scenario_direct_path.exists() or scenario_mediated_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{scenario_direct_path} or {scenario_mediated_path} already exists. "
            "Use --overwrite to replace them."
        )

    print(f"Using device: {device}")
    print(f"Base data: {base_data_path}")
    print(f"Scenario: {args.scenario_config}")
    ds_handle = xr.open_dataset(base_data_path, engine="netcdf4")
    try:
        scenario_ds, scenario_records = apply_scenario(ds_handle.load(), scenario)
    finally:
        ds_handle.close()
    save_dataset(scenario_ds, scenario_direct_path)
    print(f"Wrote direct scenario input: {scenario_direct_path}")

    pred_phys, time_indices, flat_point_indices, mediator_metadata = predict_mediators(
        checkpoint=mediator_checkpoint,
        data_path=scenario_direct_path,
        args=args,
        device=device,
    )
    mediated_ds, mediator_records = write_mediator_predictions(
        scenario_ds,
        mediator_targets=list(mediator_checkpoint["mediator_targets"]),
        pred_phys=pred_phys,
        time_indices=time_indices,
        flat_point_indices=flat_point_indices,
    )
    save_dataset(mediated_ds, scenario_mediated_path)
    print(f"Wrote mediated scenario input: {scenario_mediated_path}")

    baseline_result = run_main_inference(
        checkpoint=main_checkpoint,
        data_path=base_data_path,
        args=args,
        device=device,
        output_dir=args.output_dir,
        label="baseline_observed_input",
    )
    direct_result = run_main_inference(
        checkpoint=main_checkpoint,
        data_path=scenario_direct_path,
        args=args,
        device=device,
        output_dir=args.output_dir,
        label="scenario_direct_aod_ceres",
    )
    mediated_result = run_main_inference(
        checkpoint=main_checkpoint,
        data_path=scenario_mediated_path,
        args=args,
        device=device,
        output_dir=args.output_dir,
        label="scenario_with_lstm_mediators",
    )
    effects = effect_summary(baseline_result, direct_result, mediated_result, args.output_dir)
    save_json(
        args.output_dir / "counterfactual_run_summary.json",
        {
            "scenario_name": scenario.get("scenario_name"),
            "scenario_config": str(args.scenario_config),
            "scenario_operations": scenario_records,
            "mediator_checkpoint": str(args.mediator_checkpoint),
            "mediator_targets": list(mediator_checkpoint["mediator_targets"]),
            "mediator_prediction_mode": str(mediator_checkpoint["prediction_mode"]),
            "mediator_input_dims": mediator_metadata["input_dims"],
            "mediator_excluded_input_vars_by_group": mediator_metadata["excluded_input_vars_by_group"],
            "main_checkpoint": str(args.main_checkpoint),
            "main_targets": list(main_checkpoint["main_targets"]),
            "split": args.split,
            "base_data_path": str(base_data_path),
            "scenario_direct_path": str(scenario_direct_path),
            "scenario_mediated_path": str(scenario_mediated_path),
            "mediator_writeback": mediator_records,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )

    print("Counterfactual effect summary")
    print(effects.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")
    return 0


