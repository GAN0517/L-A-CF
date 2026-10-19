from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from data_module import Normalizer, prepare_dataloaders  # noqa: E402
from metrics import regression_metrics  # noqa: E402
from model_single_task_mlp import SingleTaskMLPModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all single-task MLP checkpoints from one training loop.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-end", type=str, help="Override config data.train_end, e.g. 2021-09-30.")
    parser.add_argument("--val-end", type=str, help="Override config data.val_end, e.g. 2021-11-15.")
    parser.add_argument(
        "--gleam-total-et-path",
        type=Path,
        help="Override data.external_et_targets.total_et path for out-of-year evaluation.",
    )
    parser.add_argument(
        "--gleam-transpiration-path",
        type=Path,
        help="Override data.external_et_targets.vegetation_transpiration path for out-of-year evaluation.",
    )
    parser.add_argument(
        "--gleam-bare-soil-path",
        type=Path,
        help="Override data.external_et_targets.bare_soil_evaporation path for out-of-year evaluation.",
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--output-horizon", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int, help="Shortcut for --device cuda:<id>.")
    return parser.parse_args()


def requested_device_from_args(args: argparse.Namespace) -> str:
    if args.device != "auto" and args.gpu_id is not None:
        raise ValueError("Use only one of --device or --gpu-id, not both.")
    if args.gpu_id is not None:
        if int(args.gpu_id) < 0:
            raise ValueError("--gpu-id must be non-negative.")
        return f"cuda:{int(args.gpu_id)}"
    return str(args.device)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in {"time_index", "point_index"}
    }


def target_input_locations(
    config: dict,
    target_names: list[str],
    exclusions_by_target: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, tuple[str, int] | None]:
    locations: dict[str, tuple[str, int] | None] = {}
    for target in target_names:
        found = None
        target_exclusions = exclusions_by_target.get(target, {}) if exclusions_by_target is not None else {}
        for group in ("atm_total", "land_total", "atm_anom", "land_anom"):
            variables = list(config["variables"].get(group, []))
            if target in set(target_exclusions.get(group, [])):
                continue
            if target in variables:
                found = (group, variables.index(target))
                break
        locations[target] = found
    return locations


def persistence_from_inputs(
    raw_batch: dict[str, torch.Tensor],
    *,
    config: dict,
    normalizer: Normalizer,
    target_names: list[str],
    exclusions_by_target: dict[str, dict[str, list[str]]] | None = None,
) -> np.ndarray:
    locations = target_input_locations(config, target_names, exclusions_by_target)
    values = []
    batch_size = int(raw_batch["main_target"].shape[0])
    for target in target_names:
        location = locations[target]
        if location is None:
            values.append(np.full((batch_size,), np.nan, dtype=np.float32))
            continue
        group, index = location
        last_norm = raw_batch[group][:, -1, index].numpy()
        last_value = last_norm * normalizer.std[group][index] + normalizer.mean[group][index]
        values.append(last_value.astype(np.float32))
    return np.stack(values, axis=-1)


def attach_units(metrics, units: list[str]):
    out = metrics.copy()
    out.insert(1, "unit", list(units))
    return out


def repeat_persistence_across_leads(
    persistence: np.ndarray,
    *,
    output_horizon: int,
) -> np.ndarray:
    if int(output_horizon) <= 1:
        return persistence
    return np.repeat(persistence[:, None, :], int(output_horizon), axis=1)


def metrics_over_output_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names: list[str],
    units: list[str],
    *,
    output_dir: Path,
    prefix: str,
    suffix: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    if y_true.ndim == 2:
        metrics = attach_units(regression_metrics(y_true, y_pred, variable_names), units)
        metrics.to_csv(output_dir / f"{prefix}_{suffix}.csv", index=False)
        return metrics, []
    if y_true.ndim != 3:
        raise ValueError(
            f"Expected metrics arrays as [sample, variable] or [sample, lead, variable], got {y_true.shape}."
        )
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction/target shape mismatch: pred={y_pred.shape}, target={y_true.shape}.")

    n_samples, n_leads, n_variables = y_true.shape
    flattened_true = y_true.reshape(n_samples * n_leads, n_variables)
    flattened_pred = y_pred.reshape(n_samples * n_leads, n_variables)
    aggregate = attach_units(regression_metrics(flattened_true, flattened_pred, variable_names), units)
    aggregate.to_csv(output_dir / f"{prefix}_{suffix}_all_leads.csv", index=False)

    per_lead = []
    for lead_idx in range(n_leads):
        lead_metrics = attach_units(
            regression_metrics(y_true[:, lead_idx, :], y_pred[:, lead_idx, :], variable_names),
            units,
        )
        lead_metrics.insert(0, "lead", lead_idx + 1)
        lead_metrics.to_csv(output_dir / f"{prefix}_{suffix}_lead{lead_idx + 1}.csv", index=False)
        per_lead.append(lead_metrics)
    pd.concat(per_lead, ignore_index=True).to_csv(output_dir / f"{prefix}_{suffix}_by_lead.csv", index=False)
    return aggregate, per_lead


def validate_unit_convention(checkpoint: dict, metadata: dict) -> None:
    expected = metadata.get("unit_convention")
    actual = checkpoint.get("unit_convention")
    if actual != expected:
        raise RuntimeError(
            "Checkpoint unit convention does not match the current data module. "
            f"checkpoint={actual!r}, current={expected!r}."
        )


def restrict_config_to_eval_targets(config: dict, target_names: list[str]) -> None:
    config["variables"]["main_targets"] = list(target_names)
    external_et = config["data"].get("external_et_targets")
    if not external_et or not external_et.get("enabled", False):
        return
    target_set = set(target_names)
    if "eb_pos" not in target_set:
        external_et.pop("bare_soil_evaporation", None)


def reduced_main_target_normalizer(checkpoints: list[dict]) -> Normalizer:
    first = checkpoints[0]
    target_indices = [int(checkpoint["target_index"]) for checkpoint in checkpoints]
    mean = copy.deepcopy(first["normalizer_mean"])
    std = copy.deepcopy(first["normalizer_std"])
    mean["main_targets"] = np.asarray(mean["main_targets"], dtype=np.float32)[target_indices]
    std["main_targets"] = np.asarray(std["main_targets"], dtype=np.float32)[target_indices]
    return Normalizer(mean=mean, std=std)


def load_checkpoints(root: Path, targets: list[str] | None) -> list[dict]:
    if targets is None:
        summary_path = root / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            target_names = [item["target"] for item in summary.get("models", [])]
            candidates = [root / target / "best_model.pt" for target in target_names]
        else:
            candidates = sorted(root.glob("*/best_model.pt"))
    else:
        candidates = [root / target / "best_model.pt" for target in targets]
    missing = [str(path) for path in candidates if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing single-task checkpoints: {missing}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in candidates]


def evaluate_one_horizon(
    *,
    checkpoints: list[dict],
    base_config: dict,
    output_dir: Path,
    split: str,
    horizon: int,
    max_batches: int | None,
    device: torch.device,
) -> None:
    eval_total_start = time.perf_counter()
    config = copy.deepcopy(base_config)
    config["data"]["horizon"] = int(horizon)
    output_horizon = int(config["data"].get("output_horizon", 1))
    target_names = [checkpoint["target_name"] for checkpoint in checkpoints]
    restrict_config_to_eval_targets(config, target_names)
    normalizer = reduced_main_target_normalizer(checkpoints)
    loaders, metadata = prepare_dataloaders(config, normalizer_override=normalizer)
    for checkpoint in checkpoints:
        validate_unit_convention(checkpoint, metadata)

    target_indices = list(range(len(checkpoints)))
    target_units = [checkpoint.get("target_unit", metadata["main_target_units"][idx]) for checkpoint, idx in zip(checkpoints, target_indices)]
    exclusions_by_target = {
        checkpoint["target_name"]: checkpoint.get("input_exclusions", {})
        for checkpoint in checkpoints
    }
    models = []
    for checkpoint in checkpoints:
        excluded_input_targets = checkpoint.get("excluded_input_targets")
        if excluded_input_targets is None and bool(checkpoint.get("exclude_target_history", False)):
            excluded_input_targets = [checkpoint.get("target_name")]
        model = SingleTaskMLPModel.from_config(
            config,
            metadata["input_dims"],
            target_name=checkpoint.get("target_name"),
            exclude_target_history=bool(checkpoint.get("exclude_target_history", False)),
            excluded_input_names=list(excluded_input_targets or []),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        models.append(model)

    pred_norm = []
    true_norm = []
    persistence = []
    time_indices = []
    point_indices = []
    split_labels = []
    eval_splits = ["train", "val", "test"] if split == "all" else [split]
    dataset_size = int(sum(len(loaders[name].dataset) for name in eval_splits))
    print("=" * 100)
    print(f"Evaluating single-task MLP loop split={split}, horizon={horizon}")
    print("Dataset size:", dataset_size)
    print("Included splits:", {name: len(loaders[name].dataset) for name in eval_splits})
    print("Target input locations:", target_input_locations(config, target_names, exclusions_by_target))
    print("Input exclusions:", exclusions_by_target)
    print("Static land features:", config["data"].get("static_land_features", {"enabled": False}))
    print("Land total variables:", config["variables"].get("land_total", []))

    inference_start = time.perf_counter()
    with torch.no_grad():
        batch_counter = 0
        for split_name in eval_splits:
            for raw_batch in loaders[split_name]:
                if max_batches is not None and batch_counter >= int(max_batches):
                    break
                batch_counter += 1
                batch = move_batch(raw_batch, device)
                outputs = [model(batch)["target"].cpu().numpy() for model in models]
                pred_norm.append(np.concatenate(outputs, axis=-1))
                main_target = raw_batch["main_target"]
                if main_target.ndim == 2:
                    true_norm.append(main_target[:, target_indices].numpy()[:, None, :])
                elif main_target.ndim == 3:
                    true_norm.append(main_target[:, :, target_indices].numpy())
                else:
                    raise ValueError(f"Unexpected main_target shape: {tuple(main_target.shape)}")
                persistence.append(
                    persistence_from_inputs(
                        raw_batch,
                        config=config,
                        normalizer=normalizer,
                        target_names=target_names,
                        exclusions_by_target=exclusions_by_target,
                    )
                )
                time_indices.append(raw_batch["time_index"].numpy())
                point_indices.append(raw_batch["point_index"].numpy())
                split_labels.append(
                    np.full(raw_batch["time_index"].shape[0], split_name, dtype="<U5")
                )
            if max_batches is not None and batch_counter >= int(max_batches):
                break
    inference_seconds = time.perf_counter() - inference_start

    pred_norm_arr = np.concatenate(pred_norm, axis=0)
    true_norm_arr = np.concatenate(true_norm, axis=0)
    persistence_arr = np.concatenate(persistence, axis=0)
    time_indices_arr = np.concatenate(time_indices, axis=0)
    point_indices_arr = np.concatenate(point_indices, axis=0)
    split_labels_arr = np.concatenate(split_labels, axis=0)

    target_mean = normalizer.mean["main_targets"][target_indices]
    target_std = normalizer.std["main_targets"][target_indices]
    pred = pred_norm_arr * target_std + target_mean
    truth = true_norm_arr * target_std + target_mean
    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred[:, 0, :]
        truth = truth[:, 0, :]
    persistence_arr = repeat_persistence_across_leads(
        persistence_arr,
        output_horizon=pred.shape[1] if pred.ndim == 3 else 1,
    )

    suffix = f"{split}_h{int(horizon)}"
    metrics, metrics_by_lead = metrics_over_output_horizon(
        truth,
        pred,
        target_names,
        target_units,
        output_dir=output_dir,
        prefix="metrics_main",
        suffix=suffix,
    )
    persistence_metrics, persistence_by_lead = metrics_over_output_horizon(
        truth,
        persistence_arr,
        target_names,
        target_units,
        output_dir=output_dir,
        prefix="metrics_main_persistence",
        suffix=suffix,
    )
    np.savez_compressed(
        output_dir / f"predictions_{suffix}.npz",
        main_pred=pred,
        main_true=truth,
        main_persistence=persistence_arr,
        time_indices=time_indices_arr,
        point_indices=point_indices_arr,
        split_labels=split_labels_arr,
        main_targets=np.asarray(target_names),
        horizon=np.asarray([int(horizon)], dtype=np.int32),
        output_horizon=np.asarray([int(output_horizon)], dtype=np.int32),
        lead=np.arange(1, int(output_horizon) + 1, dtype=np.int32),
    )
    total_seconds = time.perf_counter() - eval_total_start
    with open(output_dir / f"evaluation_summary_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "horizon": int(horizon),
                "output_horizon": int(output_horizon),
                "dataset_size": dataset_size,
                "included_splits": {name: len(loaders[name].dataset) for name in eval_splits},
                "targets": target_names,
                "input_exclusions": exclusions_by_target,
                "static_land_features": config["data"].get("static_land_features", {"enabled": False}),
                "land_total_variables": config["variables"].get("land_total", []),
                "inference_seconds": inference_seconds,
                "total_seconds": total_seconds,
                "max_batches": max_batches,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("Main metrics")
    print(metrics.to_string(index=False))
    if metrics_by_lead:
        print("Main metrics by lead")
        print(pd.concat(metrics_by_lead, ignore_index=True).to_string(index=False))
    print("Main persistence baseline")
    print(persistence_metrics.to_string(index=False))
    if persistence_by_lead:
        print("Main persistence baseline by lead")
        print(pd.concat(persistence_by_lead, ignore_index=True).to_string(index=False))
    print(f"Timing: inference={inference_seconds:.2f}s total_eval={total_seconds:.2f}s")
    print(f"Wrote horizon={horizon} outputs to {output_dir}")


def main() -> int:
    args = parse_args()
    checkpoints = load_checkpoints(args.checkpoint_root, args.targets)
    base_config = copy.deepcopy(checkpoints[0]["config"])
    if args.data_path is not None:
        base_config["data"]["path"] = str(args.data_path)
    if args.train_end is not None:
        base_config["data"]["train_end"] = str(args.train_end)
    if args.val_end is not None:
        base_config["data"]["val_end"] = str(args.val_end)
    external_et = base_config["data"].get("external_et_targets")
    if args.gleam_total_et_path is not None:
        if not external_et or not external_et.get("enabled", False):
            raise ValueError("--gleam-total-et-path requires enabled data.external_et_targets in the checkpoint config.")
        total_cfg = external_et.setdefault("total_et", {})
        total_cfg["path"] = str(args.gleam_total_et_path)
        total_cfg["path_candidates"] = [str(args.gleam_total_et_path)]
    if args.gleam_transpiration_path is not None:
        if not external_et or not external_et.get("enabled", False):
            raise ValueError(
                "--gleam-transpiration-path requires enabled data.external_et_targets in the checkpoint config."
            )
        transpiration_cfg = external_et.setdefault("vegetation_transpiration", {})
        transpiration_cfg["path"] = str(args.gleam_transpiration_path)
        transpiration_cfg["path_candidates"] = [str(args.gleam_transpiration_path)]
    if args.gleam_bare_soil_path is not None:
        if not external_et or not external_et.get("enabled", False):
            raise ValueError(
                "--gleam-bare-soil-path requires enabled data.external_et_targets in the checkpoint config."
            )
        bare_soil_cfg = external_et.setdefault("bare_soil_evaporation", {})
        bare_soil_cfg["path"] = str(args.gleam_bare_soil_path)
        bare_soil_cfg["path_candidates"] = [str(args.gleam_bare_soil_path)]
    if args.output_horizon is not None:
        base_config["data"]["output_horizon"] = int(args.output_horizon)
    output_dir = args.output_dir or args.checkpoint_root
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device_from_args(args))
    horizons = args.horizons or [int(base_config["data"].get("horizon", 1))]
    for horizon in horizons:
        evaluate_one_horizon(
            checkpoints=checkpoints,
            base_config=base_config,
            output_dir=output_dir,
            split=args.split,
            horizon=int(horizon),
            max_batches=args.max_batches,
            device=device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
