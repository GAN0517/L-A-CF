from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from data_module import Normalizer, prepare_dataloaders
    from metrics import regression_metrics
    from model_arse_mlp import ARSEMLPCoupledModel
else:
    from .data_module import Normalizer, prepare_dataloaders
    from .metrics import regression_metrics
    from .model_arse_mlp import ARSEMLPCoupledModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ARSE MLP model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help=(
            "One or more forecast horizons to evaluate. "
            "For diagnostics, use '--horizons 1 2 3 4 5 6 7'."
        ),
    )
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


def target_input_locations(config: dict, target_names: list[str]) -> dict[str, tuple[str, int] | None]:
    locations: dict[str, tuple[str, int] | None] = {}
    for target in target_names:
        found = None
        for group in ("atm_total", "land_total", "atm_anom", "land_anom"):
            variables = list(config["variables"].get(group, []))
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
) -> np.ndarray:
    locations = target_input_locations(config, target_names)
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
    if persistence.ndim == 3:
        if persistence.shape[1] != int(output_horizon):
            raise ValueError(
                "Persistence lead dimension does not match output_horizon: "
                f"pred={persistence.shape}, output_horizon={output_horizon}."
            )
        return persistence
    if persistence.ndim != 2:
        raise ValueError(f"Expected persistence as [sample, variable] or [sample, lead, variable], got {persistence.shape}.")
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
    if y_true.ndim == 3 and y_pred.ndim == 2 and y_true.shape[1] == 1:
        y_pred = y_pred[:, None, :]
    elif y_true.ndim == 2 and y_pred.ndim == 3 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, :]

    if y_true.ndim == 2:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Prediction/target shape mismatch: pred={y_pred.shape}, target={y_true.shape}.")
        metrics = attach_units(regression_metrics(y_true, y_pred, variable_names), units)
        metrics.to_csv(output_dir / f"{prefix}_{suffix}.csv", index=False)
        return metrics, []
    if y_true.ndim != 3:
        raise ValueError(f"Expected metrics arrays as [sample, variable] or [sample, lead, variable], got {y_true.shape}.")
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


def summarize_gate_weights(
    gate_weights: np.ndarray,
    *,
    head_groups: list[list[str]],
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    for head_idx, targets in enumerate(head_groups):
        for label_idx, label in enumerate(labels):
            values = gate_weights[:, head_idx, label_idx].astype(np.float64)
            rows.append(
                {
                    "head_index": head_idx,
                    "head_targets": "+".join(targets),
                    "gate_component": label,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "p25": float(np.percentile(values, 25)),
                    "median": float(np.median(values)),
                    "p75": float(np.percentile(values, 75)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def validate_unit_convention(checkpoint: dict, metadata: dict) -> None:
    expected = metadata.get("unit_convention")
    actual = checkpoint.get("unit_convention")
    if actual != expected:
        raise RuntimeError(
            "Checkpoint unit convention does not match the current data module. "
            f"checkpoint={actual!r}, current={expected!r}. Retrain the model with the current code."
        )


def evaluate_one_horizon(
    *,
    checkpoint: dict,
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
    normalizer = Normalizer(
        mean=checkpoint["normalizer_mean"],
        std=checkpoint["normalizer_std"],
    )
    loaders, metadata = prepare_dataloaders(config, normalizer_override=normalizer)
    validate_unit_convention(checkpoint, metadata)
    model = ARSEMLPCoupledModel.from_config(config, metadata["input_dims"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    main_pred_norm = []
    main_true_norm = []
    main_persistence = []
    aux_pred_norm = []
    aux_true_norm = []
    aux_persistence = []
    main_gate_weights = []
    shared_gate_weights = []
    aux_gate_weights = []
    land_decoder_fusion_weights = []
    forcing_decoder_fusion_weights = []
    time_indices = []
    point_indices = []
    split_labels = []

    main_locations = target_input_locations(config, checkpoint["main_targets"])
    aux_locations = target_input_locations(config, checkpoint["aux_targets"])
    eval_splits = ["train", "val", "test"] if split == "all" else [split]
    dataset_size = int(sum(len(loaders[name].dataset) for name in eval_splits))
    print("=" * 100)
    print(f"Evaluating split={split}, horizon={horizon}")
    print("Dataset size:", dataset_size)
    print("Included splits:", {name: len(loaders[name].dataset) for name in eval_splits})
    print("Main target input locations:", main_locations)
    print("Aux target input locations:", aux_locations)

    inference_start = time.perf_counter()
    with torch.no_grad():
        batch_counter = 0
        for split_name in eval_splits:
            for raw_batch in loaders[split_name]:
                if max_batches is not None and batch_counter >= int(max_batches):
                    break
                batch_counter += 1
                batch = move_batch(raw_batch, device)
                outputs = model(batch)
                main_pred_norm.append(outputs["main"].cpu().numpy())
                aux_pred_norm.append(outputs["aux"].cpu().numpy())
                if outputs.get("main_gate_weights") is not None:
                    main_gate_weights.append(outputs["main_gate_weights"].cpu().numpy())
                if outputs.get("shared_gate_weights") is not None:
                    shared_gate_weights.append(outputs["shared_gate_weights"].cpu().numpy())
                if outputs.get("aux_gate_weights") is not None:
                    aux_gate_weights.append(outputs["aux_gate_weights"].cpu().numpy())
                if outputs.get("land_decoder_fusion_weights") is not None:
                    land_decoder_fusion_weights.append(outputs["land_decoder_fusion_weights"].cpu().numpy())
                if outputs.get("forcing_decoder_fusion_weights") is not None:
                    forcing_decoder_fusion_weights.append(outputs["forcing_decoder_fusion_weights"].cpu().numpy())
                main_true_norm.append(batch["main_target"].cpu().numpy())
                aux_true_norm.append(batch["aux_target"].cpu().numpy())
                main_persistence.append(
                    persistence_from_inputs(
                        raw_batch,
                        config=config,
                        normalizer=normalizer,
                        target_names=checkpoint["main_targets"],
                    )
                )
                aux_persistence.append(
                    persistence_from_inputs(
                        raw_batch,
                        config=config,
                        normalizer=normalizer,
                        target_names=checkpoint["aux_targets"],
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

    main_pred_norm_arr = np.concatenate(main_pred_norm, axis=0)
    main_true_norm_arr = np.concatenate(main_true_norm, axis=0)
    aux_pred_norm_arr = np.concatenate(aux_pred_norm, axis=0)
    aux_true_norm_arr = np.concatenate(aux_true_norm, axis=0)
    main_persistence_arr = np.concatenate(main_persistence, axis=0)
    aux_persistence_arr = np.concatenate(aux_persistence, axis=0)
    main_gate_weights_arr = (
        np.concatenate(main_gate_weights, axis=0)
        if main_gate_weights
        else None
    )
    shared_gate_weights_arr = (
        np.concatenate(shared_gate_weights, axis=0)
        if shared_gate_weights
        else None
    )
    aux_gate_weights_arr = (
        np.concatenate(aux_gate_weights, axis=0)
        if aux_gate_weights
        else None
    )
    land_decoder_fusion_weights_arr = (
        np.concatenate(land_decoder_fusion_weights, axis=0)
        if land_decoder_fusion_weights
        else None
    )
    forcing_decoder_fusion_weights_arr = (
        np.concatenate(forcing_decoder_fusion_weights, axis=0)
        if forcing_decoder_fusion_weights
        else None
    )
    time_indices_arr = np.concatenate(time_indices, axis=0)
    point_indices_arr = np.concatenate(point_indices, axis=0)
    split_labels_arr = np.concatenate(split_labels, axis=0)

    main_pred, aux_pred = normalizer.denormalize_targets(main_pred_norm_arr, aux_pred_norm_arr)
    main_true, aux_true = normalizer.denormalize_targets(main_true_norm_arr, aux_true_norm_arr)
    main_units = checkpoint.get("main_target_units", metadata.get("main_target_units", [""] * len(checkpoint["main_targets"])))
    aux_units = checkpoint.get("aux_target_units", metadata.get("aux_target_units", [""] * len(checkpoint["aux_targets"])))
    suffix = f"{split}_h{int(horizon)}"
    output_horizon = int(main_true.shape[1]) if main_true.ndim == 3 else 1
    main_persistence_arr = repeat_persistence_across_leads(
        main_persistence_arr,
        output_horizon=output_horizon,
    )
    aux_persistence_arr = repeat_persistence_across_leads(
        aux_persistence_arr,
        output_horizon=output_horizon,
    )
    main_metrics, main_metrics_by_lead = metrics_over_output_horizon(
        main_true,
        main_pred,
        checkpoint["main_targets"],
        main_units,
        output_dir=output_dir,
        prefix="metrics_main",
        suffix=suffix,
    )
    aux_metrics, aux_metrics_by_lead = metrics_over_output_horizon(
        aux_true,
        aux_pred,
        checkpoint["aux_targets"],
        aux_units,
        output_dir=output_dir,
        prefix="metrics_aux",
        suffix=suffix,
    )
    main_persistence_metrics, main_persistence_by_lead = metrics_over_output_horizon(
        main_true,
        main_persistence_arr,
        checkpoint["main_targets"],
        main_units,
        output_dir=output_dir,
        prefix="metrics_main_persistence",
        suffix=suffix,
    )
    aux_persistence_metrics, aux_persistence_by_lead = metrics_over_output_horizon(
        aux_true,
        aux_persistence_arr,
        checkpoint["aux_targets"],
        aux_units,
        output_dir=output_dir,
        prefix="metrics_aux_persistence",
        suffix=suffix,
    )
    main_head_groups = checkpoint["config"]["model"].get("main_head_groups") or [checkpoint["main_targets"]]
    if main_gate_weights_arr is not None:
        main_gate_summary = summarize_gate_weights(
            main_gate_weights_arr,
            head_groups=main_head_groups,
            labels=["land", "forcing"],
        )
        main_gate_summary.to_csv(output_dir / f"gate_main_domain_{suffix}.csv", index=False)
    else:
        main_gate_summary = None
    if shared_gate_weights_arr is not None:
        shared_gate_summary = summarize_gate_weights(
            shared_gate_weights_arr,
            head_groups=main_head_groups,
            labels=["process", "shared"],
        )
        shared_gate_summary.to_csv(output_dir / f"gate_shared_{suffix}.csv", index=False)
    else:
        shared_gate_summary = None
    if aux_gate_weights_arr is not None:
        aux_gate_summary = pd.DataFrame(
            [
                {
                    "gate_component": label,
                    "mean": float(np.mean(aux_gate_weights_arr[:, idx])),
                    "std": float(np.std(aux_gate_weights_arr[:, idx])),
                    "min": float(np.min(aux_gate_weights_arr[:, idx])),
                    "p25": float(np.percentile(aux_gate_weights_arr[:, idx], 25)),
                    "median": float(np.median(aux_gate_weights_arr[:, idx])),
                    "p75": float(np.percentile(aux_gate_weights_arr[:, idx], 75)),
                    "max": float(np.max(aux_gate_weights_arr[:, idx])),
                }
                for idx, label in enumerate(["land", "forcing"])
            ]
        )
        aux_gate_summary.to_csv(output_dir / f"gate_aux_domain_{suffix}.csv", index=False)
    else:
        aux_gate_summary = None
    decoder_fusion_summaries = []
    for domain_name, gate_arr in [
        ("land", land_decoder_fusion_weights_arr),
        ("forcing", forcing_decoder_fusion_weights_arr),
    ]:
        if gate_arr is None:
            continue
        decoder_fusion_summaries.append(
            pd.DataFrame(
                [
                    {
                        "domain": domain_name,
                        "gate_component": label,
                        "mean": float(np.mean(gate_arr[:, idx])),
                        "std": float(np.std(gate_arr[:, idx])),
                        "min": float(np.min(gate_arr[:, idx])),
                        "p25": float(np.percentile(gate_arr[:, idx], 25)),
                        "median": float(np.median(gate_arr[:, idx])),
                        "p75": float(np.percentile(gate_arr[:, idx], 75)),
                        "max": float(np.max(gate_arr[:, idx])),
                    }
                    for idx, label in enumerate(["model_decoder", "raw_gru_decoder"])
                ]
            )
        )
    decoder_fusion_summary = (
        pd.concat(decoder_fusion_summaries, ignore_index=True)
        if decoder_fusion_summaries
        else None
    )
    if decoder_fusion_summary is not None:
        decoder_fusion_summary.to_csv(output_dir / f"gate_decoder_fusion_{suffix}.csv", index=False)

    arrays_to_save = {
        "main_pred": main_pred,
        "main_true": main_true,
        "aux_pred": aux_pred,
        "aux_true": aux_true,
        "main_persistence": main_persistence_arr,
        "aux_persistence": aux_persistence_arr,
        "time_indices": time_indices_arr,
        "point_indices": point_indices_arr,
        "split_labels": split_labels_arr,
        "main_targets": np.asarray(checkpoint["main_targets"]),
        "aux_targets": np.asarray(checkpoint["aux_targets"]),
        "horizon": np.asarray([int(horizon)], dtype=np.int32),
        "output_horizon": np.asarray([int(output_horizon)], dtype=np.int32),
        "lead": np.arange(1, int(output_horizon) + 1, dtype=np.int32),
    }
    if main_gate_weights_arr is not None:
        arrays_to_save["main_gate_weights"] = main_gate_weights_arr
    if shared_gate_weights_arr is not None:
        arrays_to_save["shared_gate_weights"] = shared_gate_weights_arr
    if aux_gate_weights_arr is not None:
        arrays_to_save["aux_gate_weights"] = aux_gate_weights_arr
    if land_decoder_fusion_weights_arr is not None:
        arrays_to_save["land_decoder_fusion_weights"] = land_decoder_fusion_weights_arr
    if forcing_decoder_fusion_weights_arr is not None:
        arrays_to_save["forcing_decoder_fusion_weights"] = forcing_decoder_fusion_weights_arr
    np.savez_compressed(output_dir / f"predictions_{suffix}.npz", **arrays_to_save)
    total_eval_seconds = time.perf_counter() - eval_total_start
    with open(output_dir / f"evaluation_summary_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "horizon": int(horizon),
                "output_horizon": int(output_horizon),
                "dataset_size": dataset_size,
                "included_splits": {name: len(loaders[name].dataset) for name in eval_splits},
                "main_targets": list(checkpoint["main_targets"]),
                "aux_targets": list(checkpoint["aux_targets"]),
                "main_target_input_locations": main_locations,
                "aux_target_input_locations": aux_locations,
                "inference_seconds": inference_seconds,
                "total_eval_seconds": total_eval_seconds,
                "max_batches": max_batches,
                "gate_outputs": {
                    "main_domain": main_gate_weights_arr is not None,
                    "process_shared": shared_gate_weights_arr is not None,
                    "aux_domain": aux_gate_weights_arr is not None,
                    "decoder_fusion": decoder_fusion_summary is not None,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("Main metrics")
    print(main_metrics.to_string(index=False))
    if main_metrics_by_lead:
        print("Main metrics by lead")
        print(pd.concat(main_metrics_by_lead, ignore_index=True).to_string(index=False))
    print("Main persistence baseline")
    print(main_persistence_metrics.to_string(index=False))
    if main_persistence_by_lead:
        print("Main persistence baseline by lead")
        print(pd.concat(main_persistence_by_lead, ignore_index=True).to_string(index=False))
    print("Aux metrics")
    print(aux_metrics.to_string(index=False))
    if aux_metrics_by_lead:
        print("Aux metrics by lead")
        print(pd.concat(aux_metrics_by_lead, ignore_index=True).to_string(index=False))
    print("Aux persistence baseline")
    print(aux_persistence_metrics.to_string(index=False))
    if aux_persistence_by_lead:
        print("Aux persistence baseline by lead")
        print(pd.concat(aux_persistence_by_lead, ignore_index=True).to_string(index=False))
    if main_gate_summary is not None:
        print("Main domain gate summary")
        print(main_gate_summary.to_string(index=False))
    if shared_gate_summary is not None:
        print("Process/shared gate summary")
        print(shared_gate_summary.to_string(index=False))
    if aux_gate_summary is not None:
        print("Aux domain gate summary")
        print(aux_gate_summary.to_string(index=False))
    if decoder_fusion_summary is not None:
        print("Decoder fusion gate summary")
        print(decoder_fusion_summary.to_string(index=False))
    print(f"Timing: inference={inference_seconds:.2f}s total_eval={total_eval_seconds:.2f}s")
    print(f"Wrote horizon={horizon} outputs to {output_dir}")


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    if args.data_path is not None:
        config["data"]["path"] = str(args.data_path)
    if args.train_end is not None:
        config["data"]["train_end"] = str(args.train_end)
    if args.val_end is not None:
        config["data"]["val_end"] = str(args.val_end)
    external_et = config["data"].get("external_et_targets")
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
    output_dir = args.output_dir or Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device_from_args(args))
    horizons = args.horizons or [int(config["data"].get("horizon", 1))]
    for horizon in horizons:
        evaluate_one_horizon(
            checkpoint=checkpoint,
            base_config=config,
            output_dir=output_dir,
            split=args.split,
            horizon=int(horizon),
            max_batches=args.max_batches,
            device=device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
