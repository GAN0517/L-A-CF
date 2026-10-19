from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_counterfactual_direct import (  # noqa: E402
    apply_scenario,
    load_yaml,
    predict_mediators,
    requested_device,
    resolve_device,
    save_dataset,
    save_json,
)
from run_v4_single_task_counterfactual_full_year import (  # noqa: E402
    align_mediator_delta,
    open_nc,
    summarize_input_changes,
    write_delta_mediator_predictions,
)


EXPECTED_MEDIATORS = ("r", "t2m", "skt", "sshf_wm2", "slhf_wm2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a full-year counterfactual input NetCDF using the hsign-constrained "
            "multi-task mediator LSTM. This script only edits AOD/CERES, predicts "
            "bias-corrected mediator deltas, and writes scenario_direct_input.nc plus "
            "scenario_mediated_input.nc; it does not run any main-task model."
        )
    )
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--mediator-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-mediator-batches", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_mediator_checkpoint(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    targets = list(checkpoint.get("mediator_targets", []))
    missing = [name for name in EXPECTED_MEDIATORS if name not in targets]
    if missing:
        raise ValueError(f"Mediator checkpoint {path} is missing expected targets: {missing}")
    prediction_mode = str(checkpoint.get("prediction_mode", ""))
    physical_constraints = checkpoint.get("physical_constraints", {})
    return {
        "checkpoint": str(path),
        "mediator_targets": targets,
        "prediction_mode": prediction_mode,
        "physical_constraints": physical_constraints,
        "contains_expected_hsign_targets": not missing,
    }


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device(args))

    scenario_direct_path = args.output_dir / "scenario_direct_input.nc"
    scenario_mediated_path = args.output_dir / "scenario_mediated_input.nc"
    if (scenario_direct_path.exists() or scenario_mediated_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{scenario_direct_path} or {scenario_mediated_path} already exists. "
            "Use --overwrite to replace them."
        )

    mediator_checkpoint = torch.load(args.mediator_checkpoint, map_location=device, weights_only=False)
    mediator_validation = validate_mediator_checkpoint(mediator_checkpoint, args.mediator_checkpoint)
    base_data_path = args.data_path or Path(mediator_checkpoint["config"]["data"]["path"])
    scenario = load_yaml(args.scenario_config)

    print(f"Using device: {device}")
    print(f"Base data: {base_data_path}")
    print(f"Scenario config: {args.scenario_config}")
    print(f"Mediator checkpoint: {args.mediator_checkpoint}")
    print(f"Mediator targets: {mediator_validation['mediator_targets']}")

    with open_nc(base_data_path) as ds_handle:
        original_ds = ds_handle.load()

    scenario_ds, scenario_records = apply_scenario(original_ds, scenario)
    save_dataset(scenario_ds, scenario_direct_path)
    print(f"Wrote direct scenario input: {scenario_direct_path}")

    noop_pred_phys, noop_time_indices, noop_flat_point_indices, noop_metadata = predict_mediators(
        checkpoint=mediator_checkpoint,
        data_path=base_data_path,
        args=args,
        device=device,
    )
    cf_pred_phys, cf_time_indices, cf_flat_point_indices, cf_metadata = predict_mediators(
        checkpoint=mediator_checkpoint,
        data_path=scenario_direct_path,
        args=args,
        device=device,
    )
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

    save_json(
        args.output_dir / "hsign_multitask_mediator_scenario_input_summary.json",
        {
            "workflow": "hsign_multitask_mediator_scenario_input_only",
            "scenario_name": scenario.get("scenario_name"),
            "scenario_config": str(args.scenario_config),
            "scenario_operations": scenario_records,
            "mediator_checkpoint": str(args.mediator_checkpoint),
            "mediator_validation": mediator_validation,
            "noop_mediator_input_dims": noop_metadata["input_dims"],
            "counterfactual_mediator_input_dims": cf_metadata["input_dims"],
            "mediator_excluded_input_vars_by_group": cf_metadata["excluded_input_vars_by_group"],
            "mediator_alignment": mediator_alignment,
            "base_data_path": str(base_data_path),
            "scenario_direct_path": str(scenario_direct_path),
            "scenario_mediated_path": str(scenario_mediated_path),
            "mediator_writeback": mediator_records,
            "subtraction_rule": "mediator_delta = counterfactual_mediator_prediction - no_op_mediator_prediction",
            "write_rule": "mediated_input = original_input + mediator_delta for mediator variables; drivers from direct scenario",
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
