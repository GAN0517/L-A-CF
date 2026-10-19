from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import joblib
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
sys.path.append(str(COMMON_DIR))

from baseline_utils import concat_flat_features, denormalize_main, save_json, select_target  # noqa: E402
from data_module import Normalizer, prepare_dataloaders  # noqa: E402
from metrics import regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one single-task Random Forest baseline checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-path", type=Path)
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
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def restrict_config_to_eval_target(config: dict, target: str) -> None:
    config["variables"]["main_targets"] = [target]
    external_et = config["data"].get("external_et_targets")
    if not external_et or not external_et.get("enabled", False):
        return
    if target != "eb_pos":
        external_et.pop("bare_soil_evaporation", None)


def reduced_main_target_normalizer(checkpoint: dict) -> Normalizer:
    target_idx = int(checkpoint["target_index"])
    mean = copy.deepcopy(checkpoint["normalizer_mean"])
    std = copy.deepcopy(checkpoint["normalizer_std"])
    mean["main_targets"] = np.asarray(mean["main_targets"], dtype=np.float32)[[target_idx]]
    std["main_targets"] = np.asarray(std["main_targets"], dtype=np.float32)[[target_idx]]
    return Normalizer(mean=mean, std=std)


def main() -> int:
    args = parse_args()
    checkpoint = joblib.load(args.checkpoint)
    config = checkpoint["config"]
    target = checkpoint["target_name"]
    original_target_idx = int(checkpoint["target_index"])
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
    config["data"]["horizon"] = int(args.horizon)
    restrict_config_to_eval_target(config, target)
    normalizer = reduced_main_target_normalizer(checkpoint)
    loaders, _ = prepare_dataloaders(config, normalizer_override=normalizer)
    eval_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    target_idx = 0
    keep = checkpoint["input_keep_indices"]
    model = checkpoint["model"]
    pred_batches = []
    true_batches = []
    time_batches = []
    point_batches = []
    split_batches = []
    batch_counter = 0
    for split_name in eval_splits:
        for raw_batch in loaders[split_name]:
            if args.max_batches is not None and batch_counter >= int(args.max_batches):
                break
            batch_counter += 1
            x = concat_flat_features(raw_batch, keep).numpy().astype(np.float32, copy=False)
            pred_batches.append(model.predict(x).reshape(-1, 1, 1))
            true_batches.append(select_target(raw_batch["main_target"], target_idx).numpy())
            time_batches.append(raw_batch["time_index"].numpy())
            point_batches.append(raw_batch["point_index"].numpy())
            split_batches.append(np.full(raw_batch["time_index"].shape[0], split_name, dtype="<U5"))
        if args.max_batches is not None and batch_counter >= int(args.max_batches):
            break
    pred_norm = np.concatenate(pred_batches, axis=0)
    true_norm = np.concatenate(true_batches, axis=0)
    pred = denormalize_main(pred_norm, checkpoint, original_target_idx)
    true = denormalize_main(true_norm, checkpoint, original_target_idx)
    output_dir = args.output_dir or (args.checkpoint.parent / f"{args.split}_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(true.reshape(-1, 1), pred.reshape(-1, 1), [target])
    metrics.insert(1, "unit", checkpoint.get("target_unit", ""))
    metrics.to_csv(output_dir / f"metrics_main_{args.split}_h{args.horizon}.csv", index=False)
    np.savez_compressed(
        output_dir / f"predictions_{args.split}_h{args.horizon}.npz",
        pred=pred,
        true=true,
        target=np.asarray([target]),
        unit=np.asarray([checkpoint.get("target_unit", "")]),
        time_indices=np.concatenate(time_batches, axis=0),
        point_indices=np.concatenate(point_batches, axis=0),
        split_labels=np.concatenate(split_batches, axis=0),
    )
    save_json(
        output_dir / "evaluation_summary.json",
        {
            "checkpoint": str(args.checkpoint),
            "target": target,
            "split": args.split,
            "included_splits": {name: len(loaders[name].dataset) for name in eval_splits},
            "horizon": args.horizon,
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
