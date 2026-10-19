from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from data_module import load_yaml_config, prepare_dataloaders  # noqa: E402
from model_single_task_mlp import SingleTaskMLPModel  # noqa: E402


DEFAULT_HYDRO_EXCLUSION_TARGETS = ("swvl1", "e_pos", "evavt_pos", "eb_pos")
STATIC_LAND_VARIABLES = ("cvh", "cvl", "slt", "tvh", "tvl")
DEFAULT_STATIC_SOURCE_DIR = "data/static"
DEFAULT_STATIC_FALLBACK_DIRS = (DEFAULT_STATIC_SOURCE_DIR,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the three single-task MLP baselines in one command.")
    parser.add_argument("--config", type=Path, default=PARENT_DIR / "config_arse_mlp_three_head_full_rawres.yaml")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--targets", nargs="+", default=None)
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
    parser.add_argument("--gpu-id", type=int, help="Shortcut for --device cuda:<id>.")
    parser.add_argument(
        "--use-static-land-features",
        action="store_true",
        help="Append cvh/cvl/slt/tvh/tvl to land_total for this single-task MLP run only.",
    )
    parser.add_argument(
        "--static-source-dir",
        type=str,
        default=DEFAULT_STATIC_SOURCE_DIR,
        help="Directory containing cvh.nc/cvl.nc/slt.nc/tvh.nc/tvl.nc.",
    )
    parser.add_argument(
        "--keep-target-history",
        action="store_true",
        help=(
            "Keep target-history variables in the inputs if they appear in an input group. "
            "Default excludes all hydro targets: swvl1, e_pos, evavt_pos, eb_pos."
        ),
    )
    parser.add_argument(
        "--exclude-input-targets",
        nargs="+",
        default=list(DEFAULT_HYDRO_EXCLUSION_TARGETS),
        help=(
            "Target-history variables to remove from every single-task model input. "
            "Ignored when --keep-target-history is set."
        ),
    )
    return parser.parse_args()


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def static_land_feature_config(source_dir: str) -> dict[str, Any]:
    roots = unique_in_order([source_dir, *DEFAULT_STATIC_FALLBACK_DIRS])
    return {
        "enabled": True,
        "source_name": "ERA5-Land static",
        "source_dir": source_dir,
        "variables": [
            {
                "name": name,
                "var": name,
                "path_candidates": [f"{root.rstrip('/')}/{name}.nc" for root in roots],
            }
            for name in STATIC_LAND_VARIABLES
        ],
    }


def add_static_land_features_to_config(config: dict[str, Any], source_dir: str) -> None:
    config["data"]["static_land_features"] = static_land_feature_config(source_dir)
    land_total = list(config["variables"].get("land_total", []))
    config["variables"]["land_total"] = unique_in_order([*land_total, *STATIC_LAND_VARIABLES])


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.data_path is not None:
        cfg["data"]["path"] = str(args.data_path)
    if args.output_dir is not None:
        cfg["data"]["output_dir"] = str(args.output_dir)
    else:
        cfg["data"]["output_dir"] = str(PARENT_DIR.parent / "outputs" / "single_task_mlp_loop")
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.output_horizon is not None:
        cfg["data"]["output_horizon"] = int(args.output_horizon)
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    if args.val_interval is not None:
        cfg["training"]["val_interval"] = int(args.val_interval)
    if args.patience is not None:
        cfg["training"]["patience"] = int(args.patience)
    if args.seed is not None:
        cfg["data"]["seed"] = int(args.seed)
    if args.device is not None and args.gpu_id is not None:
        raise ValueError("Use only one of --device or --gpu-id, not both.")
    if args.gpu_id is not None:
        if int(args.gpu_id) < 0:
            raise ValueError("--gpu-id must be non-negative.")
        cfg["training"]["device"] = f"cuda:{int(args.gpu_id)}"
    elif args.device is not None:
        cfg["training"]["device"] = args.device
    if args.use_static_land_features:
        add_static_land_features_to_config(cfg, args.static_source_dir)
    else:
        cfg["data"]["static_land_features"] = {"enabled": False}
    cfg.setdefault("model", {})
    cfg["model"].setdefault("layers", int(cfg["model"].get("decoder_layers", 3)))
    return cfg


def select_main_target(main_target: torch.Tensor, target_index: int) -> torch.Tensor:
    if main_target.ndim == 2:
        return main_target[:, target_index : target_index + 1].unsqueeze(1)
    if main_target.ndim == 3:
        return main_target[:, :, target_index : target_index + 1]
    raise ValueError(
        "Expected main_target as [batch, variable] or [batch, lead, variable], "
        f"got {tuple(main_target.shape)}"
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in {"time_index", "point_index"}
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: SingleTaskMLPModel,
    loader,
    *,
    target_index: int,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "n": 0}

    for batch_idx, batch_raw in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch(batch_raw, device)
        target = select_main_target(batch["main_target"], target_index)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            if outputs["target"].shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: pred={tuple(outputs['target'].shape)}, "
                    f"target={tuple(target.shape)}"
                )
            loss = criterion(outputs["target"], target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

        n = int(target.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["n"] += n

    if totals["n"] == 0:
        raise RuntimeError("No batches were processed.")
    return {"loss": totals["loss"] / totals["n"]}


def train_one_target(
    *,
    target_name: str,
    target_index: int,
    config: dict[str, Any],
    loaders,
    metadata: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_dir = Path(config["data"]["output_dir"]) / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    exclude_target_history = not bool(args.keep_target_history)
    excluded_input_targets = list(args.exclude_input_targets) if exclude_target_history else []
    model = SingleTaskMLPModel.from_config(
        config,
        metadata["input_dims"],
        target_name=target_name,
        exclude_target_history=exclude_target_history,
        excluded_input_names=excluded_input_targets,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
    val_interval = int(config["training"].get("val_interval", 1))
    patience = int(config["training"].get("patience", 5))
    epochs = int(config["training"]["epochs"])
    checkpoint_path = target_dir / "best_model.pt"
    log_path = target_dir / "train_log.csv"

    best_val = float("inf")
    best_epoch = -1
    stale_val_checks = 0
    target_start = time.perf_counter()
    total_train_seconds = 0.0
    total_val_seconds = 0.0
    print("=" * 100)
    print(f"Training single-task MLP target={target_name} index={target_index}")
    print("Excluded input targets:", excluded_input_targets)
    print("Input exclusions:", model.input_exclusions)
    print("Effective input dims:", model.effective_input_dims)
    print("Static land features:", config["data"].get("static_land_features", {"enabled": False}))

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            target_index=target_index,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            max_batches=args.max_train_batches,
        )
        train_seconds = time.perf_counter() - train_start
        total_train_seconds += train_seconds
        val_metrics = None
        val_seconds = 0.0
        if epoch % val_interval == 0 or epoch == epochs:
            val_start = time.perf_counter()
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    loaders["val"],
                    target_index=target_index,
                    device=device,
                    criterion=criterion,
                    optimizer=None,
                    max_batches=args.max_eval_batches,
                )
            val_seconds = time.perf_counter() - val_start
            total_val_seconds += val_seconds
            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                stale_val_checks = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "input_dims": metadata["input_dims"],
                        "main_targets": metadata["main_targets"],
                        "main_target_units": metadata["main_target_units"],
                        "target_name": target_name,
                        "target_index": target_index,
                        "target_unit": metadata["main_target_units"][target_index],
                        "input_window": metadata["input_window"],
                        "horizon": int(config["data"].get("horizon", 1)),
                        "output_horizon": int(metadata.get("output_horizon", config["data"].get("output_horizon", 1))),
                        "exclude_target_history": exclude_target_history,
                        "excluded_input_targets": excluded_input_targets,
                        "input_exclusions": model.input_exclusions,
                        "effective_input_dims": model.effective_input_dims,
                        "unit_convention": metadata["unit_convention"],
                        "external_et_targets": metadata["external_et_targets"],
                        "static_land_features": metadata.get("static_land_features", {"enabled": False}),
                        "finite_point_filter": metadata["finite_point_filter"],
                        "normalizer_mean": metadata["normalizer"].mean,
                        "normalizer_std": metadata["normalizer"].std,
                        "best_val_loss": best_val,
                        "best_epoch": best_epoch,
                    },
                    checkpoint_path,
                )
            else:
                stale_val_checks += 1
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - target_start
        row = {
            "target": target_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stale_val_checks": stale_val_checks,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
        }
        append_log(log_path, row)
        if val_metrics is None:
            print(
                f"target={target_name} epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val=skipped time={epoch_seconds:.2f}s"
            )
        else:
            print(
                f"target={target_name} epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} stale_val_checks={stale_val_checks} "
                f"train_time={train_seconds:.2f}s val_time={val_seconds:.2f}s epoch_time={epoch_seconds:.2f}s"
            )
        if stale_val_checks >= patience:
            print(
                f"Early stopping target={target_name} at epoch {epoch}; "
                f"best epoch={best_epoch}, val={best_val:.6f}"
            )
            break

    if not checkpoint_path.exists():
        raise RuntimeError(f"No checkpoint was saved for target={target_name}.")
    total_seconds = time.perf_counter() - target_start
    target_summary = {
        "target": target_name,
        "target_index": target_index,
        "input_window": metadata["input_window"],
        "horizon": int(config["data"].get("horizon", 1)),
        "output_horizon": int(metadata.get("output_horizon", config["data"].get("output_horizon", 1))),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint_path),
        "exclude_target_history": exclude_target_history,
        "excluded_input_targets": excluded_input_targets,
        "input_exclusions": model.input_exclusions,
        "effective_input_dims": model.effective_input_dims,
        "static_land_features": metadata.get("static_land_features", {"enabled": False}),
        "epochs_ran": epoch,
        "total_seconds": total_seconds,
        "total_train_seconds": total_train_seconds,
        "total_val_seconds": total_val_seconds,
    }
    save_json(target_dir / "target_training_summary.json", target_summary)
    return {
        **target_summary,
    }


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_yaml_config(args.config), args)
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    seed = int(config["data"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(config)
    loop_start = time.perf_counter()
    print(f"Using device: {device}")
    print("Preparing dataloaders once for all single-task models...")
    loaders, metadata = prepare_dataloaders(config)
    print(
        "Dataset sizes:",
        {name: len(loader.dataset) for name, loader in loaders.items()},
        "input_dims:",
        metadata["input_dims"],
        "output_horizon:",
        metadata.get("output_horizon", config["data"].get("output_horizon", 1)),
        "main_targets:",
        metadata["main_targets"],
    )
    print("Feature groups:", metadata.get("feature_groups", {}))
    print("Static land features:", metadata.get("static_land_features", {"enabled": False}))
    print("Finite point filter:", metadata.get("finite_point_filter", {}))

    requested_targets = args.targets or list(DEFAULT_HYDRO_EXCLUSION_TARGETS)
    missing = [name for name in requested_targets if name not in metadata["main_targets"]]
    if missing:
        raise KeyError(f"Requested targets are not in main_targets: {missing}")

    summaries = []
    for offset, target_name in enumerate(requested_targets):
        target_index = list(metadata["main_targets"]).index(target_name)
        torch.manual_seed(seed + offset)
        np.random.seed(seed + offset)
        summaries.append(
            train_one_target(
                target_name=target_name,
                target_index=target_index,
                config=config,
                loaders=loaders,
                metadata=metadata,
                device=device,
                args=args,
            )
        )

    loop_total_seconds = time.perf_counter() - loop_start
    save_json(
        output_dir / "training_summary.json",
        {
            "models": summaries,
            "total_seconds": loop_total_seconds,
            "total_train_seconds": sum(float(item["total_train_seconds"]) for item in summaries),
            "total_val_seconds": sum(float(item["total_val_seconds"]) for item in summaries),
            "exclude_target_history": not bool(args.keep_target_history),
            "input_window": metadata["input_window"],
            "horizon": int(config["data"].get("horizon", 1)),
            "output_horizon": int(metadata.get("output_horizon", config["data"].get("output_horizon", 1))),
            "static_land_features": metadata.get("static_land_features", {"enabled": False}),
            "excluded_input_targets": (
                []
                if bool(args.keep_target_history)
                else list(args.exclude_input_targets)
            ),
        },
    )
    print("=" * 100)
    print(f"Finished single-task MLP loop in {loop_total_seconds:.2f}s.")
    for item in summaries:
        print(
            f"{item['target']}: best_epoch={item['best_epoch']} "
            f"val_loss={item['best_val_loss']:.6f} total_time={item['total_seconds']:.2f}s "
            f"checkpoint={item['checkpoint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
