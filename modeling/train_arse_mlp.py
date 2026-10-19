from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from data_module import load_yaml_config, prepare_dataloaders
    from model_arse_mlp import ARSEMLPCoupledModel
else:
    from .data_module import load_yaml_config, prepare_dataloaders
    from .model_arse_mlp import ARSEMLPCoupledModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ARSE MLP coupled model.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_arse_mlp.yaml"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu-id", type=int, help="Shortcut for --device cuda:<id>.")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.data_path is not None:
        cfg["data"]["path"] = str(args.data_path)
    if args.output_dir is not None:
        cfg["data"]["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
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
    return cfg


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


def compute_main_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    *,
    mode: str,
    main_targets: list[str],
    main_head_groups: list[list[str]],
    priority_target: str | None,
    priority_rho: float,
    head_weights: list[float] | None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if mode == "variable_mean":
        return criterion(pred, target)
    if mode != "head_weighted":
        raise ValueError(f"Unsupported training.main_loss_mode={mode!r}")

    reduce_dims = tuple(range(pred.ndim - 1))
    per_variable_loss = (pred - target).pow(2).mean(dim=reduce_dims)
    target_index = {name: idx for idx, name in enumerate(main_targets)}
    head_losses = []
    for group in main_head_groups:
        weights = pred.new_ones(len(group))
        if priority_target is not None:
            for idx, name in enumerate(group):
                if name == priority_target:
                    weights[idx] = float(priority_rho)
        indices = torch.as_tensor(
            [target_index[name] for name in group],
            dtype=torch.long,
            device=pred.device,
        )
        losses = per_variable_loss.index_select(0, indices)
        head_losses.append((losses * weights).sum() / weights.sum())

    head_loss_tensor = torch.stack(head_losses)
    if head_weights is None:
        return head_loss_tensor.mean()
    if len(head_weights) != len(head_losses):
        raise ValueError(
            "training.main_head_loss_weights must have the same length as "
            f"model.main_head_groups. got {len(head_weights)} and {len(head_losses)}"
        )
    weight_tensor = torch.as_tensor(head_weights, dtype=pred.dtype, device=pred.device)
    if torch.any(weight_tensor < 0) or float(weight_tensor.sum()) <= 0:
        raise ValueError("training.main_head_loss_weights must be non-negative and sum to a positive value.")
    return (head_loss_tensor * weight_tensor).sum() / weight_tensor.sum()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    *,
    aux_weight: float,
    main_loss_mode: str = "variable_mean",
    main_targets: list[str] | None = None,
    main_head_groups: list[list[str]] | None = None,
    main_loss_priority_target: str | None = None,
    main_loss_priority_rho: float = 1.0,
    main_head_loss_weights: list[float] | None = None,
    component_weight: float = 0.0,
    main_target_mean: torch.Tensor | None = None,
    main_target_std: torch.Tensor | None = None,
    component_indices: tuple[int, ...] | None = None,
    future_aux_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    main_loss = compute_main_loss(
        outputs["main"],
        batch["main_target"],
        criterion,
        mode=main_loss_mode,
        main_targets=list(main_targets or []),
        main_head_groups=list(main_head_groups or [list(main_targets or [])]),
        priority_target=main_loss_priority_target,
        priority_rho=float(main_loss_priority_rho),
        head_weights=main_head_loss_weights,
    )
    aux_loss = criterion(outputs["aux"], batch["aux_target"])
    future_aux_loss = outputs["main"].new_zeros(())
    if float(future_aux_weight) > 0:
        if outputs.get("future_aux") is None:
            raise RuntimeError(
                "training.future_aux_loss_weight is positive, but the model did not return future_aux."
            )
        if "future_aux_target" not in batch:
            raise RuntimeError(
                "training.future_aux_loss_weight is positive, but the batch is missing future_aux_target."
            )
        future_aux_loss = criterion(outputs["future_aux"], batch["future_aux_target"])
    component_loss = outputs["main"].new_zeros(())
    if (
        component_weight > 0
        and main_target_mean is not None
        and main_target_std is not None
        and component_indices is not None
    ):
        e_idx, *component_idxs = component_indices
        main_phys = outputs["main"] * main_target_std + main_target_mean
        e_pred = main_phys[..., e_idx]
        component_sum = sum(main_phys[..., idx] for idx in component_idxs)
        component_loss = torch.relu(component_sum - e_pred).pow(2).mean()
    loss = (
        main_loss
        + float(aux_weight) * aux_loss
        + float(component_weight) * component_loss
        + float(future_aux_weight) * future_aux_loss
    )
    return (
        loss,
        main_loss.detach(),
        aux_loss.detach(),
        component_loss.detach(),
        future_aux_loss.detach(),
    )


def run_epoch(
    model: ARSEMLPCoupledModel,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    aux_weight: float,
    main_loss_mode: str = "variable_mean",
    main_targets: list[str] | None = None,
    main_head_groups: list[list[str]] | None = None,
    main_loss_priority_target: str | None = None,
    main_loss_priority_rho: float = 1.0,
    main_head_loss_weights: list[float] | None = None,
    component_weight: float = 0.0,
    main_target_mean: torch.Tensor | None = None,
    main_target_std: torch.Tensor | None = None,
    component_indices: tuple[int, ...] | None = None,
    future_aux_weight: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {
        "loss": 0.0,
        "main_loss": 0.0,
        "aux_loss": 0.0,
        "component_loss": 0.0,
        "future_aux_loss": 0.0,
        "n": 0,
    }

    for batch_idx, batch_raw in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch(batch_raw, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            loss, main_loss, aux_loss, component_loss, future_aux_loss = compute_loss(
                outputs,
                batch,
                criterion,
                aux_weight=aux_weight,
                main_loss_mode=main_loss_mode,
                main_targets=main_targets,
                main_head_groups=main_head_groups,
                main_loss_priority_target=main_loss_priority_target,
                main_loss_priority_rho=main_loss_priority_rho,
                main_head_loss_weights=main_head_loss_weights,
                component_weight=component_weight,
                main_target_mean=main_target_mean,
                main_target_std=main_target_std,
                component_indices=component_indices,
                future_aux_weight=future_aux_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

        n = int(batch["main_target"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["main_loss"] += float(main_loss.cpu()) * n
        totals["aux_loss"] += float(aux_loss.cpu()) * n
        totals["component_loss"] += float(component_loss.cpu()) * n
        totals["future_aux_loss"] += float(future_aux_loss.cpu()) * n
        totals["n"] += n

    if totals["n"] == 0:
        raise RuntimeError("No batches were processed.")
    return {key: value / totals["n"] for key, value in totals.items() if key != "n"}


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


def plot_main_train_and_val_loss(log_path: Path, output_dir: Path) -> Path | None:
    if not log_path.exists():
        return None

    epochs: list[int] = []
    train_main_losses: list[float] = []
    val_losses: list[float] = []
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_loss = row.get("val_loss", "")
            train_main_loss = row.get("train_main_loss", "")
            if val_loss == "" or train_main_loss == "":
                continue
            epochs.append(int(row["epoch"]))
            train_main_losses.append(float(train_main_loss))
            val_losses.append(float(val_loss))

    if not epochs:
        return None

    curve_csv_path = output_dir / "main_train_loss_vs_val_loss.csv"
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_main_loss", "val_loss"])
        writer.writeheader()
        for epoch, train_main_loss, val_loss in zip(epochs, train_main_losses, val_losses):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_main_loss": train_main_loss,
                    "val_loss": val_loss,
                }
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime image stack.
        error_path = output_dir / "loss_curve_plot_error.txt"
        error_path.write_text(
            f"Could not create loss curve because matplotlib import failed: {exc}\n",
            encoding="utf-8",
        )
        return None

    plot_path = output_dir / "main_train_loss_vs_val_loss.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, train_main_losses, marker="o", linewidth=1.8, label="main train loss")
    ax.plot(epochs, val_losses, marker="s", linewidth=1.8, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Main Train Loss and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    wall_start_time = time.perf_counter()
    config = apply_overrides(load_yaml_config(args.config), args)
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    torch.manual_seed(int(config["data"].get("seed", 42)))
    np.random.seed(int(config["data"].get("seed", 42)))
    device = resolve_device(config)
    print(f"Using device: {device}")
    print("Preparing dataloaders...")
    data_prepare_start_time = time.perf_counter()
    loaders, metadata = prepare_dataloaders(config)
    data_prepare_seconds = time.perf_counter() - data_prepare_start_time
    print(
        "Dataset sizes:",
        {name: len(loader.dataset) for name, loader in loaders.items()},
        "input_dims:",
        metadata["input_dims"],
    )
    print(f"Data preparation time: {data_prepare_seconds:.2f}s")
    print("Finite point filter:", metadata.get("finite_point_filter", {}))

    model_setup_start_time = time.perf_counter()
    model = ARSEMLPCoupledModel.from_config(config, metadata["input_dims"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    aux_weight = float(config["training"]["aux_loss_weight"])
    main_loss_mode = str(config["training"].get("main_loss_mode", "variable_mean"))
    main_loss_priority_target = config["training"].get("main_loss_priority_target")
    main_loss_priority_rho = float(config["training"].get("main_loss_priority_rho", 1.0))
    main_head_loss_weights = config["training"].get("main_head_loss_weights")
    if main_head_loss_weights is not None:
        main_head_loss_weights = [float(value) for value in main_head_loss_weights]
    component_weight = float(config["training"].get("component_constraint_weight", 0.0))
    future_aux_weight = float(config["training"].get("future_aux_loss_weight", 0.0))
    main_target_mean = torch.as_tensor(
        metadata["normalizer"].mean["main_targets"],
        dtype=torch.float32,
        device=device,
    )
    main_target_std = torch.as_tensor(
        metadata["normalizer"].std["main_targets"],
        dtype=torch.float32,
        device=device,
    )
    main_targets = list(metadata["main_targets"])
    required_component_targets = ("e_pos", "evavt_pos")
    optional_component_targets = ("eb_pos",)
    component_indices = None
    if component_weight > 0:
        missing_component_targets = [
            name for name in required_component_targets if name not in main_targets
        ]
        if missing_component_targets:
            print(
                "Component constraint disabled because required targets are missing:",
                missing_component_targets,
            )
            component_weight = 0.0
        else:
            component_targets = list(required_component_targets)
            component_targets.extend(name for name in optional_component_targets if name in main_targets)
            component_indices = tuple(main_targets.index(name) for name in component_targets)
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
    model_setup_seconds = time.perf_counter() - model_setup_start_time
    print(f"Model/optimizer setup time: {model_setup_seconds:.2f}s")
    print(
        "Main loss:",
        {
            "mode": main_loss_mode,
            "head_groups": model.main_head_groups,
            "priority_target": main_loss_priority_target,
            "priority_rho": main_loss_priority_rho,
            "head_weights": main_head_loss_weights,
        },
    )
    print(
        "Auxiliary losses:",
        {
            "aux_loss_weight": aux_weight,
            "component_constraint_weight": component_weight,
            "future_aux_loss_weight": future_aux_weight,
            "future_aux_targets": metadata.get("future_aux_targets", []),
            "future_aux_target_groups": metadata.get("future_aux_target_groups", {}),
        },
    )

    best_val = float("inf")
    best_epoch = -1
    val_interval = int(config["training"].get("val_interval", 5))
    if val_interval <= 0:
        raise ValueError("training.val_interval must be a positive integer.")
    patience = int(config["training"]["patience"])
    if patience <= 0:
        raise ValueError("training.patience must be a positive integer.")
    epochs = int(config["training"]["epochs"])
    max_train_batches = args.max_train_batches
    max_eval_batches = args.max_eval_batches
    validations_without_improvement = 0

    checkpoint_path = output_dir / "best_model.pt"
    log_path = output_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()
    train_start_time = time.perf_counter()
    total_train_seconds = 0.0
    total_val_seconds = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()
        epoch_train_start_time = time.perf_counter()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device=device,
            criterion=criterion,
            aux_weight=aux_weight,
            main_loss_mode=main_loss_mode,
            main_targets=main_targets,
            main_head_groups=model.main_head_groups,
            main_loss_priority_target=main_loss_priority_target,
            main_loss_priority_rho=main_loss_priority_rho,
            main_head_loss_weights=main_head_loss_weights,
            component_weight=component_weight,
            main_target_mean=main_target_mean,
            main_target_std=main_target_std,
            component_indices=component_indices,
            future_aux_weight=future_aux_weight,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
        )
        train_seconds = time.perf_counter() - epoch_train_start_time
        total_train_seconds += train_seconds

        should_validate = (epoch % val_interval == 0) or (epoch == epochs)
        val_metrics = None
        val_seconds = 0.0
        if should_validate:
            epoch_val_start_time = time.perf_counter()
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    loaders["val"],
                    device=device,
                    criterion=criterion,
                    aux_weight=aux_weight,
                    main_loss_mode=main_loss_mode,
                    main_targets=main_targets,
                    main_head_groups=model.main_head_groups,
                    main_loss_priority_target=main_loss_priority_target,
                    main_loss_priority_rho=main_loss_priority_rho,
                    main_head_loss_weights=main_head_loss_weights,
                    component_weight=component_weight,
                    main_target_mean=main_target_mean,
                    main_target_std=main_target_std,
                    component_indices=component_indices,
                    future_aux_weight=future_aux_weight,
                    optimizer=None,
                    max_batches=max_eval_batches,
                )
            val_seconds = time.perf_counter() - epoch_val_start_time
            total_val_seconds += val_seconds
        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - train_start_time

        if val_metrics is None:
            append_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_main_loss": train_metrics["main_loss"],
                    "train_aux_loss": train_metrics["aux_loss"],
                    "train_component_loss": train_metrics["component_loss"],
                    "train_future_aux_loss": train_metrics["future_aux_loss"],
                    "val_loss": "",
                    "val_main_loss": "",
                    "val_aux_loss": "",
                    "val_component_loss": "",
                    "val_future_aux_loss": "",
                    "best_val_loss": "" if best_epoch < 0 else best_val,
                    "best_epoch": "" if best_epoch < 0 else best_epoch,
                    "validations_without_improvement": validations_without_improvement,
                    "train_seconds": train_seconds,
                    "val_seconds": val_seconds,
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": elapsed_seconds,
                },
            )
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val=skipped time={epoch_seconds:.2f}s"
            )
            continue

        improved = val_metrics["loss"] < best_val
        should_stop = False
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            validations_without_improvement = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "input_dims": metadata["input_dims"],
                "main_targets": metadata["main_targets"],
                "aux_targets": metadata["aux_targets"],
                "future_aux_targets": metadata.get("future_aux_targets", []),
                "future_aux_target_groups": metadata.get("future_aux_target_groups", {}),
                "future_aux_group_names": metadata.get("future_aux_group_names", []),
                "main_target_units": metadata["main_target_units"],
                "aux_target_units": metadata["aux_target_units"],
                "future_aux_target_units": metadata.get("future_aux_target_units", []),
                "mode_variables": metadata["mode_variables"],
                "mode_features": metadata["mode_features"],
                "unit_convention": metadata["unit_convention"],
                "external_et_targets": metadata["external_et_targets"],
                "finite_point_filter": metadata["finite_point_filter"],
                "input_window": metadata["input_window"],
                "horizon": metadata["horizon"],
                "output_horizon": metadata["output_horizon"],
                "normalizer_mean": metadata["normalizer"].mean,
                "normalizer_std": metadata["normalizer"].std,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "main_head_groups": config.get("model", {}).get("main_head_groups"),
                "main_head_raw_variables": config.get("model", {}).get("main_head_raw_variables"),
                "main_loss_mode": main_loss_mode,
                "main_loss_priority_target": main_loss_priority_target,
                "main_loss_priority_rho": main_loss_priority_rho,
                "main_head_loss_weights": main_head_loss_weights,
                "future_aux_loss_weight": future_aux_weight,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            validations_without_improvement += 1
            if validations_without_improvement >= patience:
                should_stop = True

        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_main_loss": train_metrics["main_loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_component_loss": train_metrics["component_loss"],
                "train_future_aux_loss": train_metrics["future_aux_loss"],
                "val_loss": val_metrics["loss"],
                "val_main_loss": val_metrics["main_loss"],
                "val_aux_loss": val_metrics["aux_loss"],
                "val_component_loss": val_metrics["component_loss"],
                "val_future_aux_loss": val_metrics["future_aux_loss"],
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "validations_without_improvement": validations_without_improvement,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )
        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"main_val={val_metrics['main_loss']:.6f} "
            f"aux_val={val_metrics['aux_loss']:.6f} "
            f"component_val={val_metrics['component_loss']:.6f} "
            f"future_aux_val={val_metrics['future_aux_loss']:.6f} "
            f"stale_val_checks={validations_without_improvement} "
            f"train_time={train_seconds:.2f}s "
            f"val_time={val_seconds:.2f}s "
            f"epoch_time={epoch_seconds:.2f}s"
        )

        if improved:
            print(
                f"Saved new best checkpoint at epoch={best_epoch}, "
                f"val_loss={best_val:.6f}"
            )
        if should_stop:
            print(
                "Early stopping at "
                f"epoch {epoch}; best epoch={best_epoch}, "
                f"val={best_val:.6f}, "
                f"no improvement for {validations_without_improvement} validation checks"
            )
            break

    optimization_seconds = time.perf_counter() - train_start_time
    total_wall_seconds = time.perf_counter() - wall_start_time
    loss_curve_path = plot_main_train_and_val_loss(log_path, output_dir)
    loss_curve_csv_path = output_dir / "main_train_loss_vs_val_loss.csv"
    save_json(
        output_dir / "training_summary.json",
        {
            "best_checkpoint": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs_ran": epoch,
            "total_seconds": total_wall_seconds,
            "total_wall_seconds": total_wall_seconds,
            "data_prepare_seconds": data_prepare_seconds,
            "model_setup_seconds": model_setup_seconds,
            "optimization_seconds": optimization_seconds,
            "total_train_seconds": total_train_seconds,
            "total_val_seconds": total_val_seconds,
            "main_targets": main_targets,
            "aux_targets": list(metadata["aux_targets"]),
            "future_aux_targets": list(metadata.get("future_aux_targets", [])),
            "future_aux_target_groups": metadata.get("future_aux_target_groups", {}),
            "future_aux_group_names": list(metadata.get("future_aux_group_names", [])),
            "input_window": metadata["input_window"],
            "horizon": metadata["horizon"],
            "output_horizon": metadata["output_horizon"],
            "main_head_groups": config.get("model", {}).get("main_head_groups"),
            "main_head_raw_variables": config.get("model", {}).get("main_head_raw_variables"),
            "main_loss_mode": main_loss_mode,
            "main_loss_priority_target": main_loss_priority_target,
            "main_loss_priority_rho": main_loss_priority_rho,
            "main_head_loss_weights": main_head_loss_weights,
            "future_aux_loss_weight": future_aux_weight,
            "loss_curve_csv_path": str(loss_curve_csv_path) if loss_curve_csv_path.exists() else None,
            "loss_curve_path": str(loss_curve_path) if loss_curve_path is not None else None,
        },
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(
        f"Best epoch: {best_epoch}, val_loss={best_val:.6f}, "
        f"optimization_time={optimization_seconds:.2f}s, total_wall_time={total_wall_seconds:.2f}s"
    )
    if loss_curve_path is not None:
        print(f"Loss curve CSV: {loss_curve_csv_path}")
        print(f"Loss curve: {loss_curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
