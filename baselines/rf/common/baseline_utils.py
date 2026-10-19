from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


INPUT_GROUPS = ("atm_total", "land_total", "atm_anom", "land_anom")
DEFAULT_TARGETS = ("swvl1", "e_pos", "evavt_pos")
DEFAULT_HYDRO_EXCLUSIONS = ("swvl1", "e_pos", "evavt_pos", "eb_pos")


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_common_overrides(config: dict[str, Any], args: Any, *, default_output_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if getattr(args, "data_path", None) is not None:
        cfg["data"]["path"] = str(args.data_path)
    if getattr(args, "output_dir", None) is not None:
        cfg["data"]["output_dir"] = str(args.output_dir)
    else:
        cfg["data"]["output_dir"] = str(default_output_dir)
    if getattr(args, "epochs", None) is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if getattr(args, "max_points", None) is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    if getattr(args, "val_interval", None) is not None:
        cfg["training"]["val_interval"] = int(args.val_interval)
    if getattr(args, "patience", None) is not None:
        cfg["training"]["patience"] = int(args.patience)
    if getattr(args, "seed", None) is not None:
        cfg["data"]["seed"] = int(args.seed)
        cfg.setdefault("baseline", {})["seed"] = int(args.seed)
    if getattr(args, "gpu_id", None) is not None and getattr(args, "device", None) not in (None, "auto"):
        raise ValueError("Use only one of --device or --gpu-id.")
    if getattr(args, "gpu_id", None) is not None:
        cfg["training"]["device"] = f"cuda:{int(args.gpu_id)}"
    elif getattr(args, "device", None) is not None:
        cfg["training"]["device"] = str(args.device)
    return cfg


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


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


def target_index(metadata: dict[str, Any], target: str) -> int:
    targets = list(metadata["main_targets"])
    if target not in targets:
        raise KeyError(f"Target {target!r} is not in config main_targets={targets}.")
    return targets.index(target)


def target_unit(metadata: dict[str, Any], target: str) -> str:
    idx = target_index(metadata, target)
    return str(metadata["main_target_units"][idx])


def select_target(batch_or_array: torch.Tensor, target_idx: int) -> torch.Tensor:
    if batch_or_array.ndim == 2:
        return batch_or_array[:, target_idx : target_idx + 1].unsqueeze(1)
    if batch_or_array.ndim == 3:
        return batch_or_array[:, :, target_idx : target_idx + 1]
    raise ValueError(f"Expected target shape [B,V] or [B,L,V], got {tuple(batch_or_array.shape)}.")


def input_exclusions(config: dict[str, Any]) -> dict[str, list[str]]:
    excluded = set(config.get("baseline", {}).get("exclude_input_targets", DEFAULT_HYDRO_EXCLUSIONS))
    out: dict[str, list[str]] = {}
    for group in INPUT_GROUPS:
        variables = list(config["variables"].get(group, []))
        out[group] = [name for name in variables if name in excluded]
    return out


def keep_indices(config: dict[str, Any]) -> dict[str, list[int]]:
    exclusions = input_exclusions(config)
    out: dict[str, list[int]] = {}
    for group in INPUT_GROUPS:
        variables = list(config["variables"].get(group, []))
        excluded = set(exclusions[group])
        keep = [idx for idx, name in enumerate(variables) if name not in excluded]
        if not keep:
            raise ValueError(f"All features excluded for input group {group}.")
        out[group] = keep
    return out


def concat_sequence_features(batch: dict[str, torch.Tensor], keep: dict[str, list[int]]) -> torch.Tensor:
    parts = []
    for group in INPUT_GROUPS:
        x = batch[group]
        if x.ndim != 3:
            raise ValueError(f"Expected {group} shape [B,T,F], got {tuple(x.shape)}.")
        indices = torch.as_tensor(keep[group], device=x.device, dtype=torch.long)
        parts.append(x.index_select(dim=-1, index=indices))
    return torch.cat(parts, dim=-1)


def concat_flat_features(batch: dict[str, torch.Tensor], keep: dict[str, list[int]]) -> torch.Tensor:
    return concat_sequence_features(batch, keep).flatten(start_dim=1)


def denormalize_main(values: np.ndarray, checkpoint: dict[str, Any], target_idx: int) -> np.ndarray:
    mean = np.asarray(checkpoint["normalizer_mean"]["main_targets"], dtype=np.float64)[target_idx]
    std = np.asarray(checkpoint["normalizer_std"]["main_targets"], dtype=np.float64)[target_idx]
    return values * std + mean


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    target: str,
    target_idx: int,
    model_type: str,
    best_epoch: int | None,
    best_val: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalizer = metadata["normalizer"]
    payload = {
        "model_type": model_type,
        "config": config,
        "target_name": target,
        "target_index": int(target_idx),
        "target_unit": target_unit(metadata, target),
        "main_targets": list(metadata["main_targets"]),
        "main_target_units": list(metadata["main_target_units"]),
        "input_dims": dict(metadata["input_dims"]),
        "input_keep_indices": keep_indices(config),
        "input_exclusions": input_exclusions(config),
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "unit_convention": metadata.get("unit_convention"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }
    if extra:
        payload.update(extra)
    return payload
