from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
sys.path.append(str(COMMON_DIR))

from baseline_utils import (  # noqa: E402
    apply_common_overrides,
    checkpoint_payload,
    concat_flat_features,
    deep_update,
    denormalize_main,
    save_json,
    select_target,
    set_global_seed,
    target_index,
)
from data_module import load_yaml_config, prepare_dataloaders  # noqa: E402
from metrics import regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one single-task Random Forest baseline.")
    parser.add_argument("--config", type=Path, default=COMMON_DIR / "config_baseline_common.yaml")
    parser.add_argument("--model-config", type=Path, default=CURRENT_DIR / "config_rf.yaml")
    parser.add_argument("--target", choices=("swvl1", "e_pos", "evavt_pos"), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--n-jobs", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = deep_update(load_yaml_config(args.config), load_yaml_config(args.model_config))
    cfg = apply_common_overrides(
        cfg,
        args,
        default_output_dir=CURRENT_DIR.parent.parent / "outputs" / "rf",
    )
    if args.max_train_samples is not None:
        cfg.setdefault("rf", {})["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        cfg.setdefault("rf", {})["max_val_samples"] = int(args.max_val_samples)
    model_cfg = cfg.setdefault("model", {})
    if args.n_estimators is not None:
        model_cfg["n_estimators"] = int(args.n_estimators)
    if args.max_depth is not None:
        model_cfg["max_depth"] = int(args.max_depth)
    if args.min_samples_leaf is not None:
        model_cfg["min_samples_leaf"] = int(args.min_samples_leaf)
    if args.n_jobs is not None:
        model_cfg["n_jobs"] = int(args.n_jobs)
    return cfg


def collect_samples(loader, *, keep: dict[str, list[int]], target_idx: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    total = 0
    for raw_batch in loader:
        x = concat_flat_features(raw_batch, keep).numpy().astype(np.float32, copy=False)
        y = select_target(raw_batch["main_target"], target_idx).numpy()[:, 0, 0].astype(np.float32, copy=False)
        if max_samples is not None and total + x.shape[0] > int(max_samples):
            take = int(max_samples) - total
            if take <= 0:
                break
            x = x[:take]
            y = y[:take]
        xs.append(x)
        ys.append(y)
        total += int(x.shape[0])
        if max_samples is not None and total >= int(max_samples):
            break
    if not xs:
        raise RuntimeError("No samples collected for RF training/evaluation.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    seed = int(config.get("baseline", {}).get("seed", config["data"].get("seed", 42)))
    set_global_seed(seed)
    target_dir = Path(config["data"]["output_dir"]) / args.target
    target_dir.mkdir(parents=True, exist_ok=True)
    print("Preparing dataloaders...", flush=True)
    start = time.perf_counter()
    loaders, metadata = prepare_dataloaders(config)
    print(
        f"Dataset sizes: { {name: len(loader.dataset) for name, loader in loaders.items()} } "
        f"input_dims: {metadata['input_dims']}",
        flush=True,
    )
    target_idx = target_index(metadata, args.target)
    payload = checkpoint_payload(
        config=config,
        metadata=metadata,
        target=args.target,
        target_idx=target_idx,
        model_type="rf",
        best_epoch=None,
        best_val=None,
    )
    keep = payload["input_keep_indices"]
    max_train_samples = int(config.get("rf", {}).get("max_train_samples", 500000))
    max_val_samples = int(config.get("rf", {}).get("max_val_samples", 200000))
    print(f"Collecting RF samples: train={max_train_samples}, val={max_val_samples}", flush=True)
    x_train, y_train = collect_samples(
        loaders["train"],
        keep=keep,
        target_idx=target_idx,
        max_samples=max_train_samples,
    )
    x_val, y_val = collect_samples(
        loaders["val"],
        keep=keep,
        target_idx=target_idx,
        max_samples=max_val_samples,
    )
    print(f"Training RF target={args.target} x_train={x_train.shape}", flush=True)
    model_cfg = config["model"]
    max_depth = model_cfg.get("max_depth", 30)
    if max_depth in ("null", "none", None):
        max_depth = None
    rf = RandomForestRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=max_depth,
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 5)),
        max_features=model_cfg.get("max_features", "sqrt"),
        random_state=seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
        verbose=1,
    )
    rf.fit(x_train, y_train)
    val_pred_norm = rf.predict(x_val).reshape(-1, 1)
    val_true_norm = y_val.reshape(-1, 1)
    val_pred = denormalize_main(val_pred_norm, payload, target_idx)
    val_true = denormalize_main(val_true_norm, payload, target_idx)
    val_metrics = regression_metrics(val_true, val_pred, [args.target])
    val_metrics.insert(1, "unit", payload["target_unit"])
    val_metrics.to_csv(target_dir / "metrics_main_val_h1_sampled.csv", index=False)
    checkpoint = dict(payload)
    checkpoint["model"] = rf
    checkpoint["best_val_metrics"] = val_metrics.to_dict(orient="records")
    checkpoint_path = target_dir / "best_model.joblib"
    joblib.dump(checkpoint, checkpoint_path)
    summary = {
        "model_type": "rf",
        "target": args.target,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "x_train_shape": list(x_train.shape),
        "x_val_shape": list(x_val.shape),
        "elapsed_seconds": time.perf_counter() - start,
        "val_metrics": val_metrics.to_dict(orient="records"),
        "input_exclusions": payload["input_exclusions"],
    }
    save_json(target_dir / "target_training_summary.json", summary)
    pd.DataFrame([summary]).drop(columns=["val_metrics", "input_exclusions"]).to_csv(target_dir / "train_log.csv", index=False)
    print(val_metrics.to_string(index=False), flush=True)
    print(f"Best checkpoint: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
