from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PARENT_DIR))

from model_mediator_lstm import CounterfactualMediatorLSTM  # noqa: E402
from train_mediator_lstm import (  # noqa: E402
    prepare_mediator_dataloaders,
    evaluate_model,
    resolve_device,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a counterfactual mediator LSTM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu-id", type=int)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def apply_eval_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    cfg["data"] = dict(cfg["data"])
    cfg["training"] = dict(cfg["training"])
    cfg["physical_constraints"] = dict(cfg.get("physical_constraints", {}))
    cfg["physical_constraints"].setdefault("h_sign_constraint_weight", 0.0)
    cfg["physical_constraints"].setdefault("h_sign_flux_scale", 10.0)
    cfg["physical_constraints"].setdefault("h_sign_temp_scale", 1.0)
    if args.data_path is not None:
        cfg["data"]["path"] = str(args.data_path)
    if args.output_dir is not None:
        cfg["data"]["output_dir"] = str(args.output_dir)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.max_points is not None:
        cfg["data"]["max_points"] = int(args.max_points)
    if args.device is not None and args.gpu_id is not None:
        raise ValueError("Use only one of --device or --gpu-id.")
    if args.gpu_id is not None:
        cfg["training"]["device"] = f"cuda:{int(args.gpu_id)}"
    elif args.device is not None:
        cfg["training"]["device"] = args.device
    return cfg


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()

    initial_device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, initial_device)
    config = apply_eval_overrides(checkpoint["config"], args)
    device = resolve_device(config)
    checkpoint = load_checkpoint(args.checkpoint, device)

    prediction_mode = str(checkpoint["prediction_mode"])
    mediator_targets = list(checkpoint["mediator_targets"])
    output_dir = args.output_dir or (args.checkpoint.parent / f"eval_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prediction mode: {prediction_mode}")
    print(f"Mediator targets: {mediator_targets}")
    print("Preparing dataloaders...")
    data = prepare_mediator_dataloaders(config, mediator_targets)
    print(
        "Dataset sizes:",
        {name: len(loader.dataset) for name, loader in data.loaders.items()},
        "input_dims:",
        data.metadata["input_dims"],
    )

    model = CounterfactualMediatorLSTM.from_config(
        config,
        input_dims=data.metadata["input_dims"],
        mediator_dim=len(mediator_targets),
        prediction_mode=prediction_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    eval_summary = evaluate_model(
        model,
        data.loaders[args.split],
        device=device,
        metadata=data.metadata,
        split=args.split,
        output_dir=output_dir,
        max_batches=args.max_eval_batches,
        physical_constraints=config["physical_constraints"],
    )
    save_json(
        output_dir / f"evaluation_summary_{args.split}.json",
        {
            "checkpoint": str(args.checkpoint),
            "prediction_mode": prediction_mode,
            "split": args.split,
            "mediator_targets": mediator_targets,
            "physical_constraints": config.get("physical_constraints", {}),
            "metrics": eval_summary,
        },
    )

    print("Absolute-value metrics:")
    import pandas as pd

    print(pd.DataFrame(eval_summary["abs_all"]).to_string(index=False))
    print("Delta metrics:")
    print(pd.DataFrame(eval_summary["delta_all"]).to_string(index=False))
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
