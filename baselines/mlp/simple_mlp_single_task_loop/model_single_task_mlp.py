from __future__ import annotations

from typing import Any

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    *,
    layers: int,
    dropout: float,
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    blocks: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(layers)):
        blocks.extend(
            [
                nn.Linear(in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.LayerNorm(int(hidden_dim)),
            ]
        )
        in_dim = int(hidden_dim)
    return nn.Sequential(*blocks)


class SingleTaskMLPModel(nn.Module):
    """
    Pure single-task MLP baseline.

    The model uses the same four input streams as the active ARSE config, flattens
    the whole input window, and directly predicts one normalized target variable
    for one or more future leads.
    """

    input_groups = ("atm_total", "land_total", "atm_anom", "land_anom")

    def __init__(
        self,
        *,
        input_window: int,
        input_dims: dict[str, int],
        output_horizon: int = 1,
        input_variables: dict[str, list[str]] | None = None,
        target_name: str | None = None,
        excluded_input_names: list[str] | None = None,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        missing = [name for name in self.input_groups if name not in input_dims]
        if missing:
            raise KeyError(f"Missing input dimensions for groups: {missing}")

        self.input_window = int(input_window)
        self.output_horizon = int(output_horizon)
        if self.output_horizon <= 0:
            raise ValueError("output_horizon must be a positive integer.")
        self.target_name = target_name
        self.excluded_input_names = list(excluded_input_names or [])
        self.keep_indices: dict[str, list[int]] = {}
        self.input_exclusions: dict[str, list[str]] = {}
        excluded_set = set(self.excluded_input_names)
        for group in self.input_groups:
            group_dim = int(input_dims[group])
            variables = list(input_variables.get(group, [])) if input_variables is not None else []
            excluded = [name for name in variables if name in excluded_set]
            if variables:
                keep = [idx for idx, name in enumerate(variables) if name not in excluded_set]
            else:
                keep = list(range(group_dim))
            if not keep:
                raise ValueError(f"All features were excluded from input group {group!r}.")
            self.keep_indices[group] = keep
            self.input_exclusions[group] = excluded

        self.effective_input_dims = {group: len(self.keep_indices[group]) for group in self.input_groups}
        self.input_dim = int(input_window) * sum(self.effective_input_dims.values())
        self.trunk = make_mlp(
            self.input_dim,
            int(hidden_dim),
            layers=int(layers),
            dropout=float(dropout),
        )
        self.head = nn.Linear(int(hidden_dim), self.output_horizon)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        input_dims: dict[str, int],
        *,
        target_name: str | None = None,
        exclude_target_history: bool = False,
        excluded_input_names: list[str] | None = None,
    ) -> "SingleTaskMLPModel":
        model_cfg = config["model"]
        input_variables = {group: list(config["variables"].get(group, [])) for group in cls.input_groups}
        if exclude_target_history:
            excluded = list(excluded_input_names or ([target_name] if target_name is not None else []))
        else:
            excluded = []
        return cls(
            input_window=int(config["data"]["input_window"]),
            output_horizon=int(config["data"].get("output_horizon", 1)),
            input_dims=input_dims,
            input_variables=input_variables,
            target_name=target_name,
            excluded_input_names=excluded,
            hidden_dim=int(model_cfg["hidden_dim"]),
            layers=int(model_cfg.get("layers", model_cfg.get("decoder_layers", 3))),
            dropout=float(model_cfg["dropout"]),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parts = []
        for group in self.input_groups:
            x = batch[group]
            if x.ndim != 3:
                raise ValueError(f"Expected {group} as [batch, time, features], got {tuple(x.shape)}")
            keep = self.keep_indices[group]
            if len(keep) != x.shape[-1]:
                index = torch.as_tensor(keep, device=x.device, dtype=torch.long)
                x = x.index_select(dim=-1, index=index)
            parts.append(x.flatten(start_dim=1))
        features = torch.cat(parts, dim=-1)
        pred = self.head(self.trunk(features))
        return {"target": pred.view(pred.shape[0], self.output_horizon, 1)}
