from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CounterfactualMediatorLSTM(nn.Module):
    """
    Two-layer LSTM for counterfactual mediator prediction.

    The mediator variables are not fed into the LSTM input streams. Their last
    available historical value is supplied only as a shortcut to the final head.
    """

    input_groups = ("atm_total", "land_total", "atm_anom", "land_anom")

    def __init__(
        self,
        *,
        input_dims: dict[str, int],
        mediator_dim: int,
        output_horizon: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        prediction_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if prediction_mode not in {"absolute", "delta"}:
            raise ValueError("prediction_mode must be 'absolute' or 'delta'.")
        missing = [name for name in self.input_groups if name not in input_dims]
        if missing:
            raise KeyError(f"Missing input dimensions for groups: {missing}")

        self.prediction_mode = str(prediction_mode)
        self.mediator_dim = int(mediator_dim)
        self.output_horizon = int(output_horizon)
        if self.mediator_dim <= 0:
            raise ValueError("mediator_dim must be positive.")
        if self.output_horizon <= 0:
            raise ValueError("output_horizon must be positive.")

        total_input_dim = sum(int(input_dims[group]) for group in self.input_groups)
        if total_input_dim <= 0:
            raise ValueError("At least one input feature is required.")
        self.input_dims = {group: int(input_dims[group]) for group in self.input_groups}
        self.input_dim = int(total_input_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(
            self.hidden_dim + self.mediator_dim,
            self.output_horizon * self.mediator_dim,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        input_dims: dict[str, int],
        mediator_dim: int,
        prediction_mode: str,
    ) -> "CounterfactualMediatorLSTM":
        model_cfg = config.get("mediator_model", config.get("model", {}))
        return cls(
            input_dims=input_dims,
            mediator_dim=mediator_dim,
            output_horizon=int(config["data"].get("output_horizon", 1)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            prediction_mode=prediction_mode,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pieces = []
        for group in self.input_groups:
            x = batch[group]
            if x.ndim != 3:
                raise ValueError(f"Expected {group} as [batch, time, feature], got {tuple(x.shape)}")
            pieces.append(x)
        x_all = torch.cat(pieces, dim=-1)
        x_proj = self.input_projection(x_all)
        lstm_out, _ = self.lstm(x_proj)
        last = self.output_dropout(lstm_out[:, -1, :])
        residual_base = batch["residual_base"]
        if residual_base.ndim != 2:
            raise ValueError(
                "Expected residual_base as [batch, mediator], "
                f"got {tuple(residual_base.shape)}"
            )
        head_input = torch.cat([last, residual_base], dim=-1)
        head_output = self.head(head_input).view(last.shape[0], self.output_horizon, self.mediator_dim)
        residual = residual_base[:, None, :].expand(-1, self.output_horizon, -1)
        if self.prediction_mode == "absolute":
            prediction = head_output
            delta = prediction - residual
        else:
            delta = head_output
            prediction = residual + delta
        return {
            "prediction": prediction,
            "delta": delta,
            "residual_base": residual,
        }
