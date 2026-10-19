from __future__ import annotations

from typing import Any

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    layers: int = 2,
    dropout: float = 0.1,
    final_activation: bool = True,
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    blocks: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(layers - 1):
        blocks.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ]
        )
        in_dim = hidden_dim
    blocks.append(nn.Linear(in_dim, output_dim))
    if final_activation:
        blocks.extend([nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(output_dim)])
    return nn.Sequential(*blocks)


class WindowEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_window: int,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_window = int(input_window)
        self.feature_dim = int(feature_dim)
        self.encoder = make_mlp(
            self.input_window * self.feature_dim,
            hidden_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            final_activation=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape={tuple(x.shape)}")
        return self.encoder(x.flatten(start_dim=1))


class LSTMSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_layers = int(layers)
        self.lstm = nn.LSTM(
            input_size=int(feature_dim),
            hidden_size=int(hidden_dim),
            num_layers=lstm_layers,
            batch_first=True,
            dropout=float(dropout) if lstm_layers > 1 else 0.0,
        )
        self.sequence_norm = nn.LayerNorm(int(hidden_dim))
        self.token_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape={tuple(x.shape)}")
        sequence, (hidden, _) = self.lstm(x)
        sequence = self.sequence_norm(sequence)
        token = self.token_norm(hidden[-1])
        return sequence, token


class LinearSequenceProjection(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        token_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.projection = nn.Linear(self.feature_dim, int(token_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape={tuple(x.shape)}")
        return self.norm(self.dropout(self.projection(x)))


class TemporalSummary(nn.Module):
    def __init__(self, *, token_dim: int) -> None:
        super().__init__()
        self.summary = nn.Linear(int(token_dim) * 2, int(token_dim))
        self.norm = nn.LayerNorm(int(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, token], got shape={tuple(x.shape)}")
        latest = x[:, -1, :]
        mean = x.mean(dim=1)
        return self.norm(self.summary(torch.cat([latest, mean], dim=-1)))


class DomainRawSummary(nn.Module):
    """Compact separated raw-window summary: latest, mean, and end-start trend."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape={tuple(x.shape)}")
        latest = x[:, -1, :]
        mean = x.mean(dim=1)
        trend = x[:, -1, :] - x[:, 0, :]
        return torch.cat([latest, mean, trend], dim=-1)


class GlobalModeEncoder(nn.Module):
    def __init__(
        self,
        *,
        mode_window: int,
        mode_feature_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if mode_feature_dim % 2 != 0:
            raise ValueError("mode_context feature dimension must be mean/std pairs.")
        self.mode_window = int(mode_window)
        self.n_modes = int(mode_feature_dim // 2)
        self.mode_encoder = make_mlp(
            self.mode_window * 2,
            hidden_dim,
            hidden_dim,
            layers=2,
            dropout=dropout,
            final_activation=True,
        )
        self.mode_embedding = nn.Parameter(torch.zeros(self.n_modes, hidden_dim))
        nn.init.normal_(self.mode_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected mode_context as [batch, time, features], got {tuple(x.shape)}")
        batch_size, window, features = x.shape
        if int(window) != self.mode_window:
            raise ValueError(f"Expected mode_context window={self.mode_window}, got {window}")
        if int(features) != self.n_modes * 2:
            raise ValueError(f"Expected {self.n_modes * 2} mode features, got {features}")
        tokens = x.view(batch_size, self.mode_window, self.n_modes, 2)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size, self.n_modes, self.mode_window * 2)
        encoded = self.mode_encoder(tokens.reshape(batch_size * self.n_modes, self.mode_window * 2))
        encoded = encoded.view(batch_size, self.n_modes, -1)
        return encoded + self.mode_embedding.unsqueeze(0)


class ResidualPredictionHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decoder = make_mlp(
            input_dim,
            hidden_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            final_activation=True,
        )
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(x)
        skipped = self.skip(x)
        return self.out(self.norm(decoded + skipped))


class RawResidualPredictionHead(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        raw_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_decoder = make_mlp(
            context_dim,
            hidden_dim,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            final_activation=True,
        )
        self.context_skip = nn.Linear(context_dim, hidden_dim)
        self.raw_skip = nn.Linear(raw_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, context: torch.Tensor, raw_summary: torch.Tensor) -> torch.Tensor:
        decoded = self.context_decoder(context)
        skipped = self.context_skip(context) + self.raw_skip(raw_summary)
        return self.out(self.norm(decoded + skipped))


class ResidualFeatureDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decoder = make_mlp(
            input_dim,
            hidden_dim,
            output_dim,
            layers=layers,
            dropout=dropout,
            final_activation=True,
        )
        self.skip = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.decoder(x) + self.skip(x))


class ContextTokenLSTMDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        context_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_token = nn.Linear(int(context_dim), int(input_dim))
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 1, int(input_dim)))
        nn.init.normal_(self.context_type_embedding, mean=0.0, std=0.02)
        lstm_layers = int(layers)
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=lstm_layers,
            batch_first=True,
            dropout=float(dropout) if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, history_sequence: torch.Tensor, coupled_state: torch.Tensor) -> torch.Tensor:
        if history_sequence.ndim != 3:
            raise ValueError(
                f"Expected history_sequence as [batch, time, features], got {tuple(history_sequence.shape)}"
            )
        if coupled_state.ndim != 2:
            raise ValueError(f"Expected coupled_state as [batch, features], got {tuple(coupled_state.shape)}")
        context_token = self.context_token(coupled_state).unsqueeze(1) + self.context_type_embedding
        decoder_input = torch.cat([history_sequence, context_token], dim=1)
        decoded_sequence, _ = self.lstm(decoder_input)
        return self.norm(self.dropout(decoded_sequence[:, -1, :]))


class RawHistoryGRUDecoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        gru_layers = int(layers)
        self.gru = nn.GRU(
            input_size=int(feature_dim),
            hidden_size=int(hidden_dim),
            num_layers=gru_layers,
            batch_first=True,
            dropout=float(dropout) if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected raw history as [batch, time, features], got {tuple(x.shape)}")
        sequence, _ = self.gru(x)
        return self.norm(self.dropout(sequence[:, -1, :]))


class SameDomainGatedFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate = make_mlp(
            int(hidden_dim) * 2,
            int(hidden_dim),
            2,
            layers=2,
            dropout=dropout,
            final_activation=False,
        )

    def forward(self, model_path: torch.Tensor, raw_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(torch.cat([model_path, raw_path], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        fused = model_path * weights[:, 0:1] + raw_path * weights[:, 1:2]
        return fused, weights


class DomainGatedFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate = make_mlp(
            int(hidden_dim) * 2,
            int(hidden_dim),
            2,
            layers=2,
            dropout=dropout,
            final_activation=False,
        )

    def forward(self, land: torch.Tensor, forcing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(torch.cat([land, forcing], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        fused = torch.cat(
            [
                land * weights[:, 0:1],
                forcing * weights[:, 1:2],
            ],
            dim=-1,
        )
        return fused, weights


class ARSEMLPCoupledModel(nn.Module):
    """
    ARSE forcing-land coupled model.

    The default backbone keeps compatibility with older checkpoints. Newer
    two-encoder backbones encode forcing and land windows separately, mix the
    two domain tokens with a Transformer encoder, then either build a shared
    all-token or decode the two coupled domain tokens directly.
    """

    def __init__(
        self,
        *,
        input_window: int,
        mode_window: int,
        output_horizon: int,
        input_dims: dict[str, int],
        feature_groups: dict[str, list[str]],
        main_targets: list[str],
        main_head_groups: list[list[str]] | None,
        main_head_raw_variables: list[list[str]] | None,
        main_output_dim: int,
        aux_output_dim: int,
        process_token_groups: dict[str, list[str]] | None = None,
        forcing_process_tokens: list[str] | None = None,
        land_process_tokens: list[str] | None = None,
        hidden_dim: int = 128,
        token_dim: int = 128,
        encoder_layers: int = 2,
        domain_transformer_layers: int = 1,
        decoder_layers: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        attention_heads: int = 4,
        backbone: str = "unified_domain_transformer",
        raw_residual_mode: str = "summary",
    ) -> None:
        super().__init__()
        required = ("atm_total", "land_total", "atm_anom", "land_anom")
        missing = [name for name in required if name not in input_dims]
        if missing:
            raise KeyError(f"Missing input dimensions for groups: {missing}")

        forcing_dim = int(input_dims["atm_total"]) + int(input_dims["atm_anom"])
        land_dim = int(input_dims["land_total"]) + int(input_dims["land_anom"])
        mode_feature_dim = int(input_dims.get("mode_context", 0))
        if forcing_dim <= 0:
            raise ValueError("Forcing branch has zero features.")
        if land_dim <= 0:
            raise ValueError("Land branch has zero features.")

        self.input_window = int(input_window)
        self.output_horizon = int(output_horizon)
        if self.output_horizon <= 0:
            raise ValueError("output_horizon must be a positive integer.")
        self.backbone = str(backbone)
        self.raw_residual_mode = str(raw_residual_mode)
        if self.backbone not in {
            "unified_domain_transformer",
            "two_encoder_all_token",
            "two_encoder_gated_decoders",
            "process_token_gated_decoders",
            "process_token_gated_decoders_with_raw_gru",
            "process_token_lstm_context_token_decoders",
            "process_token_lstm_encoder_mlp_decoders",
            "process_token_mlp_encoder_lstm_context_decoders",
        }:
            raise ValueError(
                "Unsupported model.backbone. Expected one of "
                "{'unified_domain_transformer', 'two_encoder_all_token', "
                "'two_encoder_gated_decoders', 'process_token_gated_decoders', "
                "'process_token_gated_decoders_with_raw_gru', "
                "'process_token_lstm_context_token_decoders', "
                "'process_token_lstm_encoder_mlp_decoders', "
                "'process_token_mlp_encoder_lstm_context_decoders'}, "
                f"got {self.backbone!r}."
            )
        if self.raw_residual_mode not in {"summary", "sequence"}:
            raise ValueError(
                "Unsupported model.raw_residual_mode. Expected one of "
                "{'summary', 'sequence'}, "
                f"got {self.raw_residual_mode!r}."
            )
        land_feature_names = list(feature_groups.get("land_total", [])) + list(feature_groups.get("land_anom", []))
        forcing_feature_names = list(feature_groups.get("atm_total", [])) + list(feature_groups.get("atm_anom", []))
        self.raw_feature_names = land_feature_names + forcing_feature_names
        self.raw_feature_index_by_name = self._build_raw_feature_index_by_name(self.raw_feature_names)

        if self.backbone == "unified_domain_transformer":
            self.unified_encoder = WindowEncoder(
                input_window=input_window,
                feature_dim=forcing_dim + land_dim,
                hidden_dim=token_dim,
                layers=encoder_layers,
                dropout=dropout,
            )
            self.land_token_adapter = make_mlp(
                token_dim,
                hidden_dim,
                token_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.forcing_token_adapter = make_mlp(
                token_dim,
                hidden_dim,
                token_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
        elif self.backbone in {"two_encoder_all_token", "two_encoder_gated_decoders"}:
            self.forcing_encoder = WindowEncoder(
                input_window=input_window,
                feature_dim=forcing_dim,
                hidden_dim=token_dim,
                layers=encoder_layers,
                dropout=dropout,
            )
        else:
            resolved_process_groups = self._resolve_process_token_groups(
                process_token_groups=process_token_groups,
                raw_feature_index_by_name=self.raw_feature_index_by_name,
            )
            self.process_token_names = list(resolved_process_groups)
            self.process_token_index_by_name = {
                name: idx
                for idx, name in enumerate(self.process_token_names)
            }
            self.process_token_index_buffer_names: list[str] = []
            self.process_token_encoders = nn.ModuleList()
            for idx, (token_name, raw_names) in enumerate(resolved_process_groups.items()):
                raw_indices = [self.raw_feature_index_by_name[name] for name in raw_names]
                buffer_name = f"process_token_{idx}_raw_feature_indices"
                self.register_buffer(
                    buffer_name,
                    torch.as_tensor(raw_indices, dtype=torch.long),
                    persistent=False,
                )
                self.process_token_index_buffer_names.append(buffer_name)
                if self.backbone in {
                    "process_token_lstm_context_token_decoders",
                    "process_token_lstm_encoder_mlp_decoders",
                }:
                    self.process_token_encoders.append(
                        LSTMSequenceEncoder(
                            feature_dim=len(raw_indices),
                            hidden_dim=token_dim,
                            layers=lstm_layers,
                            dropout=dropout,
                        )
                    )
                else:
                    self.process_token_encoders.append(
                        WindowEncoder(
                            input_window=input_window,
                            feature_dim=len(raw_indices),
                            hidden_dim=token_dim,
                            layers=encoder_layers,
                            dropout=dropout,
                        )
                    )
            self.forcing_process_token_names = self._resolve_process_token_selection(
                selection=forcing_process_tokens,
                available=self.process_token_index_by_name,
                default=["aod", "meteo", "thermodynamic_energy"],
                selection_name="model.forcing_process_tokens",
                    )
            self.land_process_token_names = self._resolve_process_token_selection(
                selection=land_process_tokens,
                available=self.process_token_index_by_name,
                default=["soil_thermal", "thermodynamic_energy"],
                selection_name="model.land_process_tokens",
            )
            self.register_buffer(
                "forcing_process_token_indices",
                torch.as_tensor(
                    [self.process_token_index_by_name[name] for name in self.forcing_process_token_names],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "land_process_token_indices",
                torch.as_tensor(
                    [self.process_token_index_by_name[name] for name in self.land_process_token_names],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            if self.backbone == "process_token_mlp_encoder_lstm_context_decoders":
                self.forcing_history_projection = LinearSequenceProjection(
                    feature_dim=forcing_dim,
                    token_dim=token_dim,
                    dropout=dropout,
                )
                self.land_history_projection = LinearSequenceProjection(
                    feature_dim=land_dim,
                    token_dim=token_dim,
                    dropout=dropout,
                )
            self.land_encoder = WindowEncoder(
                input_window=input_window,
                feature_dim=land_dim,
                hidden_dim=token_dim,
                layers=encoder_layers,
                dropout=dropout,
            )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.domain_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(domain_transformer_layers),
        )
        self.domain_transformer_norm = nn.LayerNorm(token_dim)

        if self.backbone == "unified_domain_transformer":
            self.context_projection = make_mlp(
                token_dim * 2,
                hidden_dim,
                hidden_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.forcing_decoder = ResidualFeatureDecoder(
                input_dim=token_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                layers=decoder_layers,
                dropout=dropout,
            )
            aux_head_input_dim = hidden_dim * 2 + token_dim * 2
        elif self.backbone == "two_encoder_all_token":
            self.all_token_projection = make_mlp(
                token_dim * 2,
                hidden_dim,
                hidden_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.land_decoder = ResidualFeatureDecoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                layers=decoder_layers,
                dropout=dropout,
            )
            self.forcing_decoder = ResidualFeatureDecoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                layers=decoder_layers,
                dropout=dropout,
            )
            aux_head_input_dim = hidden_dim * 2
        elif self.backbone == "two_encoder_gated_decoders":
            if mode_feature_dim > 0:
                raise ValueError(
                    "mode guidance is not wired for backbone='two_encoder_gated_decoders'."
                )
            self.land_decoder = ResidualFeatureDecoder(
                input_dim=token_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                layers=decoder_layers,
                dropout=dropout,
            )
            self.forcing_decoder = ResidualFeatureDecoder(
                input_dim=token_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                layers=decoder_layers,
                dropout=dropout,
            )
            aux_head_input_dim = hidden_dim * 2
        else:
            if mode_feature_dim > 0:
                raise ValueError(
                    f"mode guidance is not wired for backbone={self.backbone!r}."
                )
            self.land_process_fusion = make_mlp(
                token_dim * len(self.land_process_token_names),
                hidden_dim,
                token_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.forcing_process_fusion = make_mlp(
                token_dim * len(self.forcing_process_token_names),
                hidden_dim,
                token_dim,
                layers=2,
                dropout=dropout,
                final_activation=True,
            )
            if self.backbone == "process_token_lstm_context_token_decoders":
                self.land_history_fusion = make_mlp(
                    token_dim * len(self.land_process_token_names),
                    hidden_dim,
                    token_dim,
                    layers=2,
                    dropout=dropout,
                    final_activation=True,
                )
                self.forcing_history_fusion = make_mlp(
                    token_dim * len(self.forcing_process_token_names),
                    hidden_dim,
                    token_dim,
                    layers=2,
                    dropout=dropout,
                    final_activation=True,
                )
                self.land_decoder = ContextTokenLSTMDecoder(
                    input_dim=token_dim,
                    context_dim=token_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
                self.forcing_decoder = ContextTokenLSTMDecoder(
                    input_dim=token_dim,
                    context_dim=token_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
            elif self.backbone == "process_token_mlp_encoder_lstm_context_decoders":
                self.land_decoder = ContextTokenLSTMDecoder(
                    input_dim=token_dim,
                    context_dim=token_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
                self.forcing_decoder = ContextTokenLSTMDecoder(
                    input_dim=token_dim,
                    context_dim=token_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
            else:
                self.land_decoder = ResidualFeatureDecoder(
                    input_dim=token_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    layers=decoder_layers,
                    dropout=dropout,
                )
                self.forcing_decoder = ResidualFeatureDecoder(
                    input_dim=token_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    layers=decoder_layers,
                    dropout=dropout,
                )
            if self.backbone == "process_token_gated_decoders_with_raw_gru":
                self.land_raw_gru_decoder = RawHistoryGRUDecoder(
                    feature_dim=land_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
                self.forcing_raw_gru_decoder = RawHistoryGRUDecoder(
                    feature_dim=forcing_dim,
                    hidden_dim=hidden_dim,
                    layers=lstm_layers,
                    dropout=dropout,
                )
                self.land_decoder_fusion_gate = SameDomainGatedFusion(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                self.forcing_decoder_fusion_gate = SameDomainGatedFusion(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            aux_head_input_dim = hidden_dim * 2
        self.land_raw_summary = DomainRawSummary()
        self.forcing_raw_summary = DomainRawSummary()

        self.use_mode_guidance = mode_feature_dim > 0
        if self.use_mode_guidance:
            self.global_mode_encoder = GlobalModeEncoder(
                mode_window=mode_window,
                mode_feature_dim=mode_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.local_to_mode_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mode_gate = make_mlp(
                hidden_dim * 2,
                hidden_dim,
                hidden_dim,
                layers=2,
                dropout=dropout,
                final_activation=False,
            )
            self.mode_guidance_norm = nn.LayerNorm(hidden_dim)

        self.main_targets = list(main_targets)
        self.main_target_index = {name: idx for idx, name in enumerate(self.main_targets)}
        self.raw_summary_index_by_name = self._build_raw_summary_index_by_name(
            feature_groups=feature_groups,
            input_window=self.input_window,
            raw_residual_mode=self.raw_residual_mode,
        )
        all_raw_names = list(self.raw_summary_index_by_name)
        self.main_head_groups = self._resolve_main_head_groups(
            self.main_targets,
            main_head_groups,
        )
        self.main_head_raw_variables = self._resolve_main_head_raw_variables(
            self.main_head_groups,
            main_head_raw_variables,
            all_raw_names,
            self.raw_summary_index_by_name,
        )
        self.main_head_output_indices = [
            [self.main_target_index[name] for name in group]
            for group in self.main_head_groups
        ]
        self.main_head_context_kinds = [
            (
                "context"
                if self.backbone == "unified_domain_transformer"
                else (
                    (
                        "gated_domain"
                    )
                    if self.backbone in {
                        "two_encoder_gated_decoders",
                        "process_token_gated_decoders",
                        "process_token_gated_decoders_with_raw_gru",
                        "process_token_lstm_context_token_decoders",
                        "process_token_lstm_encoder_mlp_decoders",
                        "process_token_mlp_encoder_lstm_context_decoders",
                    }
                    else self._infer_main_head_context_kind(group)
                )
            )
            for group in self.main_head_groups
        ]
        self.main_head_context_dims = [
            self._main_head_context_dim(kind=kind, hidden_dim=hidden_dim)
            for kind in self.main_head_context_kinds
        ]
        self.main_head_raw_index_buffer_names: list[str] = []
        for head_idx, raw_names in enumerate(self.main_head_raw_variables):
            raw_indices = [
                index
                for name in raw_names
                for index in self.raw_summary_index_by_name[name]
            ]
            buffer_name = f"main_head_{head_idx}_raw_indices"
            self.register_buffer(
                buffer_name,
                torch.as_tensor(raw_indices, dtype=torch.long),
                persistent=False,
            )
            self.main_head_raw_index_buffer_names.append(buffer_name)
        if self.backbone in {
            "two_encoder_gated_decoders",
            "process_token_gated_decoders",
            "process_token_gated_decoders_with_raw_gru",
            "process_token_lstm_context_token_decoders",
            "process_token_lstm_encoder_mlp_decoders",
            "process_token_mlp_encoder_lstm_context_decoders",
        }:
            self.main_domain_gates = nn.ModuleList(
                [
                    DomainGatedFusion(hidden_dim=hidden_dim, dropout=dropout)
                    for _ in self.main_head_groups
                ]
            )
            self.aux_domain_gate = DomainGatedFusion(hidden_dim=hidden_dim, dropout=dropout)
        self.main_heads = nn.ModuleList(
            [
                RawResidualPredictionHead(
                    context_dim=context_dim,
                    raw_dim=sum(len(self.raw_summary_index_by_name[name]) for name in raw_names),
                    hidden_dim=hidden_dim,
                    output_dim=self.output_horizon * len(group),
                    layers=decoder_layers,
                    dropout=dropout,
                )
                for group, raw_names, context_dim in zip(
                    self.main_head_groups,
                    self.main_head_raw_variables,
                    self.main_head_context_dims,
                )
            ]
        )
        self.aux_head = ResidualPredictionHead(
            input_dim=aux_head_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_horizon * aux_output_dim,
            layers=decoder_layers,
            dropout=dropout,
        )

    @staticmethod
    def _resolve_main_head_groups(
        main_targets: list[str],
        main_head_groups: list[list[str]] | None,
    ) -> list[list[str]]:
        if not main_head_groups:
            return [list(main_targets)]
        groups = [list(group) for group in main_head_groups]
        flattened = [name for group in groups for name in group]
        unknown = [name for name in flattened if name not in main_targets]
        if unknown:
            raise ValueError(f"model.main_head_groups contains unknown targets: {unknown}")
        if len(flattened) != len(set(flattened)):
            duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
            raise ValueError(f"model.main_head_groups contains duplicate targets: {duplicates}")
        if set(flattened) != set(main_targets):
            missing = [name for name in main_targets if name not in flattened]
            raise ValueError(
                "model.main_head_groups must contain every main target exactly once "
                f"with no missing targets. missing={missing}, "
                f"main_targets={main_targets}, flattened_groups={flattened}"
            )
        return groups

    @staticmethod
    def _build_raw_feature_index_by_name(raw_feature_names: list[str]) -> dict[str, int]:
        index_by_name: dict[str, int] = {}
        for idx, name in enumerate(raw_feature_names):
            if name in index_by_name:
                raise ValueError(f"Duplicate raw feature variable name: {name}")
            index_by_name[name] = idx
        return index_by_name

    @staticmethod
    def _resolve_process_token_groups(
        *,
        process_token_groups: dict[str, list[str]] | None,
        raw_feature_index_by_name: dict[str, int],
    ) -> dict[str, list[str]]:
        if not process_token_groups:
            raise ValueError(
                "model.process_token_groups is required when "
                "model.backbone='process_token_gated_decoders'."
            )
        resolved: dict[str, list[str]] = {}
        for token_name, raw_names in process_token_groups.items():
            if not raw_names:
                raise ValueError(f"Process token group {token_name!r} is empty.")
            missing = [name for name in raw_names if name not in raw_feature_index_by_name]
            if missing:
                raise ValueError(
                    f"Process token group {token_name!r} contains variables that are "
                    f"not available in model inputs: {missing}"
                )
            resolved[str(token_name)] = list(raw_names)
        return resolved

    @staticmethod
    def _resolve_process_token_selection(
        *,
        selection: list[str] | None,
        available: dict[str, int],
        default: list[str],
        selection_name: str,
    ) -> list[str]:
        names = list(selection) if selection else list(default)
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"{selection_name} contains unknown process tokens: {missing}")
        if not names:
            raise ValueError(f"{selection_name} cannot be empty.")
        return names

    @staticmethod
    def _infer_main_head_context_kind(group: list[str]) -> str:
        targets = set(group)
        if targets.issubset({"swvl1", "stl1"}):
            return "soil"
        if targets.issubset({"t2m", "skt"}):
            return "temperature"
        if targets.issubset({"e_pos", "evavt_pos", "eb_pos"}):
            return "et"
        return "all"

    @staticmethod
    def _main_head_context_dim(*, kind: str, hidden_dim: int) -> int:
        if kind == "context":
            return int(hidden_dim)
        if kind == "soil":
            return int(hidden_dim) * 2
        if kind == "gated_domain":
            return int(hidden_dim) * 2
        if kind in {"temperature", "et", "all"}:
            return int(hidden_dim) * 3
        raise ValueError(f"Unsupported main head context kind: {kind}")

    @staticmethod
    def _build_main_head_context(
        *,
        kind: str,
        context: torch.Tensor,
        land_decoded: torch.Tensor | None,
        forcing_decoded: torch.Tensor,
    ) -> torch.Tensor:
        if kind == "context":
            return context
        if land_decoded is None:
            raise RuntimeError("land_decoded is required by the two_encoder_all_token backbone.")
        if kind == "soil":
            return torch.cat([context, land_decoded], dim=-1)
        if kind in {"temperature", "et", "all"}:
            return torch.cat([context, forcing_decoded, land_decoded], dim=-1)
        raise ValueError(f"Unsupported main head context kind: {kind}")

    @staticmethod
    def _build_raw_summary_index_by_name(
        *,
        feature_groups: dict[str, list[str]],
        input_window: int,
        raw_residual_mode: str,
    ) -> dict[str, list[int]]:
        land_names = list(feature_groups.get("land_total", [])) + list(feature_groups.get("land_anom", []))
        forcing_names = list(feature_groups.get("atm_total", [])) + list(feature_groups.get("atm_anom", []))
        raw_index_by_name: dict[str, list[int]] = {}

        if raw_residual_mode == "summary":
            for idx, name in enumerate(land_names):
                if name in raw_index_by_name:
                    raise ValueError(f"Duplicate raw residual variable name: {name}")
                raw_index_by_name[name] = [idx, len(land_names) + idx, 2 * len(land_names) + idx]
            forcing_offset = 3 * len(land_names)
            for idx, name in enumerate(forcing_names):
                if name in raw_index_by_name:
                    raise ValueError(f"Duplicate raw residual variable name: {name}")
                raw_index_by_name[name] = [
                    forcing_offset + idx,
                    forcing_offset + len(forcing_names) + idx,
                    forcing_offset + 2 * len(forcing_names) + idx,
                ]
            return raw_index_by_name

        raw_names = land_names + forcing_names
        raw_feature_dim = len(raw_names)
        for idx, name in enumerate(raw_names):
            if name in raw_index_by_name:
                raise ValueError(f"Duplicate raw residual variable name: {name}")
            raw_index_by_name[name] = [
                time_idx * raw_feature_dim + idx
                for time_idx in range(int(input_window))
            ]
        return raw_index_by_name

    @staticmethod
    def _resolve_main_head_raw_variables(
        main_head_groups: list[list[str]],
        main_head_raw_variables: list[list[str]] | None,
        all_raw_names: list[str],
        raw_summary_index_by_name: dict[str, list[int]],
    ) -> list[list[str]]:
        if not main_head_raw_variables:
            return [list(all_raw_names) for _ in main_head_groups]
        if len(main_head_raw_variables) != len(main_head_groups):
            raise ValueError(
                "model.main_head_raw_variables must have the same number of "
                "groups as model.main_head_groups."
            )
        resolved = [list(group) for group in main_head_raw_variables]
        missing = [
            name
            for group in resolved
            for name in group
            if name not in raw_summary_index_by_name
        ]
        if missing:
            raise ValueError(
                "model.main_head_raw_variables contains variables that are not "
                f"available in raw summaries: {missing}"
            )
        empty = [idx for idx, group in enumerate(resolved) if not group]
        if empty:
            raise ValueError(f"Every main head needs at least one raw residual variable; empty heads={empty}")
        return resolved

    @classmethod
    def from_config(cls, config: dict[str, Any], input_dims: dict[str, int]) -> "ARSEMLPCoupledModel":
        model_cfg = config["model"]
        main_targets = list(config["variables"]["main_targets"])
        feature_groups = {
            key: list(config["variables"].get(key, []))
            for key in ("atm_total", "land_total", "atm_anom", "land_anom")
        }
        return cls(
            input_window=int(config["data"]["input_window"]),
            mode_window=int(config["data"].get("mode_window", config["data"]["input_window"])),
            output_horizon=int(config["data"].get("output_horizon", 1)),
            input_dims=input_dims,
            feature_groups=feature_groups,
            main_targets=main_targets,
            main_head_groups=model_cfg.get("main_head_groups"),
            main_head_raw_variables=model_cfg.get("main_head_raw_variables"),
            main_output_dim=len(main_targets),
            aux_output_dim=len(config["variables"]["aux_targets"]),
            process_token_groups=model_cfg.get("process_token_groups"),
            forcing_process_tokens=model_cfg.get("forcing_process_tokens"),
            land_process_tokens=model_cfg.get("land_process_tokens"),
            hidden_dim=int(model_cfg["hidden_dim"]),
            token_dim=int(model_cfg["token_dim"]),
            encoder_layers=int(model_cfg["encoder_layers"]),
            domain_transformer_layers=int(model_cfg.get("domain_transformer_layers", 1)),
            decoder_layers=int(model_cfg["decoder_layers"]),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            dropout=float(model_cfg["dropout"]),
            attention_heads=int(model_cfg["attention_heads"]),
            backbone=str(model_cfg.get("backbone", "unified_domain_transformer")),
            raw_residual_mode=str(model_cfg.get("raw_residual_mode", "summary")),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        forcing_input = torch.cat([batch["atm_total"], batch["atm_anom"]], dim=-1)
        land_input = torch.cat([batch["land_total"], batch["land_anom"]], dim=-1)

        process_tokens = None
        process_sequence_tokens = None
        if self.backbone == "unified_domain_transformer":
            unified_repr = self.unified_encoder(torch.cat([forcing_input, land_input], dim=-1))
            forcing_repr = self.forcing_token_adapter(unified_repr)
            land_repr = self.land_token_adapter(unified_repr)
            forcing_tokens = forcing_repr.unsqueeze(1)
            land_tokens = land_repr.unsqueeze(1)
            domain_tokens = torch.cat([land_tokens, forcing_tokens], dim=1)
        elif self.backbone in {"two_encoder_all_token", "two_encoder_gated_decoders"}:
            forcing_repr = self.forcing_encoder(forcing_input)
            land_repr = self.land_encoder(land_input)
            unified_repr = torch.cat([forcing_repr, land_repr], dim=-1)
            forcing_tokens = forcing_repr.unsqueeze(1)
            land_tokens = land_repr.unsqueeze(1)
            domain_tokens = torch.cat([land_tokens, forcing_tokens], dim=1)
        else:
            combined_input = torch.cat([land_input, forcing_input], dim=-1)
            process_token_parts = []
            process_sequence_parts = []
            for encoder, buffer_name in zip(self.process_token_encoders, self.process_token_index_buffer_names):
                process_input = combined_input.index_select(
                    dim=-1,
                    index=getattr(self, buffer_name),
                )
                if self.backbone in {
                    "process_token_lstm_context_token_decoders",
                    "process_token_lstm_encoder_mlp_decoders",
                }:
                    process_sequence, process_token = encoder(process_input)
                    process_sequence_parts.append(process_sequence)
                    process_token_parts.append(process_token)
                else:
                    process_token_parts.append(encoder(process_input))
            process_tokens = torch.stack(process_token_parts, dim=1)
            process_sequence_tokens = (
                torch.stack(process_sequence_parts, dim=1)
                if process_sequence_parts
                else None
            )
            unified_repr = process_tokens.flatten(start_dim=1)
            forcing_repr = None
            land_repr = None
            forcing_tokens = None
            land_tokens = None
            domain_tokens = process_tokens

        domain_coupled_tokens = self.domain_transformer_norm(self.domain_transformer(domain_tokens))
        land_history_sequence = None
        forcing_history_sequence = None
        if self.backbone in {
            "process_token_gated_decoders",
            "process_token_gated_decoders_with_raw_gru",
            "process_token_lstm_context_token_decoders",
            "process_token_lstm_encoder_mlp_decoders",
            "process_token_mlp_encoder_lstm_context_decoders",
        }:
            process_coupled_tokens = domain_coupled_tokens
            land_coupled_tokens = process_coupled_tokens.index_select(
                dim=1,
                index=self.land_process_token_indices,
            )
            forcing_coupled_tokens = process_coupled_tokens.index_select(
                dim=1,
                index=self.forcing_process_token_indices,
            )
            land_coupled = self.land_process_fusion(land_coupled_tokens.flatten(start_dim=1))
            forcing_coupled = self.forcing_process_fusion(forcing_coupled_tokens.flatten(start_dim=1))
            if self.backbone == "process_token_lstm_context_token_decoders":
                if process_sequence_tokens is None:
                    raise RuntimeError(
                        "process_sequence_tokens is required by "
                        "process_token_lstm_context_token_decoders."
                    )
                land_history_tokens = process_sequence_tokens.index_select(
                    dim=1,
                    index=self.land_process_token_indices,
                )
                forcing_history_tokens = process_sequence_tokens.index_select(
                    dim=1,
                    index=self.forcing_process_token_indices,
                )
                land_history_sequence = self.land_history_fusion(
                    land_history_tokens.permute(0, 2, 1, 3).flatten(start_dim=2)
                )
                forcing_history_sequence = self.forcing_history_fusion(
                    forcing_history_tokens.permute(0, 2, 1, 3).flatten(start_dim=2)
                )
            elif self.backbone == "process_token_mlp_encoder_lstm_context_decoders":
                land_history_sequence = self.land_history_projection(land_input)
                forcing_history_sequence = self.forcing_history_projection(forcing_input)
        else:
            process_coupled_tokens = None
            land_coupled_tokens = domain_coupled_tokens[:, 0:1, :]
            forcing_coupled_tokens = domain_coupled_tokens[:, 1:2, :]
            land_coupled = land_coupled_tokens.squeeze(1)
            forcing_coupled = forcing_coupled_tokens.squeeze(1)

        coupled_pair = torch.cat([land_coupled, forcing_coupled], dim=-1)
        if self.backbone == "unified_domain_transformer":
            context = self.context_projection(coupled_pair)
        elif self.backbone == "two_encoder_all_token":
            context = self.all_token_projection(coupled_pair)
        else:
            context = None
        mode_feedback = None
        mode_gate = None
        if self.use_mode_guidance:
            if context is None:
                raise RuntimeError("Mode guidance requires a shared context tensor.")
            if "mode_context" not in batch:
                raise KeyError("Model was configured with mode guidance but batch is missing mode_context.")
            mode_tokens = self.global_mode_encoder(batch["mode_context"])
            mode_feedback_token, _ = self.local_to_mode_attention(
                query=context.unsqueeze(1),
                key=mode_tokens,
                value=mode_tokens,
                need_weights=False,
            )
            mode_feedback = mode_feedback_token.squeeze(1)
            mode_gate = torch.sigmoid(self.mode_gate(torch.cat([context, mode_feedback], dim=-1)))
            context = self.mode_guidance_norm(context + mode_gate * mode_feedback)

        land_decoded_model = None
        forcing_decoded_model = None
        land_decoded_raw_gru = None
        forcing_decoded_raw_gru = None
        land_decoder_fusion_weights = None
        forcing_decoder_fusion_weights = None
        if self.backbone == "unified_domain_transformer":
            land_decoded = None
            forcing_decoded = self.forcing_decoder(torch.cat([forcing_coupled, land_coupled], dim=-1))
            aux_features = torch.cat([forcing_decoded, context, forcing_coupled, land_coupled], dim=-1)
            aux_gate_weights = None
        elif self.backbone == "two_encoder_all_token":
            land_decoded = self.land_decoder(context)
            forcing_decoded = self.forcing_decoder(context)
            aux_features = torch.cat([context, forcing_decoded], dim=-1)
            aux_gate_weights = None
        elif self.backbone == "two_encoder_gated_decoders":
            land_decoded = self.land_decoder(land_coupled)
            forcing_decoded = self.forcing_decoder(forcing_coupled)
            aux_features, aux_gate_weights = self.aux_domain_gate(land_decoded, forcing_decoded)
        elif self.backbone == "process_token_gated_decoders_with_raw_gru":
            land_decoded_model = self.land_decoder(land_coupled)
            forcing_decoded_model = self.forcing_decoder(forcing_coupled)
            land_decoded_raw_gru = self.land_raw_gru_decoder(land_input)
            forcing_decoded_raw_gru = self.forcing_raw_gru_decoder(forcing_input)
            land_decoded, land_decoder_fusion_weights = self.land_decoder_fusion_gate(
                land_decoded_model,
                land_decoded_raw_gru,
            )
            forcing_decoded, forcing_decoder_fusion_weights = self.forcing_decoder_fusion_gate(
                forcing_decoded_model,
                forcing_decoded_raw_gru,
            )
            aux_features, aux_gate_weights = self.aux_domain_gate(land_decoded, forcing_decoded)
        elif self.backbone == "process_token_lstm_context_token_decoders":
            if land_history_sequence is None or forcing_history_sequence is None:
                raise RuntimeError("LSTM decoder history sequences were not built.")
            land_decoded = self.land_decoder(land_history_sequence, land_coupled)
            forcing_decoded = self.forcing_decoder(forcing_history_sequence, forcing_coupled)
            aux_features, aux_gate_weights = self.aux_domain_gate(land_decoded, forcing_decoded)
        elif self.backbone == "process_token_mlp_encoder_lstm_context_decoders":
            if land_history_sequence is None or forcing_history_sequence is None:
                raise RuntimeError("LSTM decoder history sequences were not built.")
            land_decoded = self.land_decoder(land_history_sequence, land_coupled)
            forcing_decoded = self.forcing_decoder(forcing_history_sequence, forcing_coupled)
            aux_features, aux_gate_weights = self.aux_domain_gate(land_decoded, forcing_decoded)
        else:
            land_decoded = self.land_decoder(land_coupled)
            forcing_decoded = self.forcing_decoder(forcing_coupled)
            aux_features, aux_gate_weights = self.aux_domain_gate(land_decoded, forcing_decoded)
        if self.raw_residual_mode == "summary":
            land_raw_summary = self.land_raw_summary(land_input)
            forcing_raw_summary = self.forcing_raw_summary(forcing_input)
            main_raw_summary = torch.cat([land_raw_summary, forcing_raw_summary], dim=-1)
        else:
            land_raw_summary = land_input.flatten(start_dim=1)
            forcing_raw_summary = forcing_input.flatten(start_dim=1)
            main_raw_summary = torch.cat([land_input, forcing_input], dim=-1).flatten(start_dim=1)
        main_output_parts: list[torch.Tensor | None] = [None] * len(self.main_targets)
        main_gate_weights = []
        main_head_contexts = []
        batch_size = int(forcing_input.shape[0])
        for head_idx, head in enumerate(self.main_heads):
            if self.backbone in {
                "two_encoder_gated_decoders",
                "process_token_gated_decoders",
                "process_token_gated_decoders_with_raw_gru",
                "process_token_lstm_context_token_decoders",
                "process_token_lstm_encoder_mlp_decoders",
                "process_token_mlp_encoder_lstm_context_decoders",
            }:
                head_context, head_gate_weights = self.main_domain_gates[head_idx](
                    land_decoded,
                    forcing_decoded,
                )
                main_gate_weights.append(head_gate_weights)
                main_head_contexts.append(head_context)
            else:
                head_context = self._build_main_head_context(
                    kind=self.main_head_context_kinds[head_idx],
                    context=context,
                    land_decoded=land_decoded,
                    forcing_decoded=forcing_decoded,
                )
            head_output = head(
                head_context,
                main_raw_summary.index_select(
                    dim=-1,
                    index=getattr(self, self.main_head_raw_index_buffer_names[head_idx]),
                ),
            )
            head_output = head_output.view(
                batch_size,
                self.output_horizon,
                len(self.main_head_groups[head_idx]),
            )
            for local_idx, target_idx in enumerate(self.main_head_output_indices[head_idx]):
                main_output_parts[target_idx] = head_output[:, :, local_idx : local_idx + 1]
        if any(part is None for part in main_output_parts):
            raise RuntimeError("Internal error: not every main target received a head output.")
        main_output = torch.cat([part for part in main_output_parts if part is not None], dim=-1)
        aux_output = self.aux_head(aux_features).view(batch_size, self.output_horizon, -1)

        et_component_names = [
            name for name in ("e_pos", "evavt_pos", "eb_pos") if name in self.main_target_index
        ]
        et_components = (
            torch.stack(
                [
                    main_output[:, :, self.main_target_index[name]]
                    for name in et_component_names
                ],
                dim=-1,
            )
            if et_component_names
            else None
        )
        sm_output = None
        if "swvl1" in self.main_target_index:
            sm_idx = self.main_target_index["swvl1"]
            sm_output = main_output[:, :, sm_idx : sm_idx + 1]

        return {
            "main": main_output,
            "sm": sm_output,
            "et_components": et_components,
            "et_component_names": et_component_names,
            "aux": aux_output,
            "context": context,
            "all_token": context,
            "forcing_decoded": forcing_decoded,
            "land_decoded": land_decoded,
            "forcing_decoded_model": forcing_decoded_model,
            "land_decoded_model": land_decoded_model,
            "forcing_decoded_raw_gru": forcing_decoded_raw_gru,
            "land_decoded_raw_gru": land_decoded_raw_gru,
            "forcing_decoder_fusion_weights": forcing_decoder_fusion_weights,
            "land_decoder_fusion_weights": land_decoder_fusion_weights,
            "main_gate_weights": (
                torch.stack(main_gate_weights, dim=1)
                if main_gate_weights
                else None
            ),
            "main_head_contexts": (
                torch.stack(main_head_contexts, dim=1)
                if main_head_contexts
                else None
            ),
            "aux_gate_weights": aux_gate_weights,
            "land_raw_summary": land_raw_summary,
            "forcing_raw_summary": forcing_raw_summary,
            "main_raw_summary": main_raw_summary,
            "main_head_raw_variables": self.main_head_raw_variables,
            "main_head_context_kinds": self.main_head_context_kinds,
            "unified_repr": unified_repr,
            "forcing_repr": forcing_repr,
            "land_repr": land_repr,
            "process_token_names": getattr(self, "process_token_names", None),
            "process_tokens": process_tokens,
            "process_sequence_tokens": process_sequence_tokens,
            "process_coupled_tokens": process_coupled_tokens,
            "forcing_process_token_names": getattr(self, "forcing_process_token_names", None),
            "land_process_token_names": getattr(self, "land_process_token_names", None),
            "land_coupled": land_coupled,
            "forcing_coupled": forcing_coupled,
            "land_history_sequence": land_history_sequence,
            "forcing_history_sequence": forcing_history_sequence,
            "domain_coupled_tokens": domain_coupled_tokens,
            "land_coupled_tokens": land_coupled_tokens,
            "forcing_coupled_tokens": forcing_coupled_tokens,
            "land_gate": None,
            "forcing_gate": None,
            "mode_feedback": mode_feedback,
            "mode_gate": mode_gate,
        }
