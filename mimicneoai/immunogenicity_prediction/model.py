import collections
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = collections.Counter()
        for token in tokens:
            if isinstance(token, list):
                counter.update(token)
            else:
                counter[token] += 1

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<pad>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]


class PhyschemEmbeddingRefiner(nn.Module):
    """Condition peptide-token embeddings on global peptide properties.

    The module is deliberately peptide-local: HLA, separator and padding tokens
    are returned unchanged. Zero-initialized modulation heads make the initial
    full and no-physchem models mathematically identical.
    """

    def __init__(
        self,
        embedding_dim: int,
        x2_dim: int = 25,
        composition_dim: int = 18,
        composition_hidden_dim: int = 48,
        scalar_hidden_dim: int = 32,
        fusion_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if composition_dim <= 0 or composition_dim >= x2_dim:
            raise ValueError("composition_dim must split the physicochemical feature vector")
        self.embedding_dim = int(embedding_dim)
        self.x2_dim = int(x2_dim)
        self.composition_dim = int(composition_dim)
        scalar_dim = x2_dim - composition_dim

        self.composition_encoder = nn.Sequential(
            nn.Linear(composition_dim, composition_hidden_dim),
            nn.LayerNorm(composition_hidden_dim),
            nn.GELU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden_dim),
            nn.LayerNorm(scalar_hidden_dim),
            nn.GELU(),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(composition_hidden_dim + scalar_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.gamma = nn.Linear(embedding_dim, embedding_dim)
        self.beta = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.base_norm = nn.LayerNorm(embedding_dim)
        self.layer_scale = nn.Parameter(torch.full((embedding_dim,), float(layer_scale_init)))

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    @staticmethod
    def peptide_mask(peptide_lengths: torch.Tensor, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=peptide_lengths.device).unsqueeze(0)
        return positions < peptide_lengths.long().unsqueeze(1)

    @staticmethod
    def terminal_positions(peptide_lengths: torch.Tensor, sequence_length: int, dtype) -> torch.Tensor:
        pos = torch.arange(sequence_length, device=peptide_lengths.device, dtype=dtype).unsqueeze(0)
        denom = (peptide_lengths.to(dtype) - 1.0).clamp_min(1.0).unsqueeze(1)
        n_terminal = pos / denom
        c_terminal = (denom - pos).clamp_min(0.0) / denom
        return torch.stack((n_terminal, c_terminal), dim=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        physchem: torch.Tensor,
        peptide_lengths: torch.Tensor,
        physchem_present: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, embedding]")
        if physchem.shape[-1] != self.x2_dim:
            raise ValueError(f"physchem must have {self.x2_dim} dimensions")

        mask = self.peptide_mask(peptide_lengths, embeddings.shape[1])
        present = physchem_present.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).reshape(-1, 1, 1)
        composition = self.composition_encoder(physchem[:, : self.composition_dim])
        scalar = self.scalar_encoder(physchem[:, self.composition_dim :])
        feature_embedding = self.feature_fusion(torch.cat((composition, scalar), dim=-1))

        gamma = torch.tanh(self.gamma(feature_embedding)).unsqueeze(1)
        beta = torch.tanh(self.beta(feature_embedding)).unsqueeze(1)
        terminal = self.terminal_positions(peptide_lengths, embeddings.shape[1], embeddings.dtype)
        gate = self.gate(torch.cat((embeddings, terminal), dim=-1))
        delta = gate * (gamma * self.base_norm(embeddings) + beta)
        delta = delta * mask.unsqueeze(-1).to(delta.dtype) * present
        scaled_delta = self.layer_scale.reshape(1, 1, -1) * delta
        refined = embeddings + scaled_delta

        if not return_diagnostics:
            return refined, {}

        masked = mask.unsqueeze(-1).to(embeddings.dtype)
        token_count = masked.sum().clamp_min(1.0)
        base_norm = (embeddings.norm(dim=-1, keepdim=True) * masked).sum() / token_count
        delta_norm = (scaled_delta.norm(dim=-1, keepdim=True) * masked).sum() / token_count
        active_gate = gate.masked_select(mask.unsqueeze(-1))
        diagnostics = {
            "peptide_mask": mask,
            "delta": scaled_delta,
            "base_embeddings": embeddings,
            "refined_embeddings": refined,
            "base_embedding_norm": base_norm,
            "delta_norm": delta_norm,
            "delta_to_base_ratio": delta_norm / base_norm.clamp_min(1e-8),
            "gate_mean": active_gate.mean() if active_gate.numel() else gate.new_zeros(()),
            "gate_std": active_gate.std(unbiased=False) if active_gate.numel() else gate.new_zeros(()),
            "gate_min": active_gate.min() if active_gate.numel() else gate.new_zeros(()),
            "gate_max": active_gate.max() if active_gate.numel() else gate.new_zeros(()),
            "gamma_norm": gamma.norm(dim=-1).mean(),
            "beta_norm": beta.norm(dim=-1).mean(),
            "layer_scale_norm": self.layer_scale.norm(),
        }
        return refined, diagnostics


class GroupedPhyschemEncoder(nn.Module):
    """Encode global peptide descriptors without broadcasting them to tokens."""

    def __init__(
        self,
        x2_dim: int = 25,
        count_dim: int = 9,
        fraction_dim: int = 9,
        scalar_dim: int = 7,
        group_hidden_dim: int = 16,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        if count_dim + fraction_dim + scalar_dim != x2_dim:
            raise ValueError("Physicochemical feature groups must sum to x2_dim")
        self.x2_dim = int(x2_dim)
        self.count_dim = int(count_dim)
        self.fraction_dim = int(fraction_dim)
        self.scalar_dim = int(scalar_dim)
        self.output_dim = int(output_dim)

        def group_encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, group_hidden_dim),
                nn.LayerNorm(group_hidden_dim),
                nn.GELU(),
            )

        self.count_encoder = group_encoder(count_dim)
        self.fraction_encoder = group_encoder(fraction_dim)
        self.scalar_encoder = group_encoder(scalar_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * group_hidden_dim, 2 * output_dim),
            nn.LayerNorm(2 * output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.x2_dim:
            raise ValueError(f"Physicochemical features must have {self.x2_dim} dimensions")
        count_end = self.count_dim
        fraction_end = count_end + self.fraction_dim
        counts = self.count_encoder(features[:, :count_end])
        fractions = self.fraction_encoder(features[:, count_end:fraction_end])
        scalars = self.scalar_encoder(features[:, fraction_end:])
        return self.fusion(torch.cat((counts, fractions, scalars), dim=-1))


class GatedLateFusion(nn.Module):
    """Add a bounded physicochemical residual to the pooled pHLA representation."""

    def __init__(
        self,
        sequence_dim: int,
        x2_dim: int = 25,
        physchem_dim: int = 32,
        group_hidden_dim: int = 16,
        gate_hidden_dim: int = 64,
        dropout: float = 0.2,
        alpha_init: float = 0.1,
        alpha_max: float = 0.5,
    ):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.sequence_dim = int(sequence_dim)
        self.alpha_max = float(alpha_max)
        self.physchem_encoder = GroupedPhyschemEncoder(
            x2_dim=x2_dim,
            group_hidden_dim=group_hidden_dim,
            output_dim=physchem_dim,
            dropout=dropout,
        )
        self.sequence_norm = nn.LayerNorm(sequence_dim)
        self.delta_projection = nn.Linear(physchem_dim, sequence_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(sequence_dim + physchem_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        alpha_ratio = alpha_init / alpha_max
        self.raw_alpha = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(alpha_ratio)))))
        nn.init.zeros_(self.delta_projection.weight)

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        sequence: torch.Tensor,
        physchem: torch.Tensor,
        physchem_present: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if physchem_present is None:
            raise ValueError("gated_late requires explicit physchem_present")
        present = physchem_present.to(device=sequence.device, dtype=sequence.dtype).reshape(-1, 1)
        physchem_embedding = self.physchem_encoder(physchem) * present
        gate = self.gate(torch.cat((self.sequence_norm(sequence), physchem_embedding), dim=-1))
        delta = present * self.alpha() * gate * self.delta_projection(physchem_embedding)
        fused = sequence + delta
        if not return_diagnostics:
            return fused, {}

        base_norm = sequence.norm(dim=-1).mean()
        delta_norm = delta.norm(dim=-1).mean()
        active = physchem_present.to(dtype=torch.bool, device=sequence.device)
        active_gate = gate[active]
        diagnostics = {
            "base_representation_norm": base_norm,
            "delta_norm": delta_norm,
            "delta_to_base_ratio": delta_norm / base_norm.clamp_min(1e-8),
            "physchem_embedding_norm": physchem_embedding.norm(dim=-1).mean(),
            "gate_mean": active_gate.mean() if active_gate.numel() else gate.new_zeros(()),
            "gate_std": active_gate.std(unbiased=False) if active_gate.numel() else gate.new_zeros(()),
            "gate_min": active_gate.min() if active_gate.numel() else gate.new_zeros(()),
            "gate_max": active_gate.max() if active_gate.numel() else gate.new_zeros(()),
            "alpha": self.alpha(),
        }
        return fused, diagnostics


class AuxiliaryFiLMFusion(nn.Module):
    """Condition a pooled pHLA representation with an auxiliary-supervised branch."""

    def __init__(
        self,
        sequence_dim: int,
        output_dim: int,
        x2_dim: int = 25,
        physchem_dim: int = 32,
        group_hidden_dim: int = 16,
        dropout: float = 0.2,
        film_scale: float = 0.1,
    ):
        super().__init__()
        if film_scale <= 0.0:
            raise ValueError("film_scale must be positive")
        self.sequence_dim = int(sequence_dim)
        self.film_scale = float(film_scale)
        self.physchem_encoder = GroupedPhyschemEncoder(
            x2_dim=x2_dim,
            group_hidden_dim=group_hidden_dim,
            output_dim=physchem_dim,
            dropout=dropout,
        )
        self.sequence_norm = nn.LayerNorm(sequence_dim)
        self.film_projection = nn.Linear(physchem_dim, 2 * sequence_dim)
        self.physchem_classifier = nn.Linear(physchem_dim, output_dim)
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

    def forward(
        self,
        sequence: torch.Tensor,
        physchem: torch.Tensor,
        physchem_present: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if physchem_present is None:
            raise ValueError("film_aux requires explicit physchem_present")
        present = physchem_present.to(device=sequence.device, dtype=sequence.dtype).reshape(-1, 1)
        physchem_embedding = self.physchem_encoder(physchem) * present
        gamma, beta = self.film_projection(physchem_embedding).chunk(2, dim=-1)
        modulation = self.film_scale * (
            torch.tanh(gamma) * self.sequence_norm(sequence) + torch.tanh(beta)
        )
        modulation = modulation * present
        fused = sequence + modulation
        physchem_logits = self.physchem_classifier(physchem_embedding)
        if not return_diagnostics:
            return fused, physchem_logits, {}

        base_norm = sequence.norm(dim=-1).mean()
        modulation_norm = modulation.norm(dim=-1).mean()
        active = physchem_present.to(dtype=torch.bool, device=sequence.device)
        active_gamma = gamma[active]
        active_beta = beta[active]
        diagnostics = {
            "base_representation_norm": base_norm,
            "modulation_norm": modulation_norm,
            "modulation_to_base_ratio": modulation_norm / base_norm.clamp_min(1e-8),
            "physchem_embedding_norm": physchem_embedding.norm(dim=-1).mean(),
            "gamma_abs_mean": (
                active_gamma.abs().mean() if active_gamma.numel() else gamma.new_zeros(())
            ),
            "beta_abs_mean": (
                active_beta.abs().mean() if active_beta.numel() else beta.new_zeros(())
            ),
            "film_scale": sequence.new_tensor(self.film_scale),
        }
        return fused, physchem_logits, diagnostics


class PHLASequenceEncoder(nn.Module):
    """Common pHLA encoder used by all model-comparison architectures."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim_x1: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
        pad_idx: int,
        x2_dim: int,
        physchem_fusion: str,
        refiner_embedding_dim: int = 64,
        refiner_composition_dim: int = 18,
        refiner_composition_hidden_dim: int = 48,
        refiner_scalar_hidden_dim: int = 32,
        refiner_fusion_hidden_dim: int = 128,
        refiner_gate_hidden_dim: int = 64,
        refiner_dropout: float = 0.1,
        refiner_layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if physchem_fusion not in {
            "sequence_only",
            "legacy_concat",
            "residual",
            "embedding_refine",
            "gated_late",
            "film_aux",
        }:
            raise ValueError(f"Unsupported physicochemical fusion: {physchem_fusion}")
        self.physchem_fusion = physchem_fusion
        self.pad_idx = int(pad_idx)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if physchem_fusion in {"embedding_refine", "gated_late", "film_aux"}:
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim, refiner_embedding_dim, bias=False),
                nn.LayerNorm(refiner_embedding_dim),
                nn.GELU(),
            )
            lstm_input_dim = refiner_embedding_dim
            self.physchem_refiner = (
                PhyschemEmbeddingRefiner(
                    embedding_dim=refiner_embedding_dim,
                    x2_dim=x2_dim,
                    composition_dim=refiner_composition_dim,
                    composition_hidden_dim=refiner_composition_hidden_dim,
                    scalar_hidden_dim=refiner_scalar_hidden_dim,
                    fusion_hidden_dim=refiner_fusion_hidden_dim,
                    gate_hidden_dim=refiner_gate_hidden_dim,
                    dropout=refiner_dropout,
                    layer_scale_init=refiner_layer_scale_init,
                )
                if physchem_fusion == "embedding_refine"
                else None
            )
        else:
            self.embedding_projection = nn.Identity()
            self.physchem_refiner = None
            lstm_input_dim = embedding_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim_x1,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim_x1 * (2 if bidirectional else 1)
        self.bidirectional = bidirectional

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        peptide_lengths: Optional[torch.Tensor] = None,
        physchem_present: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        token_mask = x1.ne(self.pad_idx)
        embedded = self.embedding_projection(self.embedding(x1))
        embedded = embedded * token_mask.unsqueeze(-1).to(embedded.dtype)
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.physchem_refiner is not None:
            if peptide_lengths is None:
                raise ValueError("embedding_refine requires explicit peptide_lengths")
            if physchem_present is None:
                raise ValueError("embedding_refine requires explicit physchem_present")
            embedded, diagnostics = self.physchem_refiner(
                embedded,
                x2,
                peptide_lengths,
                physchem_present,
                return_diagnostics=return_diagnostics,
            )
            # Keep separator and padding exactly zero after every transformation.
            embedded = embedded * token_mask.unsqueeze(-1).to(embedded.dtype)
            if return_diagnostics:
                diagnostics["encoder_embeddings"] = embedded

        self.lstm.flatten_parameters()
        sequence = self.dropout(embedded).transpose(0, 1)
        _, (hidden, _) = self.lstm(sequence)
        if self.bidirectional:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            pooled = hidden[-1]
        return self.dropout(pooled), diagnostics


class NewBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim_x1,
        hidden_dim_x2,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        x2_dim,
        physchem_fusion="legacy_concat",
        physchem_adapter_dim=64,
        physchem_dropout=0.2,
        late_physchem_dim=32,
        late_group_hidden_dim=16,
        late_gate_hidden_dim=64,
        late_alpha_init=0.1,
        late_alpha_max=0.5,
        film_physchem_dim=32,
        film_group_hidden_dim=16,
        film_scale=0.1,
        **refiner_kwargs,
    ):
        super().__init__()
        self.physchem_fusion = physchem_fusion
        self.encoder = PHLASequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim_x1=hidden_dim_x1,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=pad_idx,
            x2_dim=x2_dim,
            physchem_fusion=physchem_fusion,
            **refiner_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        if physchem_fusion == "sequence_only":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            classifier_dim = self.encoder.output_dim
        elif physchem_fusion == "legacy_concat":
            self.fc_x2 = nn.Linear(x2_dim, hidden_dim_x2)
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            classifier_dim = self.encoder.output_dim + hidden_dim_x2
        elif physchem_fusion == "residual":
            self.fc_x2 = None
            self.physchem_adapter = nn.Sequential(
                nn.Linear(x2_dim, physchem_adapter_dim, bias=False),
                nn.GELU(),
                nn.Dropout(physchem_dropout),
                nn.Linear(physchem_adapter_dim, self.encoder.output_dim, bias=False),
            )
            nn.init.zeros_(self.physchem_adapter[-1].weight)
            classifier_dim = self.encoder.output_dim
            self.physchem_late_fusion = None
        elif physchem_fusion == "gated_late":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = GatedLateFusion(
                sequence_dim=self.encoder.output_dim,
                x2_dim=x2_dim,
                physchem_dim=late_physchem_dim,
                group_hidden_dim=late_group_hidden_dim,
                gate_hidden_dim=late_gate_hidden_dim,
                dropout=physchem_dropout,
                alpha_init=late_alpha_init,
                alpha_max=late_alpha_max,
            )
            classifier_dim = self.encoder.output_dim
        elif physchem_fusion == "film_aux":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            self.physchem_film = AuxiliaryFiLMFusion(
                sequence_dim=self.encoder.output_dim,
                output_dim=output_dim,
                x2_dim=x2_dim,
                physchem_dim=film_physchem_dim,
                group_hidden_dim=film_group_hidden_dim,
                dropout=physchem_dropout,
                film_scale=film_scale,
            )
            classifier_dim = self.encoder.output_dim
        else:
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            self.physchem_film = None
            classifier_dim = self.encoder.output_dim
        if physchem_fusion != "film_aux":
            self.physchem_film = None
        self.fc = nn.Linear(classifier_dim, output_dim)

    @property
    def embedding(self):
        return self.encoder.embedding

    @property
    def lstm(self):
        return self.encoder.lstm

    def forward(
        self,
        x1,
        x2,
        peptide_lengths=None,
        physchem_present=None,
        return_diagnostics=False,
        return_components=False,
    ):
        hidden_x1, diagnostics = self.encoder(
            x1,
            x2,
            peptide_lengths=peptide_lengths,
            physchem_present=physchem_present,
            return_diagnostics=return_diagnostics,
        )
        if self.physchem_fusion == "sequence_only":
            pass
        elif self.physchem_fusion == "legacy_concat":
            hidden_x2 = self.dropout(self.fc_x2(x2))
            hidden_x1 = torch.cat((hidden_x1, hidden_x2), dim=1)
        elif self.physchem_fusion == "residual":
            hidden_x1 = hidden_x1 + self.physchem_adapter(x2)
        elif self.physchem_fusion == "gated_late":
            hidden_x1, late_diagnostics = self.physchem_late_fusion(
                hidden_x1,
                x2,
                physchem_present,
                return_diagnostics=return_diagnostics,
            )
            diagnostics.update(late_diagnostics)
        if self.physchem_fusion == "film_aux":
            sequence_logits = self.fc(hidden_x1)
            hidden_full, physchem_logits, film_diagnostics = self.physchem_film(
                hidden_x1,
                x2,
                physchem_present,
                return_diagnostics=return_diagnostics,
            )
            logits = self.fc(hidden_full)
            diagnostics.update(film_diagnostics)
            output = (
                {
                    "logits": logits,
                    "fused_logits": logits,
                    "sequence_logits": sequence_logits,
                    "physchem_logits": physchem_logits,
                }
                if return_components
                else logits
            )
        else:
            output = self.fc(hidden_x1)
        return (output, diagnostics) if return_diagnostics else output


class SourceAwareBiLSTM(nn.Module):
    """Shared pHLA encoder with an explicit antigen-source embedding."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim_x1,
        hidden_dim_x2,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        x2_dim,
        source_count,
        source_embedding_dim=8,
        physchem_fusion="legacy_concat",
        physchem_adapter_dim=64,
        physchem_dropout=0.2,
        late_physchem_dim=32,
        late_group_hidden_dim=16,
        late_gate_hidden_dim=64,
        late_alpha_init=0.1,
        late_alpha_max=0.5,
        film_physchem_dim=32,
        film_group_hidden_dim=16,
        film_scale=0.1,
        **refiner_kwargs,
    ):
        super().__init__()
        self.physchem_fusion = physchem_fusion
        self.encoder = PHLASequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim_x1=hidden_dim_x1,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=pad_idx,
            x2_dim=x2_dim,
            physchem_fusion=physchem_fusion,
            **refiner_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        if physchem_fusion == "sequence_only":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            phla_dim = self.encoder.output_dim
        elif physchem_fusion == "legacy_concat":
            self.fc_x2 = nn.Linear(x2_dim, hidden_dim_x2)
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            phla_dim = self.encoder.output_dim + hidden_dim_x2
        elif physchem_fusion == "residual":
            self.fc_x2 = None
            self.physchem_adapter = nn.Sequential(
                nn.Linear(x2_dim, physchem_adapter_dim, bias=False),
                nn.GELU(),
                nn.Dropout(physchem_dropout),
                nn.Linear(physchem_adapter_dim, self.encoder.output_dim, bias=False),
            )
            nn.init.zeros_(self.physchem_adapter[-1].weight)
            phla_dim = self.encoder.output_dim
            self.physchem_late_fusion = None
        elif physchem_fusion == "gated_late":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = GatedLateFusion(
                sequence_dim=self.encoder.output_dim,
                x2_dim=x2_dim,
                physchem_dim=late_physchem_dim,
                group_hidden_dim=late_group_hidden_dim,
                gate_hidden_dim=late_gate_hidden_dim,
                dropout=physchem_dropout,
                alpha_init=late_alpha_init,
                alpha_max=late_alpha_max,
            )
            phla_dim = self.encoder.output_dim
        elif physchem_fusion == "film_aux":
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            self.physchem_film = AuxiliaryFiLMFusion(
                sequence_dim=self.encoder.output_dim,
                output_dim=output_dim,
                x2_dim=x2_dim,
                physchem_dim=film_physchem_dim,
                group_hidden_dim=film_group_hidden_dim,
                dropout=physchem_dropout,
                film_scale=film_scale,
            )
            phla_dim = self.encoder.output_dim
        else:
            self.fc_x2 = None
            self.physchem_adapter = None
            self.physchem_late_fusion = None
            self.physchem_film = None
            phla_dim = self.encoder.output_dim
        if physchem_fusion != "film_aux":
            self.physchem_film = None
        self.source_embedding = nn.Embedding(source_count, source_embedding_dim)
        self.fc = nn.Linear(phla_dim + source_embedding_dim, output_dim)

    @property
    def embedding(self):
        return self.encoder.embedding

    @property
    def lstm(self):
        return self.encoder.lstm

    def forward(
        self,
        x1,
        x2,
        source_id,
        peptide_lengths=None,
        physchem_present=None,
        return_diagnostics=False,
        return_components=False,
    ):
        hidden_x1, diagnostics = self.encoder(
            x1,
            x2,
            peptide_lengths=peptide_lengths,
            physchem_present=physchem_present,
            return_diagnostics=return_diagnostics,
        )
        if self.physchem_fusion == "sequence_only":
            pass
        elif self.physchem_fusion == "legacy_concat":
            hidden_x1 = torch.cat((hidden_x1, self.dropout(self.fc_x2(x2))), dim=1)
        elif self.physchem_fusion == "residual":
            hidden_x1 = hidden_x1 + self.physchem_adapter(x2)
        elif self.physchem_fusion == "gated_late":
            hidden_x1, late_diagnostics = self.physchem_late_fusion(
                hidden_x1,
                x2,
                physchem_present,
                return_diagnostics=return_diagnostics,
            )
            diagnostics.update(late_diagnostics)
        hidden_source = self.dropout(self.source_embedding(source_id.long()))
        if self.physchem_fusion == "film_aux":
            sequence_logits = self.fc(torch.cat((hidden_x1, hidden_source), dim=1))
            hidden_full, physchem_logits, film_diagnostics = self.physchem_film(
                hidden_x1,
                x2,
                physchem_present,
                return_diagnostics=return_diagnostics,
            )
            logits = self.fc(torch.cat((hidden_full, hidden_source), dim=1))
            diagnostics.update(film_diagnostics)
            output = (
                {
                    "logits": logits,
                    "fused_logits": logits,
                    "sequence_logits": sequence_logits,
                    "physchem_logits": physchem_logits,
                }
                if return_components
                else logits
            )
        else:
            output = self.fc(torch.cat((hidden_x1, hidden_source), dim=1))
        return (output, diagnostics) if return_diagnostics else output


def build_model(vocab_size: int, pad_idx: int, model_config: Optional[Dict] = None) -> nn.Module:
    config = dict(model_config or {})
    architecture = config.pop("architecture", "phla")
    model_cls = SourceAwareBiLSTM if architecture == "source_aware" else NewBiLSTM
    defaults = {
        "vocab_size": vocab_size,
        "embedding_dim": 12,
        "hidden_dim_x1": 256,
        "hidden_dim_x2": 16,
        "output_dim": 2,
        "n_layers": 2,
        "bidirectional": True,
        "dropout": 0.5,
        "pad_idx": pad_idx,
        "x2_dim": 25,
    }
    defaults.update(config)
    return model_cls(**defaults)


def checkpoint_payload(model_path: str, device: torch.device):
    return torch.load(model_path, map_location=device)


def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def remap_legacy_encoder_keys(state_dict):
    remapped = OrderedDict()
    for key, value in state_dict.items():
        clean = key.replace("module.", "", 1)
        if clean.startswith("embedding.") or clean.startswith("lstm."):
            clean = f"encoder.{clean}"
        remapped[clean] = value
    return remapped


def load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> None:
    state_dict = extract_state_dict(checkpoint_payload(model_path, device))
    candidate_dicts = [state_dict]
    stripped = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    candidate_dicts.append(stripped)
    prefixed = OrderedDict(
        (k if k.startswith("module.") else f"module.{k}", v) for k, v in state_dict.items()
    )
    candidate_dicts.append(prefixed)
    candidate_dicts.append(remap_legacy_encoder_keys(state_dict))
    last_error = None
    for candidate in candidate_dicts:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load model weights from {model_path}. Last error: {last_error}")
