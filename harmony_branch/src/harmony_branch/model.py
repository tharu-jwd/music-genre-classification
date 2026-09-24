"""Lightweight reference harmony branch and target-appropriate masked losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HarmonyBranchOutput:
    embedding: Tensor
    chroma_logits: Tensor
    chord_logits: Tensor | None
    availability: Tensor
    prediction_mask: Tensor
    pooling_weights: Tensor


class TemporalHarmonyBranch(nn.Module):
    """A compact baseline that keeps discontinuous model windows separate.

    This is a reference implementation for interface, masking, and cheap frozen-
    encoder screening. Its temporal-convolution depth is not asserted to be the
    empirically optimal harmony architecture.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        chord_classes: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (embedding_dim, "embedding_dim"),
            (hidden_dim, "hidden_dim"),
            (temporal_layers, "temporal_layers"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if chord_classes is not None and (
            not isinstance(chord_classes, int)
            or isinstance(chord_classes, bool)
            or chord_classes < 2
        ):
            raise ValueError("chord_classes must be None or an integer of at least two")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_layers = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(temporal_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(hidden_dim, embedding_dim)
        self.chroma_head = nn.Linear(embedding_dim, 12)
        self.chord_head = (
            nn.Linear(embedding_dim, chord_classes) if chord_classes is not None else None
        )

    @staticmethod
    def _validate_layout(
        sequence: Tensor,
        mask: Tensor,
        window_index: Tensor,
        windows: int,
        tokens_per_window: int,
    ) -> tuple[int, int]:
        if sequence.ndim != 3:
            raise ValueError("encoded_sequence must have shape batch x time x feature")
        batch, tokens, _ = sequence.shape
        if mask.shape != (batch, tokens) or window_index.shape != (batch, tokens):
            raise ValueError("sequence mask and window index must share batch/time axes")
        if windows < 1 or tokens_per_window < 1 or windows * tokens_per_window != tokens:
            raise ValueError("windows * tokens_per_window must equal the sequence length")
        expected = (
            torch.arange(windows, device=sequence.device)
            .view(1, windows, 1)
            .expand(batch, windows, tokens_per_window)
            .reshape(batch, tokens)
        )
        bool_mask = mask.to(dtype=torch.bool)
        if not torch.equal(window_index[bool_mask].to(expected.dtype), expected[bool_mask]):
            raise ValueError("sequence_window_index is inconsistent with the declared layout")
        if torch.any(window_index[~bool_mask] != -1):
            raise ValueError("masked sequence positions must use window index -1")
        return batch, tokens

    def forward(
        self,
        encoded_sequence: Tensor,
        sequence_mask: Tensor,
        sequence_window_index: Tensor,
        *,
        windows: int,
        tokens_per_window: int,
    ) -> HarmonyBranchOutput:
        batch, tokens = self._validate_layout(
            encoded_sequence,
            sequence_mask,
            sequence_window_index,
            windows,
            tokens_per_window,
        )
        if encoded_sequence.shape[-1] != self.input_dim:
            raise ValueError(
                f"encoded feature width must be {self.input_dim}, got {encoded_sequence.shape[-1]}"
            )
        mask = sequence_mask.to(dtype=torch.bool)
        hidden = self.input_projection(encoded_sequence)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        structured_mask = mask.reshape(batch * windows, tokens_per_window)
        context = hidden.reshape(batch * windows, tokens_per_window, self.hidden_dim)
        context = context.transpose(1, 2)
        for convolution in self.temporal_layers:
            update = F.gelu(convolution(context))
            update = self.dropout(update)
            context = (context + update) * structured_mask.unsqueeze(1).to(context.dtype)
        context = context.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)
        context = context * mask.unsqueeze(-1).to(context.dtype)

        token_embeddings = self.embedding_projection(context)
        token_embeddings = token_embeddings * mask.unsqueeze(-1).to(token_embeddings.dtype)
        chroma_logits = self.chroma_head(token_embeddings).masked_fill(~mask.unsqueeze(-1), 0)
        chord_logits = (
            self.chord_head(token_embeddings).masked_fill(~mask.unsqueeze(-1), 0)
            if self.chord_head is not None
            else None
        )
        availability = mask.any(dim=1)
        weights = mask.to(token_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        embedding = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)
        embedding = embedding * availability.unsqueeze(-1).to(embedding.dtype)
        return HarmonyBranchOutput(
            embedding=embedding,
            chroma_logits=chroma_logits,
            chord_logits=chord_logits,
            availability=availability,
            prediction_mask=mask,
            pooling_weights=weights,
        )
