"""Harmony branches: the v3 reference model and the chroma-grounded v4 model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import (
    CHROMA_MEAN_FEATURES,
    CHROMA_STD_FEATURES,
    DESCRIPTOR_GROUPS,
    HARMONY_DESCRIPTORS,
    N_HARMONY_DESCRIPTORS,
    TONNETZ_MEAN_FEATURES,
    TONNETZ_STD_FEATURES,
)
from .descriptors import (
    HarmonyTargetTransform,
    forward_transform_torch,
    inverse_transform_torch,
    tonnetz_phi,
    transform_codes,
)


@dataclass(frozen=True)
class HarmonyBranchOutput:
    embedding: Tensor
    chroma_logits: Tensor
    descriptor_values: Tensor | None
    chord_logits: Tensor | None
    availability: Tensor
    prediction_mask: Tensor
    pooling_weights: Tensor
    descriptor_raw: Tensor | None = None  # inverse-transformed, physical units


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
        descriptor_dim: int | None = None,
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
        if descriptor_dim is not None and (
            not isinstance(descriptor_dim, int)
            or isinstance(descriptor_dim, bool)
            or descriptor_dim < 1
        ):
            raise ValueError("descriptor_dim must be None or a positive integer")
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
        self.descriptor_head = (
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, descriptor_dim),
            )
            if descriptor_dim is not None
            else None
        )
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
        descriptor_values = (
            self.descriptor_head(embedding) * availability.unsqueeze(-1).to(embedding.dtype)
            if self.descriptor_head is not None
            else None
        )
        return HarmonyBranchOutput(
            embedding=embedding,
            chroma_logits=chroma_logits,
            descriptor_values=descriptor_values,
            chord_logits=chord_logits,
            availability=availability,
            prediction_mask=mask,
            pooling_weights=weights,
        )


# ---------------------------------------------------------------------------
# Harmony v4: chroma-grounded 45-descriptor branch
# ---------------------------------------------------------------------------

_INDEX = {name: i for i, name in enumerate(HARMONY_DESCRIPTORS)}
# Descriptors the extractor defines as exact functions of per-frame chroma means.
_EXACT = (*CHROMA_MEAN_FEATURES, *TONNETZ_MEAN_FEATURES)
_LEARNED = tuple(name for name in HARMONY_DESCRIPTORS if name not in _EXACT)
_LEARNED_GROUPS = {
    group: tuple(name for name in names if name in _LEARNED)
    for group, names in DESCRIPTOR_GROUPS.items()
    if any(name in _LEARNED for name in names)
}
assert tuple(name for names in _LEARNED_GROUPS.values() for name in names) == _LEARNED
# Token-chroma statistics computed with the extractor's own formulas. Each one
# is the model-resolution analogue of the learned descriptor of the same name.
CHROMA_STATISTICS: tuple[str, ...] = (
    *CHROMA_STD_FEATURES,
    *TONNETZ_STD_FEATURES,
    "tonal_concentration_mean", "tonal_concentration_std",
    "chroma_entropy_mean", "chroma_entropy_std",
    "chroma_flux_mean", "chroma_flux_std",
    "tonnetz_movement_mean", "tonnetz_movement_std",
)
N_CHROMA_STATISTICS = len(CHROMA_STATISTICS)


def _masked_moments(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and population std over dim 1; rows with no weight return zeros."""
    w = weights.to(values.dtype).unsqueeze(-1)
    count = w.sum(dim=1).clamp_min(1.0)
    mean = (values * w).sum(dim=1) / count
    variance = (((values - mean.unsqueeze(1)) * w) ** 2).sum(dim=1) / count
    # sqrt has an undefined derivative at 0 (constant or single-token rows).
    positive = variance > 1e-12
    std = torch.sqrt(torch.where(positive, variance, torch.ones_like(variance)))
    return mean, std * positive.to(values.dtype)


class _DilatedResidualBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, width, kernel_size, padding=dilation * (kernel_size - 1) // 2,
                      dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        # values (N, C, L), mask (N, L)
        update = self.network(values)
        values = self.norm((values + update).transpose(1, 2)).transpose(1, 2)
        return values * mask.unsqueeze(1).to(values.dtype)


class ChromaGroundedHarmonyBranch(nn.Module):
    """Predict all 45 harmony descriptors from ordered shared-encoder tokens.

    1. Gap-safe dilated residual convolutions (receptive field ~2 s) run inside
       each sampled window, never across the gap to the next one.
    2. A per-token pitch-class head yields chroma probabilities ``q_t``.
    3. The 12 chroma means are the masked mean of ``q_t`` and the 6 Tonnetz
       means are ``TONNETZ_PHI @ chroma_mean``: the extractor's exact
       definitions, so these 18 outputs need no free parameters of their own.
    4. The remaining 27 descriptors come from a regression head over
       attention/mean/std statistics pooling plus 26 token-chroma statistics
       computed with the extractor's formulas (std, entropy, max-bin, L2 flux,
       L2 Tonnetz movement). Each such descriptor also gets a direct learned
       path from its matching statistic.

    ``descriptor_values`` are in the training target space (named transform then
    train-only z-score; call :meth:`set_target_transform` after fitting).
    ``descriptor_raw`` is the inverse transform in physical units.
    """

    def __init__(
        self,
        input_dim: int = 128,
        *,
        hidden_dim: int = 96,
        embedding_dim: int = 64,
        temporal_layers: int = 4,
        kernel_size: int = 3,
        chord_classes: int | None = None,
        dropout: float = 0.1,
        grouped_descriptor_heads: bool = False,
    ):
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (hidden_dim, "hidden_dim"),
            (embedding_dim, "embedding_dim"),
            (temporal_layers, "temporal_layers"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if chord_classes is not None and chord_classes < 2:
            raise ValueError("chord_classes must be None or an integer of at least two")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.grouped_descriptor_heads = bool(grouped_descriptor_heads)

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.temporal_blocks = nn.ModuleList(
            _DilatedResidualBlock(hidden_dim, kernel_size, 2**layer, dropout)
            for layer in range(temporal_layers)
        )
        self.chroma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 12)
        )
        self.chord_head = nn.Linear(hidden_dim, chord_classes) if chord_classes else None
        self.attention = nn.Linear(hidden_dim, 1)
        self.summary = nn.Sequential(
            nn.Linear(3 * hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
        )
        head_in = embedding_dim + N_CHROMA_STATISTICS
        def descriptor_head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        # The 18 chroma/Tonnetz means remain exact deterministic outputs.  This
        # optional experiment only separates the 27 learned descriptors into
        # semantically coherent internal heads; the public output stays [B,45].
        if self.grouped_descriptor_heads:
            self.descriptor_head = None
            self.descriptor_heads = nn.ModuleDict({
                group: descriptor_head(len(names))
                for group, names in _LEARNED_GROUPS.items()
            })
        else:
            self.descriptor_head = descriptor_head(len(_LEARNED))
            self.descriptor_heads = nn.ModuleDict()
        self.statistic_gain = nn.Parameter(torch.zeros(len(_LEARNED)))

        self.register_buffer("tonnetz_phi", torch.tensor(tonnetz_phi(), dtype=torch.float32))
        self.register_buffer("transform_codes", torch.tensor(transform_codes(), dtype=torch.long))
        self.register_buffer("target_mean", torch.zeros(N_HARMONY_DESCRIPTORS))
        self.register_buffer("target_scale", torch.ones(N_HARMONY_DESCRIPTORS))
        self.register_buffer("exact_index", torch.tensor([_INDEX[n] for n in _EXACT]))
        self.register_buffer("learned_index", torch.tensor([_INDEX[n] for n in _LEARNED]))
        aligned = [CHROMA_STATISTICS.index(n) if n in CHROMA_STATISTICS else -1 for n in _LEARNED]
        self.register_buffer("aligned_statistic", torch.tensor(aligned))

    def set_target_transform(self, transform: HarmonyTargetTransform) -> None:
        state = transform.state_dict()
        self.target_mean.copy_(torch.tensor(state["mean"], dtype=self.target_mean.dtype))
        self.target_scale.copy_(torch.tensor(state["scale"], dtype=self.target_scale.dtype))

    def _standardize(self, raw: Tensor, index: Tensor) -> Tensor:
        codes = self.transform_codes[index]
        return (forward_transform_torch(raw, codes) - self.target_mean[index]) / self.target_scale[index]

    def forward(
        self,
        encoded_sequence: Tensor,
        sequence_mask: Tensor,
        sequence_window_index: Tensor,
        *,
        windows: int,
        tokens_per_window: int,
    ) -> HarmonyBranchOutput:
        batch, tokens = TemporalHarmonyBranch._validate_layout(
            encoded_sequence, sequence_mask, sequence_window_index, windows, tokens_per_window
        )
        if encoded_sequence.shape[-1] != self.input_dim:
            raise ValueError(
                f"encoded feature width must be {self.input_dim}, got {encoded_sequence.shape[-1]}"
            )
        mask = sequence_mask.to(torch.bool)
        fmask = mask.to(encoded_sequence.dtype)
        availability = mask.any(dim=1)

        # Gap-safe temporal context: each sampled window is its own sequence.
        hidden = self.input_projection(encoded_sequence) * fmask.unsqueeze(-1)
        window_mask = mask.reshape(batch * windows, tokens_per_window)
        context = hidden.reshape(batch * windows, tokens_per_window, self.hidden_dim).transpose(1, 2)
        for block in self.temporal_blocks:
            context = block(context, window_mask)
        context = context.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)
        context = context * fmask.unsqueeze(-1)

        chroma_logits = self.chroma_head(context).masked_fill(~mask.unsqueeze(-1), 0)
        chord_logits = (
            self.chord_head(context).masked_fill(~mask.unsqueeze(-1), 0)
            if self.chord_head is not None else None
        )

        # Statistics pooling over token context.
        scores = self.attention(context).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        scores = torch.where(availability.unsqueeze(1), scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, dim=1) * fmask
        attention_mean = (context * attention.unsqueeze(-1)).sum(dim=1)
        context_mean, context_std = _masked_moments(context, fmask)
        embedding = self.summary(torch.cat((attention_mean, context_mean, context_std), dim=-1))
        embedding = embedding * availability.unsqueeze(-1).to(embedding.dtype)

        # Token chroma and the extractor's statistics at model resolution.
        q = torch.softmax(chroma_logits, dim=-1)
        tonnetz = q @ self.tonnetz_phi.T
        chroma_mean, chroma_std = _masked_moments(q, fmask)
        _, tonnetz_std = _masked_moments(tonnetz, fmask)
        entropy = -(q * torch.log(q.clamp_min(1e-12))).sum(-1) / math.log(12.0)
        concentration = q.max(dim=-1).values
        pair = mask[:, 1:] & mask[:, :-1] & (sequence_window_index[:, 1:] == sequence_window_index[:, :-1])
        flux = torch.linalg.vector_norm(q[:, 1:] - q[:, :-1], dim=-1)
        movement = torch.linalg.vector_norm(tonnetz[:, 1:] - tonnetz[:, :-1], dim=-1)
        scalars = torch.stack((concentration, entropy), dim=-1)
        scalar_mean, scalar_std = _masked_moments(scalars, fmask)
        deltas = torch.stack((flux, movement), dim=-1)
        delta_mean, delta_std = _masked_moments(deltas, pair)
        statistics = torch.cat((
            chroma_std, tonnetz_std,
            scalar_mean[:, :1], scalar_std[:, :1],
            scalar_mean[:, 1:], scalar_std[:, 1:],
            delta_mean[:, :1], delta_std[:, :1],
            delta_mean[:, 1:], delta_std[:, 1:],
        ), dim=-1)
        log_statistics = torch.log(statistics + 1e-4)

        # Exact descriptors: transform + standardize the derived raw values.
        exact_raw = torch.cat((chroma_mean, chroma_mean @ self.tonnetz_phi.T), dim=-1)
        exact_values = self._standardize(exact_raw, self.exact_index)

        # Learned descriptors: pooled context + chroma statistics, plus a direct
        # gain from each descriptor's matching statistic.
        head_input = torch.cat((embedding, log_statistics), dim=-1)
        if self.grouped_descriptor_heads:
            learned_values = torch.cat(
                [self.descriptor_heads[group](head_input) for group in _LEARNED_GROUPS],
                dim=-1,
            )
        else:
            assert self.descriptor_head is not None
            learned_values = self.descriptor_head(head_input)
        aligned = self.aligned_statistic
        has_statistic = aligned >= 0
        matched = log_statistics[:, aligned.clamp_min(0)] * has_statistic.to(log_statistics.dtype)
        learned_values = learned_values + self.statistic_gain * matched

        values = torch.zeros(batch, N_HARMONY_DESCRIPTORS, dtype=context.dtype, device=context.device)
        values = values.index_copy(1, self.exact_index, exact_values)
        values = values.index_copy(1, self.learned_index, learned_values)
        values = values * availability.unsqueeze(-1).to(values.dtype)
        raw = inverse_transform_torch(values * self.target_scale + self.target_mean, self.transform_codes)
        raw = raw * availability.unsqueeze(-1).to(raw.dtype)

        return HarmonyBranchOutput(
            embedding=embedding,
            chroma_logits=chroma_logits,
            descriptor_values=values,
            chord_logits=chord_logits,
            availability=availability,
            prediction_mask=mask,
            pooling_weights=attention,
            descriptor_raw=raw,
        )
