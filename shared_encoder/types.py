"""Named shared-encoder output contracts."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SharedEncoderOutput:
    """All branch representations produced by one shared CNN evaluation."""

    encoded_sequence: Tensor
    sequence_times: Tensor
    sequence_start_times: Tensor
    sequence_end_times: Tensor
    sequence_mask: Tensor
    sequence_window_index: Tensor
    pooled_song: Tensor
    window_repr: Tensor
    availability: Tensor

    @property
    def song_repr(self) -> Tensor:
        """Instrument/timbre compatibility alias for ``pooled_song``."""
        return self.pooled_song


# Older code imported this name from scripts.shared_audio_encoder.
TemporalEncoderOutput = SharedEncoderOutput

