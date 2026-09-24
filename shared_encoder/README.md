# Shared CNN encoder

This package owns the audio encoder used by all four concept branches.

```text
shared_encoder/
├── __init__.py      # stable public imports
├── constants.py     # frozen dimensions and temporal geometry
├── geometry.py      # token masks, timestamps, and window identity
├── model.py         # SharedAudioEncoder and checkpoint/freezing behavior
├── types.py         # named output structure
└── validation.py    # audio, metadata, and padded-frame validation
```

Use the canonical import:

```python
from shared_encoder import SharedAudioEncoder

encoder = SharedAudioEncoder()  # production output width: 128
output = encoder(
    log_mel_windows,
    window_mask,
    window_valid_frames,
    window_start_seconds,
)
```

The older import remains a compatibility shim:

```python
from scripts.shared_audio_encoder import SharedAudioEncoder
```

Both imports resolve to the same class. New implementation code belongs in this
package, not in `scripts/`.

See [the architecture specification](../docs/shared-cnn-encoder-architecture.md)
for the exact tensor, masking, timestamp, receptive-field, branch, and checkpoint
contracts.

