"""Read, validate and snapshot the current architecture without changing label contracts."""
from copy import deepcopy
import json
from pathlib import Path

from torch import nn

DEFAULT_ARCHITECTURE_PATH = Path(__file__).resolve().parents[1] / "configs/architecture.json"


def activation(name):
    factories = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    if name not in factories:
        raise ValueError(f"Unsupported activation: {name}; use gelu, relu or silu")
    return factories[name]()


def load_architecture(value=None):
    if isinstance(value, dict):
        data = deepcopy(value)
    else:
        path = Path(value) if value is not None else DEFAULT_ARCHITECTURE_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
    validate_architecture(data)
    return data


def validate_architecture(data):
    if data.get("schema_version") != "architecture_config_v1":
        raise ValueError("Unsupported architecture config schema")
    expected_sections = {"schema_version", "shared_encoder", "instrument", "timbre", "rhythm",
                         "harmony", "genre_predictor", "direct_cnn"}
    if set(data) != expected_sections:
        raise ValueError("Architecture config has missing or unknown sections")
    # Reject typos instead of silently ignoring a requested architecture change.
    fields = {
        "shared_encoder": "channels kernel_size conv_bias group_norm_groups activation pool_sizes output_dim frequency_pooling song_pooling",
        "instrument": "input_dim hidden_dim output_dim activation normalization dropout",
        "timbre": "input_dim hidden_dim output_dim activation normalization dropout",
        "rhythm": "input_dim hidden_dim embedding_dim output_dim temporal_layers kernel_size dilation_base activation normalization dropout pooling",
        "harmony": "input_dim hidden_dim embedding_dim output_dim temporal_layers kernel_size dilation_base activation normalization dropout pooling chroma_head_trainable chord_head_enabled",
        "genre_predictor": "fusion predictor activation hidden_dim hidden_dropout concept_dropout output_dim",
        "direct_cnn": "hidden_dim activation dropout output_dim",
    }
    for section, names in fields.items():
        values = data[section]
        if set(values) != set(names.split()):
            raise ValueError(f"Missing or unknown fields in {section}")
        for key, value in values.items():
            if key.endswith("dim") or key in ("kernel_size", "temporal_layers", "dilation_base", "group_norm_groups"):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{section}.{key} must be a positive integer")
            if "dropout" in key and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1):
                raise ValueError(f"{section}.{key} must be in [0,1)")
        if "activation" in values:
            activation(values["activation"])
    encoder = data["shared_encoder"]
    if len(encoder["channels"]) != 3 or any(type(c) is not int or c < 1 or c % encoder["group_norm_groups"] for c in encoder["channels"]):
        raise ValueError("Encoder needs three positive channel widths divisible by group_norm_groups")
    if type(encoder["conv_bias"]) is not bool:
        raise ValueError("shared_encoder.conv_bias must be boolean")
    if encoder["kernel_size"] % 2 != 1:
        raise ValueError("Encoder kernel_size must be odd for same padding")
    if encoder["pool_sizes"] != [[2, 2], [2, 1]] or encoder["output_dim"] != 128:
        raise ValueError("Encoder pooling geometry and 128-D output are fixed by the audio/branch contracts")
    if encoder["frequency_pooling"] != "learned_attention" or encoder["song_pooling"] != "masked_mean":
        raise ValueError("Only learned_attention and masked_mean pooling are implemented")
    for name, width in (("instrument", 41), ("timbre", 35), ("rhythm", 10), ("harmony", 12)):
        values = data[name]
        if values["input_dim"] != 128 or values["output_dim"] != width or values["normalization"] != "layer_norm":
            raise ValueError(f"{name} requires 128-D input, {width} target outputs and layer_norm")
        if name in ("rhythm", "harmony"):
            if values["kernel_size"] < 3 or values["kernel_size"] % 2 != 1 or values["pooling"] != "masked_mean":
                raise ValueError(f"{name} requires an odd kernel >=3 and masked_mean pooling")
    if data["rhythm"]["embedding_dim"] != 64:
        raise ValueError("Rhythm diagnostic embedding_dim is fixed at 64")
    if data["harmony"]["chroma_head_trainable"] is not False or data["harmony"]["chord_head_enabled"] is not False:
        raise ValueError("The descriptor dataset does not supervise chroma/chord heads")
    genre = data["genre_predictor"]
    if genre["fusion"] not in ("native_concat", "gated") or genre["predictor"] not in ("linear", "mlp"):
        raise ValueError("Fusion must be native_concat or gated; predictor must be linear or mlp")
    if genre["predictor"] == "linear" and genre["hidden_dropout"] != 0:
        raise ValueError("Linear genre predictor has no hidden dropout layer")
    if genre["output_dim"] != 6 or data["direct_cnn"]["output_dim"] != 6:
        raise ValueError("Genre output_dim must match the six dataset labels")


def configure_training(cfg):
    """Explicit snapshots configure fusion too; legacy programmatic flags still work."""
    explicit = cfg.architecture is not None
    cfg.architecture = load_architecture(cfg.architecture)
    default_fusion = (cfg.fusion, cfg.genre_predictor, cfg.genre_hidden_dim, cfg.genre_dropout, cfg.concept_dropout) == \
        ("native_concat", "linear", 64, 0.0, 0.0)
    if explicit or default_fusion:
        genre = cfg.architecture["genre_predictor"]
        cfg.fusion, cfg.genre_predictor = genre["fusion"], genre["predictor"]
        cfg.genre_hidden_dim = genre["hidden_dim"]
        cfg.genre_dropout, cfg.concept_dropout = genre["hidden_dropout"], genre["concept_dropout"]
    return cfg.architecture
