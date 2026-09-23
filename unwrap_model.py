import argparse
import json
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.training.autoencoders import AutoencoderTrainingWrapper


SSL_KEY_FRAGMENTS = (
    "ssl_loss",
    "wavlm",
    "ssl_model",
    "feature_extractor",
    "post_extract_proj",
)


def fix_ema_state_dict(state_dict):
    """Normalize old EMA scalar buffers to the shape expected by ema-pytorch."""
    for key in ("autoencoder_ema.initted", "autoencoder_ema.step"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            state_dict[key] = value.unsqueeze(0)
    return state_dict


def strip_ssl_keys(state_dict):
    """Drop training-only SSL/WavLM bottleneck weights from an exported model."""
    kept = {}
    removed = []
    for key, value in state_dict.items():
        lower_key = key.lower()
        if any(fragment in lower_key for fragment in SSL_KEY_FRAGMENTS):
            removed.append(key)
            continue
        kept[key] = value
    return kept, removed


def build_ema_copy(model_config, model):
    ema_copy = create_model_from_config(model_config)
    ema_copy = create_model_from_config(model_config)
    for name, param in model.state_dict().items():
        if isinstance(param, Parameter):
            param = param.data
        ema_copy.state_dict()[name].copy_(param)
    return ema_copy


def load_autoencoder_wrapper(model_config, ckpt_path):
    training_config = model_config["training"]
    model = create_model_from_config(model_config)
    use_ema = training_config.get("use_ema", False)
    ema_copy = build_ema_copy(model_config, model) if use_ema else None

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint["state_dict"] = fix_ema_state_dict(checkpoint["state_dict"])

    wrapper = AutoencoderTrainingWrapper(
        autoencoder=model,
        sample_rate=model_config.get("sample_rate", 16000),
        loss_config=training_config["loss_configs"],
        use_ema=use_ema,
        ema_copy=ema_copy,
        optimizer_configs=training_config.get("optimizer_configs"),
        latent_subspace=training_config.get("latent_subspace"),
        adaptive_loss_weight=model_config.get("adaptive_loss_weight", False),
        adaptive_loss_keys=model_config.get("adaptive_loss_keys"),
        reference_loss_key=model_config.get("reference_loss_key"),
        model_param_names=model_config.get("model_param_names"),
    )
    missing_keys, unexpected_keys = wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    return wrapper


def export_autoencoder(wrapper, output_path, strip_ssl=True, use_safetensors=False):
    if wrapper.autoencoder_ema is not None:
        model = wrapper.autoencoder_ema.ema_model
    else:
        model = wrapper.autoencoder

    state_dict = model.state_dict()
    removed = []
    if strip_ssl:
        state_dict, removed = strip_ssl_keys(state_dict)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_safetensors:
        from safetensors.torch import save_file

        save_file(state_dict, str(output_path))
    else:
        torch.save({"state_dict": state_dict}, output_path)

    print(f"Exported: {output_path}")
    print(f"Removed training-only SSL keys: {len(removed)}")


def main():
    parser = argparse.ArgumentParser(description="Export a JMAS-VAE/vanilla VAE inference checkpoint.")
    parser.add_argument("--model-config", required=True, help="Training model JSON.")
    parser.add_argument("--ckpt-path", required=True, help="Lightning checkpoint path.")
    parser.add_argument("--output", required=True, help="Output .ckpt or .safetensors path.")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--keep-ssl", action="store_true", help="Keep SSL/WavLM bottleneck weights in the export.")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config.get("model_type") != "autoencoder":
        raise ValueError("Only autoencoder checkpoints are supported by this exporter.")

    wrapper = load_autoencoder_wrapper(model_config, args.ckpt_path)
    export_autoencoder(
        wrapper,
        output_path=args.output,
        strip_ssl=not args.keep_ssl,
        use_safetensors=args.use_safetensors,
    )


if __name__ == "__main__":
    main()
