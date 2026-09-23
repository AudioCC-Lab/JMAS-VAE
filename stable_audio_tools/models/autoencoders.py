import torch
import math
import numpy as np
import sys
import torchaudio
from functools import partial
from pathlib import Path

# timm is only needed by GeGluMlp (WavLM-based encoders, not part of the
# released autoencoder configs), so import it lazily to keep the terminal
# output clean for the default training/inference paths.
from torch import nn, sin, pow
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore", message=r"torch\.nn\.utils\.weight_norm is deprecated.*")
from torch.nn.utils import weight_norm
from torchaudio import transforms as T
from alias_free_torch import Activation1d
from typing import List, Literal, Dict, Any, Callable
from einops import rearrange
from dac.nn.layers import Snake1d, WNConv1d

from ..inference.utils import prepare_audio
from .blocks import SnakeBeta
from .bottleneck import Bottleneck, DiscreteBottleneck
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform, AutoencoderPretransform
from .transformer import ContinuousTransformer, TransformerBlock, RotaryEmbedding


_FOCALCODEC_CLASSES = None


def _get_focalcodec_classes():
    global _FOCALCODEC_CLASSES
    if _FOCALCODEC_CLASSES is not None:
        return _FOCALCODEC_CLASSES

    try:
        from focalcodec.focalnet import FocalDecoder as FCFocalDecoder
        from focalcodec.focalnet import FocalEncoder as FCFocalEncoder
        from focalcodec.vocos import Vocos as FCVocos
        from focalcodec.wavlm import WavLM as FCWavLM
    except ImportError:
        repo_root = Path(__file__).resolve()
        for parent in repo_root.parents:
            candidate = parent / "focalcodec-main" / "focalcodec-main"
            if candidate.exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                break
        from focalcodec.focalnet import FocalDecoder as FCFocalDecoder
        from focalcodec.focalnet import FocalEncoder as FCFocalEncoder
        from focalcodec.vocos import Vocos as FCVocos
        from focalcodec.wavlm import WavLM as FCWavLM

    _FOCALCODEC_CLASSES = {
        "WavLM": FCWavLM,
        "FocalEncoder": FCFocalEncoder,
        "FocalDecoder": FCFocalDecoder,
        "Vocos": FCVocos,
    }
    return _FOCALCODEC_CLASSES


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _load_wavlm_from_checkpoint(
    checkpoint_path: str,
    num_layers: int = 6,
):
    from .WavLM import WavLM as StableWavLM
    from .WavLM import WavLMConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = WavLMConfig(checkpoint["cfg"])
    model = StableWavLM(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)

    if num_layers is not None:
        model.encoder.layers = nn.ModuleList(list(model.encoder.layers[:num_layers]))
        model.cfg.encoder_layers = num_layers

    return model, cfg


def _freeze_wavlm_prefix(
    model: nn.Module,
    frozen_layers: int = 5,
    freeze_feature_extractor: bool = True,
    freeze_post_extract_proj: bool = True,
    freeze_input_layer_norm: bool = True,
    freeze_pos_conv: bool = True,
    freeze_encoder_layer_norm: bool = True,
) -> None:
    if freeze_feature_extractor:
        _freeze_module(model.feature_extractor)
    if freeze_post_extract_proj and model.post_extract_proj is not None:
        _freeze_module(model.post_extract_proj)
    if freeze_input_layer_norm:
        _freeze_module(model.layer_norm)
    if freeze_pos_conv:
        _freeze_module(model.encoder.pos_conv)
    if freeze_encoder_layer_norm:
        _freeze_module(model.encoder.layer_norm)

    for layer in list(model.encoder.layers[:frozen_layers]):
        _freeze_module(layer)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        
        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        res = x
        
        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)

        return x + res

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')

class TAAEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, type = 'encoder', transformer_depth = 3, use_snake = False, sliding_window = [31,32], checkpointing = False, conformer = False, layer_scale = True, use_dilated_conv = False):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")
        
        self.checkpointing = checkpointing
        
        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        transformers.append(Transpose())

        self.sliding_window = sliding_window

        for _ in range(transformer_depth):
            transformers.append(TransformerBlock(transformer_dim, 
                                                 dim_heads = 128, 
                                                 causal = False, 
                                                 zero_init_branch_outputs = True if not layer_scale else False, 
                                                 remove_norms = False, 
                                                 conformer = conformer, 
                                                 layer_scale = layer_scale, 
                                                 add_rope = True, 
                                                 attn_kwargs={'qk_norm': "ln"}, 
                                                 ff_kwargs={'mult': 4, 'no_bias': False},
                                                 norm_kwargs = {'eps': 1e-2}))
        transformers.append(Transpose())
        transformers = nn.ModuleList(transformers)

        if type == 'encoder':
            layers = []
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake))
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=in_channels))
            layers.append(WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            layers.append(transformers)
            self.layers = nn.ModuleList(layers)
        elif type == 'decoder':
            layers = []
            layers.append(transformers)
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=out_channels))
            layers.append(WNConvTranspose1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for transformer in layer:
                    if self.checkpointing:
                        x = checkpoint(transformer, x, self_attention_flash_sliding_window = self.sliding_window)
                    else:
                        x = transformer(x, self_attention_flash_sliding_window = self.sliding_window)
            else:
                if self.checkpointing:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
        return x

class TAAEEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()
          
        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [WNConv1d(in_channels=in_channels, out_channels=channel_dims[0], kernel_size=7, padding=3, bias = True)]

        for i in range(self.depth):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]

        layers += [
            get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[-1]),
            WNConv1d(in_channels=channel_dims[-1], out_channels=latent_dim, kernel_size=3, padding=1, bias = True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TAAEDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=channel_dims[-1], kernel_size=3, padding=1, bias = True)
        ]
        
        for i in range(self.depth, 0, -1):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]  

        layers += [get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[0]),
                    WNConv1d(in_channels=channel_dims[0], out_channels=out_channels, kernel_size=7, padding=3, bias = False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()
        self.in_channels = in_channels
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Decoder as DACDecoder

        self.decoder = DACDecoder(**kwargs, input_channel = latent_dim, d_out=out_channels)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class BigVGANDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from .bigvgan.bigvgan import BigVGAN as BigVGANDecoder
        from .bigvgan.env import AttrDict

        h = AttrDict(
            **kwargs,
            input_channel=latent_dim,
            d_out=out_channels,
        )

        self.decoder = BigVGANDecoder(h)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)


class WavCubeDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super().__init__()

        from .wavcube_decoder import WavCubeLatentDecoder

        self.decoder = WavCubeLatentDecoder(
            latent_dim=latent_dim,
            **kwargs,
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class DecomBigVGANDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from .bigvgan.bigvgan import BigVGAN as BigVGANDecoder
        from .bigvgan.env import AttrDict

        
        decom_factor = kwargs.get("decom_factor", 8)
        num_atten_layers = kwargs.get("num_atten_layers", 1)
        is_quant = kwargs.get("is_quant", True)
        h = AttrDict(
            **kwargs,
            input_channel=latent_dim*decom_factor,
            d_out=out_channels,
        )
        self.proj = AttnProjection(
            in_dim=latent_dim,
            out_dim=latent_dim*decom_factor,
            num_heads=max(1, decom_factor),
            num_layers=num_atten_layers,
            is_quant=is_quant
        )

        self.decoder = BigVGANDecoder(h)

        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, T, latent_dim) -> (B, T, latent_dim*decom_factor)
        return self.decoder(x)

class GloBigVGANDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1,**kwargs):
        super().__init__()
        from .bigvgan.globigvgan import GloBigVGAN as GloBigVGANDecoder
        from .bigvgan.env import AttrDict

        h = AttrDict(
            **kwargs,
            input_channel=latent_dim,
            d_out=out_channels,
        )
        # 传入z_dim=你的基准模型输出维度
        self.decoder = GloBigVGANDecoder(h)
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        x: 原始解码器输入 [B, latent_dim, T]
        z: 你的基准模型输出 [B, T, z_dim]
        """
        return self.decoder(x)

class BigVGANATTNDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from .bigvgan.bigvgan_attn import BigVGAN_ATTN as BigVGANDecoder
        from .bigvgan.env import AttrDict

        h = AttrDict(
            **kwargs,
            input_channel=latent_dim,
            d_out=out_channels,
        )

        self.decoder = BigVGANDecoder(h)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class HIFIGANDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from .hifigan.models import Generator as HIFIGANDecoder
        from .bigvgan.env import AttrDict

        h = AttrDict(
            **kwargs,
            latent_dim=latent_dim,
            d_out=out_channels,
        )

        self.decoder = HIFIGANDecoder(h)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class ZeroDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.upsample_ratio = kwargs.get("upsample_ratio", 320)

    def forward(self, x):
        # 输入形状: [B, latent_dim, T]
        # 输出全0张量，形状与标准解码器输出一致: [B, out_channels, T]
        return torch.zeros(
            x.shape[0], 
            self.out_channels, 
            x.shape[-1]*self.upsample_ratio, 
            device=x.device, 
            dtype=x.dtype
        )

class ConvNext2LatentEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from .convnext import ConvNeXtEncoder

        input_latent_dim = kwargs.pop("vae_latent_dim", None)
        self.latent_dim = kwargs.get("latent_dim", None)

        self.encoder = ConvNeXtEncoder(in_channels=input_latent_dim, **kwargs)

    def forward(self, x):
        return self.encoder(x)


class ConvNext2LatentDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from .convnext import ConvNeXtDecoder

        output_latent_dim = kwargs.pop("vae_latent_dim", None)
        self.latent_dim = latent_dim
        self.decoder = ConvNeXtDecoder(
            out_channels=output_latent_dim, latent_dim=latent_dim, **kwargs
        )

    def forward(self, x):
        return self.decoder(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, skip_bottleneck: bool = False, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)

        encoder_info = {}
        wavlm_feat = None
        if self.encoder is not None:
            if iterate_batch:
                latents = []
                wavlm_feat_list = []
                for i in range(audio.shape[0]):
                    if isinstance(self.encoder, WavLMEncoder) or isinstance(self.encoder, SFMEncoder):
                        latent_batch, info_batch = self.encoder(audio[i:i+1])
                        wavlm_feat_list.append(info_batch["wavlm_feat"])
                    else:
                        latent_batch = self.encoder(audio[i:i+1])
                    latents.append(latent_batch)
                latents = torch.cat(latents, dim=0)

                if (isinstance(self.encoder, WavLMEncoder) or isinstance(self.encoder, SFMEncoder)) and len(wavlm_feat_list) > 0:
                    wavlm_feat = torch.cat(wavlm_feat_list, dim=0)
                    encoder_info["wavlm_feat"] = wavlm_feat

            else:
                if isinstance(self.encoder, WavLMEncoder) or isinstance(self.encoder, SFMEncoder):
                    latents, encoder_info = self.encoder(audio)
                    wavlm_feat = encoder_info.get("wavlm_feat")
                else:
                    latents = self.encoder(audio)
                    encoder_info = {}
        else:
            latents = audio
            encoder_info = {}

        info["pre_bottleneck_latents"] = latents

        if self.bottleneck is not None and not skip_bottleneck:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            if (isinstance(self.encoder, WavLMEncoder) or isinstance(self.encoder, SFMEncoder)) and wavlm_feat is not None:
                kwargs["wavlm_feat"] = wavlm_feat
            latents, bottleneck_info = self.bottleneck.encode(latents, audio = audio, return_info=True, **kwargs)

            info.update(bottleneck_info)
        
        if return_info:
            return latents, info

        return latents

    def decode(self, latents, skip_bottleneck: bool = False, iterate_batch=False, **kwargs):

        if self.bottleneck is not None and not skip_bottleneck:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        
        return decoded
          
    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)
  
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = int(self.downsampling_ratio)
            total_size = audio.shape[2] # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent # converting metric in latents to samples
            overlap *= samples_per_latent # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = audio[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size,self.latent_dim,y_size), dtype = chunks.dtype).to(audio.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap//samples_per_latent//2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = latents[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = int(self.downsampling_ratio)
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size,self.out_channels,y_size), dtype = chunks.dtype).to(latents.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap//2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final

    
class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion: "ConditionedDiffusionModel",
        diffusion_downsampling_ratio,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

        if self.encoder is not None:
            # Shrink the initial encoder parameters to avoid saturated latents
            with torch.no_grad():
                for param in self.encoder.parameters():
                    param *= 0.5

    def decode(self, latents, steps=100):

        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder is not None:
            latents = self.decode(latents)
    
        # Upsample latents to match diffusion length
        if latents.shape[2] != upsampled_length:
            latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        from ..inference.sampling import sample
        decoded = sample(self.diffusion, noise, steps, 0, input_concat_cond=latents)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)

        return decoded
        
# AE factories

class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
    ):
        super().__init__()
        try:
            from timm.layers import get_norm_layer
        except ImportError:
            from timm.models.layers import get_norm_layer
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


class AttnProjectionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, is_quant, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        if is_quant:
            self.blocks = nn.ModuleList([
                AttnProjectionBlock(in_dim, in_dim, num_heads, norm_layer, mlp_ratio)
                if i < num_layers - 1 else
                AttnProjectionBlock(in_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                for i in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                AttnProjectionBlock(in_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                if i == 0 else
                AttnProjectionBlock(out_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                for i in range(num_layers)
            ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
class RAEEncoder(nn.Module):
    def __init__(self, model_name="WAVLM_LARGE", layer=23):  
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        
        # 加载WavLM并冻结参数
        self._load_wavlm()

    def _load_wavlm(self):
        """加载预训练WavLM并冻结参数"""
        if self.model_name == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
        elif self.model_name == "WAVLM_BASE":
            bundle = torchaudio.pipelines.WAVLM_BASE
        else:
            raise ValueError(f"Unsupported model type: {self.model_name} (only WAVLM_LARGE/WAVLM_BASE are supported)")

        self.model = bundle.get_model()
        # 冻结所有参数（仅用于特征提取，不微调）
        for param in self.model.parameters():
            param.requires_grad = False

    #提取layer层的特征
    def _extract_layer_feature(self, audio):
        """提取指定层的特征"""
        # 输入形状转换: (batch, 1, time) -> (batch, time)
        audio = audio.squeeze(1)
        
        with torch.no_grad():  # 禁用梯度计算，提升效率
            features, _ = self.model.extract_features(audio, num_layers=self.layer)
        
        feat = features[self.layer - 1]  # WavLM返回的features索引为 layer-1
        return feat
    def forward(self, x):
        feat = self._extract_layer_feature(x)
        return feat.permute(0, 2, 1)  # 转换为 (batch, feat_dim, time)

    

class WavLMEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        model_name="WAVLM_LARGE",
        layers=23,  # 要求传入三元数组或单个数字，单个数字代表某一层，三元数组代表三层
        align_layer=23, #后续进行语义对齐的层
        num_atten_layers=1,
        is_quant=False,
        downsampling_ratio = 400,
        wavlm_source = "torchaudio",
        wavlm_ckpt = None,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.align_layer = align_layer
        self.layers = layers
        self.downsampling_ratio = downsampling_ratio
        self.wavlm_source = wavlm_source
        self.wavlm_ckpt = wavlm_ckpt
        
        # 输入合法性检查
        #self._validate_layers() if self.layers is not None else None
        
        # 加载WavLM并冻结参数
        self._load_wavlm()
        
        # 基础配置（根据模型类型确定特征维度）
        self.feat_dim = 1024 if model_name == "WAVLM_LARGE" else 768
        #如果layers是三元数组，则总特征维度是feat_dim的三倍，如果layers是一个数字，则总特征维度就是feat_dim
        self.sum_feat_dim = self.feat_dim * 3 if isinstance(self.layers, (list, tuple)) else self.feat_dim
        self.num_heads = kwargs.get("num_heads", max(1, self.sum_feat_dim // self.latent_dim))
        # 三个层特征拼接后的总维度

        self.proj = AttnProjection(
            in_dim=self.sum_feat_dim,
            out_dim=self.latent_dim,
            num_heads=self.num_heads,
            num_layers=num_atten_layers,
            is_quant=is_quant
        )

    def _validate_layers(self):
        """验证layers参数是否为合法三元数组"""
        if not isinstance(self.layers, (list, tuple)) or len(self.layers) != 3:
            raise ValueError(f"layers must be an array of length 3, got: {self.layers}")
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, int) or layer <= 0:
                raise ValueError(f"layers[{idx}] must be a positive integer, got: {layer}")

    def _load_wavlm(self):
        """加载预训练WavLM并冻结参数"""
        if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
            if self.model_name == "WAVLM_LARGE":
                bundle = torchaudio.pipelines.WAVLM_LARGE
            elif self.model_name == "WAVLM_BASE":
                bundle = torchaudio.pipelines.WAVLM_BASE
            else:
                raise ValueError(f"Unsupported model type: {self.model_name} (only WAVLM_LARGE/WAVLM_BASE are supported)")
            self.model = bundle.get_model()
        elif self.wavlm_source == "microsoft":
            from .WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(self.wavlm_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            self.model = model
        else:
            raise ValueError(f"Unsupported wavlm_source: {self.wavlm_source} (only torchaudio/microsoft are supported)")

        
        # 冻结所有参数（仅用于特征提取，不微调）
        for param in self.model.parameters():
            param.requires_grad = False

    def _extract_multi_layer_features(self, audio):
        """单次WavLM调用提取三个指定层的特征（降低计算复杂度）"""
        # 输入形状转换: (batch, 1, time) -> (batch, time)
        audio = audio.squeeze(1)
        T_audio = audio.shape[1]
        T_target_length = T_audio//self.downsampling_ratio
        
        # 仅计算到最大层的特征（避免冗余计算）
        max_layer = max(self.layers)
        if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
            with torch.no_grad():  # 禁用梯度计算，进一步提升效率
                features, _ = self.model.extract_features(audio, num_layers=max_layer)
        
            # 提取指定三个层的特征（WavLM返回的features索引为 layer-1）
            extracted_feats = []
            for layer in self.layers:
                if layer > max_layer:
                    raise IndexError(f"Layer {layer} exceeds the maximum computed layer {max_layer}")
                feat = features[layer - 1]  # 形状: (batch, time, feat_dim)
                feat_for_sample = feat.permute(0, 2, 1)
                feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
                feat_back = feat_interpolated.permute(0, 2, 1)
                extracted_feats.append(feat_back)
        else:
            with torch.no_grad():
                # 由于微软的WavLM实现可能不支持一次性提取多层特征，我们需要逐层提取
                #这种情况下提取单层的代码为 feature, _ = self.model.extract_features(audio, output_layer=self.layers)，feature就是那一层的特征，我们需要从列表中逐个提取并插值
                extracted_feats = []
                for layer in self.layers:
                    feature, _ = self.model.extract_features(audio, output_layer=layer)
                    feat_for_sample = feature.permute(0, 2, 1)
                    feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
                    feat_back = feat_interpolated.permute(0, 2, 1)
                    extracted_feats.append(feat_back)
        return extracted_feats
    
    def _extract_single_layer_feature(self, audio, layer_num):
        """如果layers参数是单个数字，则只提取该层特征"""
        audio = audio.squeeze(1)
        T_audio = audio.shape[1]
        T_target_length = T_audio//self.downsampling_ratio
        with torch.no_grad():
            if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
                features, _ = self.model.extract_features(audio, num_layers=layer_num)
                feat = features[layer_num - 1]  # 形状: (batch, time, feat_dim)
            else:   
                feat, _ = self.model.extract_features(audio, output_layer=layer_num)
        
        feat_for_sample = feat.permute(0, 2, 1)
        feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
        feat_back = feat_interpolated.permute(0, 2, 1)
        
        return feat_back

    def forward(self, x, return_info=True):
        """
        输入: x -> (batch, 1, time) （音频波形）
        输出: latent -> (batch, latent_dim, time) （投影后的特征）
              encoder_info -> dict （包含最后一层WavLM特征）
        """
        # 1. 单次提取三个层的特征
        if self.layers is not None and isinstance(self.layers, (list, tuple)):
            layer_feats = self._extract_multi_layer_features(x)  # [f1, f2, f3], 各为 (B, T, D)
            concat_feat = torch.cat(layer_feats, dim=-1)
        else:
            layer_feats = self._extract_single_layer_feature(x, self.layers)  # [f], f为 (B, T, D)
            concat_feat = layer_feats
        #提取对齐层特征
        if self.align_layer is not None:
            align_feat = self._extract_single_layer_feature(x, self.align_layer)  # (B, T, D)
        # 3. 多头注意力投影 -> (batch, time, latent_dim)
        proj_feat = self.proj(concat_feat)
        
        # 4. 调整维度为 (batch, latent_dim, time)
        latent = proj_feat.permute(0, 2, 1)
        
        # 5. 保留对齐层的WavLM特征
        encoder_info = {"wavlm_feat": align_feat} if return_info else None
        
        return latent, encoder_info


class SFMEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        model_name="WAVLM_LARGE",
        layers=23,  # 要求传入三元数组或单个数字，单个数字代表某一层，三元数组代表三层
        align_layer=23, #后续进行语义对齐的层
        num_atten_layers=1,
        is_quant=False,
        downsampling_ratio = 400,
        wavlm_source = "torchaudio",
        wavlm_ckpt = None,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.align_layer = align_layer
        self.layers = layers
        self.downsampling_ratio = downsampling_ratio
        self.wavlm_source = wavlm_source
        self.wavlm_ckpt = wavlm_ckpt
        
        # 输入合法性检查
        #self._validate_layers() if self.layers is not None else None
        
        # 加载WavLM并冻结参数
        self._load_wavlm()
        
        # 基础配置（根据模型类型确定特征维度）
        self.feat_dim = 1024 if model_name == "WAVLM_LARGE" else 768
        #如果layers是三元数组，则总特征维度是feat_dim的三倍，如果layers是一个数字，则总特征维度就是feat_dim
        self.sum_feat_dim = self.feat_dim
        self.num_heads = kwargs.get("num_heads", max(1, self.sum_feat_dim // self.latent_dim))
        # 三个层特征拼接后的总维度

        self.proj = AttnProjection(
            in_dim=self.sum_feat_dim,
            out_dim=self.latent_dim,
            num_heads=self.num_heads,
            num_layers=num_atten_layers,
            is_quant=is_quant
        )

        self.proj_final = AttnProjection(
            in_dim=3*self.latent_dim,
            out_dim=self.latent_dim,
            num_heads=self.num_heads,
            num_layers=num_atten_layers,
            is_quant=is_quant
        )

    def _validate_layers(self):
        """验证layers参数是否为合法三元数组"""
        if not isinstance(self.layers, (list, tuple)) or len(self.layers) != 3:
            raise ValueError(f"layers must be an array of length 3, got: {self.layers}")
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, int) or layer <= 0:
                raise ValueError(f"layers[{idx}] must be a positive integer, got: {layer}")

    def _load_wavlm(self):
        """加载预训练WavLM并冻结参数"""
        if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
            if self.model_name == "WAVLM_LARGE":
                bundle = torchaudio.pipelines.WAVLM_LARGE
            elif self.model_name == "WAVLM_BASE":
                bundle = torchaudio.pipelines.WAVLM_BASE
            else:
                raise ValueError(f"Unsupported model type: {self.model_name} (only WAVLM_LARGE/WAVLM_BASE are supported)")
            self.model = bundle.get_model()
        elif self.wavlm_source == "microsoft":
            from .WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(self.wavlm_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            self.model = model
        else:
            raise ValueError(f"Unsupported wavlm_source: {self.wavlm_source} (only torchaudio/microsoft are supported)")

        
        # 冻结所有参数（仅用于特征提取，不微调）
        for param in self.model.parameters():
            param.requires_grad = False

    def _extract_multi_layer_features(self, audio):
        """单次WavLM调用提取三个指定层的特征（降低计算复杂度）"""
        # 输入形状转换: (batch, 1, time) -> (batch, time)
        audio = audio.squeeze(1)
        T_audio = audio.shape[1]
        T_target_length = T_audio//self.downsampling_ratio
        
        # 仅计算到最大层的特征（避免冗余计算）
        max_layer = max(self.layers)
        if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
            with torch.no_grad():  # 禁用梯度计算，进一步提升效率
                features, _ = self.model.extract_features(audio, num_layers=max_layer)
        
            # 提取指定三个层的特征（WavLM返回的features索引为 layer-1）
            extracted_feats = []
            for layer in self.layers:
                if layer > max_layer:
                    raise IndexError(f"Layer {layer} exceeds the maximum computed layer {max_layer}")
                feat = features[layer - 1]  # 形状: (batch, time, feat_dim)
                feat_for_sample = feat.permute(0, 2, 1)
                feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
                feat_back = feat_interpolated.permute(0, 2, 1)
                extracted_feats.append(feat_back)
        else:
            with torch.no_grad():
                # 由于微软的WavLM实现可能不支持一次性提取多层特征，我们需要逐层提取
                #这种情况下提取单层的代码为 feature, _ = self.model.extract_features(audio, output_layer=self.layers)，feature就是那一层的特征，我们需要从列表中逐个提取并插值
                extracted_feats = []
                for layer in self.layers:
                    feature, _ = self.model.extract_features(audio, output_layer=layer)
                    feat_for_sample = feature.permute(0, 2, 1)
                    feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
                    feat_back = feat_interpolated.permute(0, 2, 1)
                    extracted_feats.append(feat_back)
        return extracted_feats
    
    def _extract_single_layer_feature(self, audio, layer_num):
        """如果layers参数是单个数字，则只提取该层特征"""
        audio = audio.squeeze(1)
        T_audio = audio.shape[1]
        T_target_length = T_audio//self.downsampling_ratio
        with torch.no_grad():
            if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
                features, _ = self.model.extract_features(audio, num_layers=layer_num)
                feat = features[layer_num - 1]  # 形状: (batch, time, feat_dim)
            else:   
                feat, _ = self.model.extract_features(audio, output_layer=layer_num)
        
        feat_for_sample = feat.permute(0, 2, 1)
        feat_interpolated = F.interpolate(feat_for_sample, size=T_target_length, mode='linear', align_corners=False) if self.downsampling_ratio != 320 else feat_for_sample
        feat_back = feat_interpolated.permute(0, 2, 1)
        
        return feat_back

    def forward(self, x, return_info=True):
        """
        输入: x -> (batch, 1, time) （音频波形）
        输出: latent -> (batch, latent_dim, time) （投影后的特征）
              encoder_info -> dict （包含最后一层WavLM特征）
        """
        # 1. 单次提取三个层的特征
        if self.layers is not None and isinstance(self.layers, (list, tuple)):
            layer_feats = self._extract_multi_layer_features(x)  # [f1, f2, f3], 各为 (B, T, D)
            #对f1, f2, f3先进行self.proj投影，然后拼接，然后进行self.proj_final投影
            projected_feats = [self.proj(feat) for feat in layer_feats]
            concat_feat = torch.cat(projected_feats, dim=-1)
            concat_feat = self.proj_final(concat_feat)
            proj_feat = concat_feat
        else:
            layer_feats = self._extract_single_layer_feature(x, self.layers)  # [f], f为 (B, T, D)
            concat_feat = layer_feats
            proj_feat = self.proj(concat_feat)
        #提取对齐层特征
        if self.align_layer is not None:
            align_feat = self._extract_single_layer_feature(x, self.align_layer)  # (B, T, D)
            # 3. 多头注意力投影 -> (batch, time, latent_dim)
            
        
        # 4. 调整维度为 (batch, latent_dim, time)
        latent = proj_feat.permute(0, 2, 1)
        
        # 5. 保留对齐层的WavLM特征
        encoder_info = {"wavlm_feat": align_feat} if return_info else None
        
        return latent, encoder_info

class DACWavLMEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from .dac import Encoder as DACWavLMEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACWavLMEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACWavLM_2_EncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from .dac2 import Encoder2 as DACWavLMEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACWavLMEncoder(**kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x


class FocalCodecEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        in_channels=1,
        wavlm_hidden_dims=(512,) + (512,) * 4 + (512,) * 2,
        wavlm_kernel_sizes=(10,) + (3,) * 4 + (2,) * 2,
        wavlm_strides=(5,) + (2,) * 4 + (2,) * 2,
        wavlm_num_layers=6,
        wavlm_dim=1024,
        wavlm_ffn_dim=4096,
        wavlm_num_heads=16,
        wavlm_num_buckets=320,
        wavlm_max_distance=800,
        wavlm_max_cached_steps=2048,
        wavlm_dropout=0.0,
        wavlm_conv_pos=128,
        wavlm_conv_pos_groups=16,
        wavlm_causal=False,
        wavlm_window_size=512,
        wavlm_lookahead_size=3,
        wavlm_use_flex_attention=False,
        compressor_hidden_dims=(1024, 512, 256),
        compressor_downscale_factors=(1, 1, 1),
        compressor_focal_window=7,
        compressor_focal_level=2,
        compressor_focal_factor=2,
        compressor_dropout=0.0,
        compressor_use_post_norm=False,
        compressor_use_layerscale=False,
        compressor_layerscale_init=1e-4,
        compressor_tanhscale_init=0.5,
        compressor_normalize_modulator=False,
        compressor_causal=False,
        compressor_window_size=512,
        l2_normalize=True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        classes = _get_focalcodec_classes()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.l2_normalize = l2_normalize

        self.wavlm = classes["WavLM"](
            hidden_dims=wavlm_hidden_dims,
            kernel_sizes=wavlm_kernel_sizes,
            strides=wavlm_strides,
            num_layers=wavlm_num_layers,
            dim=wavlm_dim,
            ffn_dim=wavlm_ffn_dim,
            num_heads=wavlm_num_heads,
            num_buckets=wavlm_num_buckets,
            max_distance=wavlm_max_distance,
            max_cached_steps=wavlm_max_cached_steps,
            dropout=wavlm_dropout,
            conv_pos=wavlm_conv_pos,
            conv_pos_groups=wavlm_conv_pos_groups,
            causal=wavlm_causal,
            window_size=wavlm_window_size,
            lookahead_size=wavlm_lookahead_size,
            use_flex_attention=wavlm_use_flex_attention,
        )
        self.compressor = classes["FocalEncoder"](
            input_dim=wavlm_dim,
            output_dim=latent_dim,
            hidden_dims=compressor_hidden_dims,
            downscale_factors=compressor_downscale_factors,
            focal_window=compressor_focal_window,
            focal_level=compressor_focal_level,
            focal_factor=compressor_focal_factor,
            dropout=compressor_dropout,
            use_post_norm=compressor_use_post_norm,
            use_layerscale=compressor_use_layerscale,
            layerscale_init=compressor_layerscale_init,
            tanhscale_init=compressor_tanhscale_init,
            normalize_modulator=compressor_normalize_modulator,
            causal=compressor_causal,
            window_size=compressor_window_size,
        )

        self.downsampling_ratio = int(
            np.prod(wavlm_strides) * np.prod(compressor_downscale_factors)
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected audio tensor of shape (batch, channels, time), got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        feats, *_ = self.wavlm(x[:, 0])
        latents, *_ = self.compressor(feats)
        if self.l2_normalize:
            latents = F.normalize(latents, dim=-1)
        return latents.permute(0, 2, 1)


class FocalCodecDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        out_channels=1,
        decompressor_output_dim=1024,
        decompressor_hidden_dims=(256, 512, 1024),
        decompressor_upscale_factors=(1, 1, 1),
        decompressor_focal_window=7,
        decompressor_focal_level=2,
        decompressor_focal_factor=2,
        decompressor_dropout=0.0,
        decompressor_use_post_norm=False,
        decompressor_use_layerscale=False,
        decompressor_layerscale_init=1e-4,
        decompressor_tanhscale_init=0.5,
        decompressor_normalize_modulator=False,
        decompressor_causal=False,
        decompressor_window_size=512,
        decompressor_last_window_size=512,
        decompressor_lookahead_size=3,
        vocos_num_layers=8,
        vocos_dim=512,
        vocos_ffn_dim=1536,
        vocos_kernel_size=7,
        vocos_n_fft=1024,
        vocos_hop_length=320,
        vocos_layerscale_init=None,
        vocos_causal=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        classes = _get_focalcodec_classes()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.decompressor = classes["FocalDecoder"](
            input_dim=latent_dim,
            output_dim=decompressor_output_dim,
            hidden_dims=decompressor_hidden_dims,
            upscale_factors=decompressor_upscale_factors,
            focal_window=decompressor_focal_window,
            focal_level=decompressor_focal_level,
            focal_factor=decompressor_focal_factor,
            dropout=decompressor_dropout,
            use_post_norm=decompressor_use_post_norm,
            use_layerscale=decompressor_use_layerscale,
            layerscale_init=decompressor_layerscale_init,
            tanhscale_init=decompressor_tanhscale_init,
            normalize_modulator=decompressor_normalize_modulator,
            causal=decompressor_causal,
            window_size=decompressor_window_size,
            last_window_size=decompressor_last_window_size,
            lookahead_size=decompressor_lookahead_size,
        )
        self.vocoder = classes["Vocos"](
            input_dim=decompressor_output_dim,
            num_layers=vocos_num_layers,
            dim=vocos_dim,
            ffn_dim=vocos_ffn_dim,
            kernel_size=vocos_kernel_size,
            layerscale_init=vocos_layerscale_init,
            n_fft=vocos_n_fft,
            hop_length=vocos_hop_length,
            causal=vocos_causal,
        )
        self.upsample_ratio = int(
            np.prod(decompressor_upscale_factors) * vocos_hop_length
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected latent tensor of shape (batch, latent_dim, time), got {tuple(x.shape)}"
            )

        feats, *_ = self.decompressor(x.permute(0, 2, 1))
        audio, *_ = self.vocoder(feats)
        audio = audio[:, None, :]
        if self.out_channels > 1:
            audio = audio.repeat(1, self.out_channels, 1)
        return audio


class WavLM6FocalCodecEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        in_channels=1,
        wavlm_ckpt=None,
        wavlm_num_layers=6,
        wavlm_frozen_layers=5,
        freeze_feature_extractor=True,
        freeze_post_extract_proj=True,
        freeze_input_layer_norm=True,
        freeze_pos_conv=True,
        freeze_encoder_layer_norm=True,
        compressor_hidden_dims=(1024, 512, 256),
        compressor_downscale_factors=(1, 1, 1),
        compressor_focal_window=7,
        compressor_focal_level=2,
        compressor_focal_factor=2,
        compressor_dropout=0.0,
        compressor_use_post_norm=False,
        compressor_use_layerscale=False,
        compressor_layerscale_init=1e-4,
        compressor_tanhscale_init=0.5,
        compressor_normalize_modulator=False,
        compressor_causal=False,
        compressor_window_size=512,
        l2_normalize=True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        classes = _get_focalcodec_classes()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.wavlm_num_layers = wavlm_num_layers
        self.l2_normalize = l2_normalize

        self.wavlm, wavlm_cfg = _load_wavlm_from_checkpoint(
            checkpoint_path=wavlm_ckpt,
            num_layers=wavlm_num_layers,
        )
        _freeze_wavlm_prefix(
            self.wavlm,
            frozen_layers=wavlm_frozen_layers,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_post_extract_proj=freeze_post_extract_proj,
            freeze_input_layer_norm=freeze_input_layer_norm,
            freeze_pos_conv=freeze_pos_conv,
            freeze_encoder_layer_norm=freeze_encoder_layer_norm,
        )

        self.compressor = classes["FocalEncoder"](
            input_dim=wavlm_cfg.encoder_embed_dim,
            output_dim=latent_dim,
            hidden_dims=compressor_hidden_dims,
            downscale_factors=compressor_downscale_factors,
            focal_window=compressor_focal_window,
            focal_level=compressor_focal_level,
            focal_factor=compressor_focal_factor,
            dropout=compressor_dropout,
            use_post_norm=compressor_use_post_norm,
            use_layerscale=compressor_use_layerscale,
            layerscale_init=compressor_layerscale_init,
            tanhscale_init=compressor_tanhscale_init,
            normalize_modulator=compressor_normalize_modulator,
            causal=compressor_causal,
            window_size=compressor_window_size,
        )

        conv_feature_layers = eval(wavlm_cfg.conv_feature_layers)
        wavlm_stride = int(np.prod([stride for _, _, stride in conv_feature_layers]))
        self.downsampling_ratio = int(
            wavlm_stride * np.prod(compressor_downscale_factors)
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected audio tensor of shape (batch, channels, time), got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        feats, _ = self.wavlm.extract_features(x[:, 0], output_layer=self.wavlm_num_layers)
        latents, *_ = self.compressor(feats)
        if self.l2_normalize:
            latents = F.normalize(latents, dim=-1)
        return latents.permute(0, 2, 1)


class WavLM6FocalCodecDecoder(FocalCodecDecoder):
    pass


def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "dac":
        dac_config = encoder_config["config"]

        encoder = DACEncoderWrapper(**dac_config)
    else:
        raise ValueError(f"Unsupported encoder type for JMAS-VAE release: {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "bigvgan":
        bigvgan_config = decoder_config["config"]

        decoder = BigVGANDecoderWrapper(
            **bigvgan_config
        )
    else:
        raise ValueError(f"Unsupported decoder type for JMAS-VAE release: {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"]) if "decoder" in ae_config else None

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False) if decoder is not None else False

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )

def create_diffAE_from_config(config: Dict[str, Any]):
    
    from .diffusion import DAU1DCondWrapper, UNet1DCondWrapper, DiTWrapper

    diffae_config = config["model"]

    if "encoder" in diffae_config:
        encoder = create_encoder_from_config(diffae_config["encoder"])
    else:
        encoder = None

    if "decoder" in diffae_config:
        decoder = create_decoder_from_config(diffae_config["decoder"])
    else:
        decoder = None

    diffusion_model_type = diffae_config["diffusion"]["type"]

    if diffusion_model_type == "DAU1d":
        diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "adp_1d":
        diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "dit":
        diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

    latent_dim = diffae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = diffae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = diffae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    bottleneck = diffae_config.get("bottleneck", None)

    pretransform = diffae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    diffusion_downsampling_ratio = None,

    if diffusion_model_type == "DAU1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
    elif diffusion_model_type == "adp_1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
    elif diffusion_model_type == "dit":
        diffusion_downsampling_ratio = 1

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        diffusion_downsampling_ratio=diffusion_downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )
