import random 

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio.transforms as T
import torchaudio
from typing import Optional
from .transformer import ContinuousTransformer
class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class DiscreteBottleneck(Bottleneck):
    def __init__(self, num_quantizers, codebook_size, tokens_id):
        super().__init__(is_discrete=True)

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.tokens_id = tokens_id

    def decode_tokens(self, codes, **kwargs):
        raise NotImplementedError


class SoftNormBottleneck(Bottleneck):
    def __init__(self, dim = 32, noise_augment_dim=0, noise_regularize = False):
        super().__init__(is_discrete=False)

        self.noise_augment_dim = noise_augment_dim
        self.scaling_factor = nn.Parameter(torch.ones(1,dim,1))
        self.bias = nn.Parameter(torch.zeros(1,dim,1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1,noise_augment_dim,1))
        self.noise_regularize = noise_regularize
        #self.norm = RunningInstanceNorm(dim, momentum = 0.999, trainable_gain = False)

    def encode(self, x, return_info=False):
        info = {}

        x = x * self.scaling_factor + self.bias
        #x = rearrange(x, "b c n -> b n c")
        #x = self.norm(x)
        #x = rearrange(x, "b n c -> b c n")

        if self.training and return_info:
            var = (x.std(dim=-1) ** 2).clip(min = 1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-1)
            loss = (mean * mean + var - logvar - 1).mean()
            var = (x.std(dim=-2) ** 2).clip(min = 1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-2)
            loss = loss + 0.4 * (mean * mean + var - logvar - 1).mean()
            info["softnorm_loss"] = loss 
        
        if return_info:
            return x, info
        
        return x

    def decode(self, x):
        if self.noise_regularize and self.training:
            scaling = x.std(dim = -1)
            noise = torch.randn_like(x) * scaling.unsqueeze(-1) * 1e-2
            x = x + noise
        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x

class TanhBottleneck(Bottleneck):
    def __init__(self, scale=1.0):
        super().__init__(is_discrete=False)
        self.tanh = nn.Tanh()

        self.scale = scale

    def encode(self, x, return_info=False):
        info = {}

        x = x / self.scale

        x = torch.tanh(x)

        x = x * self.scale

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

#以下为新增部分
# SSL损失类
class SSLoss(nn.Module):
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=80,
                 projection_dim=16,
                 codebook_size=1024,
                 mask_time=40,
                 stride_time=1,
                 mask_prob=0.3):
        super().__init__()
        
        # 梅尔谱特征提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Best-RQ参数
        self.projection_dim = projection_dim
        self.codebook_size = codebook_size
        
        # 随机投影矩阵 (D_mel -> projection_dim)
        self.random_projection = nn.Linear(
            n_mels, 
            projection_dim, 
            bias=False
        )
        nn.init.xavier_uniform_(self.random_projection.weight)
        self.random_projection.weight.requires_grad = False
        
        # 码本矩阵
        self.register_buffer('codebook', torch.randn(codebook_size, projection_dim))
        
        # 输出投影层 - 匹配模型输出维度
        self.output_projection = nn.Linear(512, codebook_size)
        
        # 掩码参数
        self.mask_time = mask_time
        self.stride_time = stride_time
        self.mask_prob = mask_prob
        
        # 计算时间步数量
        self.num_time_steps = int(mask_time // stride_time)

    def masking(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """生成掩码矩阵 (0=未掩码, 1=掩码)"""
        batch_size, num_steps, _ = input_values.size()
        
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_time_steps,
            device=input_values.device, dtype=torch.bool
        )
        
        for batch in range(batch_size):
            valid_steps = int(input_lengths[batch])
            if valid_steps <= 0:
                continue
                
            # 随机选择掩码起始位置
            candidates = list(range(valid_steps))
            k = max(1, int(self.mask_prob * valid_steps))
            start_indices = torch.tensor(
                random.sample(candidates, k=min(k, len(candidates))), 
                device=input_values.device, 
                dtype=torch.long
            )
            
            # 扩展掩码范围
            for i in range(self.num_time_steps):
                indices = torch.clamp(start_indices + i, 0, num_steps + self.num_time_steps - 1)
                time_mask_indices[batch, indices] = 1
                
        # 调整掩码形状
        time_mask_indices = time_mask_indices[:, :-self.num_time_steps]
        return time_mask_indices

    def forward(self, audio, model_output, input_lengths=None):
        """计算自监督损失"""
        batch_size, T_out, _ = model_output.shape
        audio = audio.squeeze(1)  # 若audio是[B, 1, T]，转为[B, T]；若有多余维度，移除所有大小为1的维度
        # 1. 提取梅尔谱特征
        mel_features = self.mel_transform(audio)
        mel_features = self.amplitude_to_db(mel_features)  # (B, n_mels, T_audio)
        
        # 2. 维度调整: (B, n_mels, T_audio) -> (B, T_audio, n_mels)
        audio_features = mel_features.transpose(1, 2)
        
        # 3. 下采样对齐时间维度（保留interpolate部分）
        T_audio = audio_features.shape[1]
        if T_audio != T_out:
            audio_features = F.interpolate(
                audio_features.transpose(1, 2),  # (B, D_mel, T_audio)
                size=T_out,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (B, T_out, D_mel)
        
        # 4. 生成掩码
        if input_lengths is None:
            input_lengths = torch.full((batch_size,), T_out, device=audio_features.device)
        
        mask_indices = self.masking(audio_features, input_lengths)
        
        # 5. 生成标签 (从音频特征)
        masked_audio = audio_features[mask_indices]  # (N, D_mel)
        projected_features = self.random_projection(masked_audio)  # (N, projection_dim)
        distances = torch.cdist(projected_features.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        labels = torch.argmin(distances, dim=-1)  # (N,)
        
        # 6. 生成预测 (从模型输出)
        masked_model_output = model_output[mask_indices]  # (N, 512)
        logits = self.output_projection(masked_model_output)  # (N, codebook_size)
        
        # 7. 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

# VAE-SSL瓶颈层
class VAESSLBottleneck(Bottleneck):
    def __init__(self, mask_prob=0.3, mask_time=40, stride_time=1):
        super().__init__(is_discrete=False)

        self.transformer = ContinuousTransformer(
            dim=512,
            depth=6,
            dim_in=64,
            dim_out=64,
        )
        self.encode_layer = 4
        self.encode_project = nn.Linear(512, 64)

        # 初始化SSL损失计算器
        self.ssl_loss = SSLoss(
            mask_time=mask_time,
            stride_time=stride_time,
            mask_prob=mask_prob
        )

    def encode(self, x, audio, return_info=False, plot_mask=False, **kwargs):
        info = {}

        # VAE采样
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        x_initial = x.clone()
        info["kl"] = kl

        # 训练时应用掩码
        mask_original = None
        if self.training:
            batch_size, channels, seq_len = x.shape
            x_for_masking = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            
            # 生成掩码
            input_lengths = torch.full((batch_size,), seq_len, device=x.device)
            mask_indices = self.ssl_loss.masking(x_for_masking, input_lengths)
            mask_original = mask_indices[:, :seq_len]
            mask_expanded = mask_original.unsqueeze(1).expand(-1, channels, -1)
            
            # 应用掩码 (替换为随机噪声)
            x_masked = x.clone()
            x_masked[mask_expanded] = torch.normal(0, 0.1, size=x_masked[mask_expanded].shape, device=x.device)
            x = x_masked
            info["mask"] = mask_original
            
        # Transformer处理
        _, out_info = self.transformer(x.permute(0, 2, 1), return_info=True)  # (B, T, C)
        layer_outputs = out_info["hidden_states"]
        model_output = layer_outputs[-1]  # (B, T, 512)
        
        # 计算SSL损失
        valid_lengths = torch.full((x.shape[0],), x.shape[2], device=x.device)
        ssl_loss = self.ssl_loss(audio, model_output, valid_lengths)
        info["ssl"] = ssl_loss

        # 编码输出
        #x = self.encode_project(layer_outputs[self.encode_layer])
        #x = x.permute(0, 2, 1)  # 恢复为 (B, C, T)

        if return_info:
            return x_initial, info
        return x_initial

    def decode(self, x):
        return x
#以上为新增部分

class WAVLM_loss(nn.Module):
    def __init__(self,
        model_name: str = "WAVLM_LARGE",
        m1: Optional[float] = None,
        m2: Optional[float] = None,
        weight: float = 1.0,
        wavlm_layers: Optional[int] = None,
        zA_dim: Optional[int] = 64,
        wavlm_source: Optional[str] = "torchaudio",
        wavlm_ckpt: Optional[str] = None
    ):
        super().__init__()
        
        self.weight = weight  # 总损失权重
        self.model_name = model_name
        self.wavlm_source = wavlm_source
        if m1 is not None and m2 is not None:
            self.m1 = m1  # 余弦相似度边际
            self.m2 = m2  # 距离矩阵相似度边际
        else:
            self.m1 = 0.5
            self.m2 = 0.25
        self.layers = wavlm_layers
        self.wavlm_ckpt = wavlm_ckpt
        # 加载预训练WavLM模型
        self._load_wavlm()
        
        
        # 线性投影层（将VAE输出x投影到WavLM特征维度）
        if model_name == "WAVLM_LARGE":
            self.proj = nn.Linear(in_features=zA_dim, out_features=1024)  # WAVLM_LARGE输出维度为1024
        else:  # 如WAVLM_BASE
            self.proj = nn.Linear(in_features=zA_dim, out_features=768)   # WAVLM_BASE输出维度为768

    def _load_wavlm(self):
        if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
            """使用torchaudio加载预训练WavLM模型"""
            if self.model_name == "WAVLM_LARGE":
                bundle = torchaudio.pipelines.WAVLM_LARGE
            elif self.model_name == "WAVLM_BASE":
                bundle = torchaudio.pipelines.WAVLM_BASE
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")
            self.model = bundle.get_model()
        else:
            raise ValueError("JMAS-VAE release only supports torchaudio WavLM")
        
        
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

    def _extract_ssl_feature(self, audio):
        """使用WavLM提取音频特征"""
        # audio形状: (batch, 1, time) -> 转换为 (batch, time)
        audio = audio.squeeze(1)
        
        # 提取特征
        with torch.no_grad():
            features, _ = self.model.extract_features(audio, num_layers = self.layers)  # 使用所有层特征
            f = features[-1]  # (batch, time, feat_dim)
            
        # 默认使用最后一层特征
        
        
        return f

    def _interpolate_features(self, f, target_length):
        """将特征的时间维度插值到目标长度"""
        f_transposed = f.permute(0, 2, 1)  # (batch, feat_dim, time)
        f_interpolated = F.interpolate(f_transposed, size=target_length, mode='linear', align_corners=False)
        return f_interpolated.permute(0, 2, 1)  # (batch, target_length, feat_dim)

    def marginal_cosine_loss(self, x_proj, f):
        """边际余弦相似度损失 L_mcos"""
        cos_sim = F.cosine_similarity(x_proj, f, dim=-1)  # (batch, time)
        loss = F.relu(1 - self.m1 - cos_sim).mean()  # 应用边际和ReLU
        return loss

    def marginal_distance_matrix_loss(self, x_proj, f):
        """边际距离矩阵相似度损失 L_mdms"""
        batch, time, dim = x_proj.shape
        
        # 展平时序维度
        x_flat = x_proj.reshape(-1, dim)
        f_flat = f.reshape(-1, dim)
        
        # 计算所有时序对的余弦相似度矩阵
        x_cos = F.cosine_similarity(x_flat.unsqueeze(1), x_flat.unsqueeze(0), dim=-1)
        f_cos = F.cosine_similarity(f_flat.unsqueeze(1), f_flat.unsqueeze(0), dim=-1)
        
        # 计算所有时序对的余弦相似度矩阵
        #x_cos = F.cosine_similarity(x_proj.unsqueeze(2), x_proj.unsqueeze(1), dim=-1)  # (batch, time, time)
        #f_cos = F.cosine_similarity(f.unsqueeze(2), f.unsqueeze(1), dim=-1)  # (batch, time, time)
        # 计算相似度差异的绝对值
        diff = torch.abs(x_cos - f_cos)
        
        # 应用边际和ReLU
        loss = F.relu(diff - self.m2).mean()
        return loss

    def forward(self, audio, x):
        """计算总SSL损失（WavLM Loss）"""
        x = x.permute(0, 2, 1)  # (batch, time, 512)
        x_proj = self.proj(x)   # (batch, time, feat_dim)
        
        target_length = x.shape[1]
        
        f = self._extract_ssl_feature(audio)  # (batch, time_wavlm, feat_dim)
        f = self._interpolate_features(f, target_length)  # (batch, target_length, feat_dim)
        
        # 计算两个子损失
        loss_mcos = self.marginal_cosine_loss(x_proj, f)
        loss_mdms = self.marginal_distance_matrix_loss(x_proj, f)
        
        # 总损失
        total_loss = self.weight * (loss_mcos + loss_mdms)
        return loss_mcos, loss_mdms


class DOUBLEDISTILL_loss(WAVLM_loss):
    def __init__(
        self,
        model_name: str = "WAVLM_LARGE",
        m1: Optional[float] = None,
        m2: Optional[float] = None,
        weight: float = 1.0,
        wavlm_layers: Optional[tuple] = None,
        zA_dim: Optional[int] = 64,
        wavlm_source: Optional[str] = "microsoft",
        wavlm_ckpt: Optional[str] = None
    ):
        if wavlm_layers is None or len(wavlm_layers) != 2:
            raise ValueError("DOUBLEDISTILL_loss expects wavlm_layers to contain exactly two layer indices.")

        low_layer, high_layer = sorted(int(layer) for layer in wavlm_layers)
        if low_layer < 1:
            raise ValueError("WavLM layer indices must be positive integers.")
        if low_layer == high_layer:
            raise ValueError("DOUBLEDISTILL_loss expects two different WavLM layer indices.")

        self.low_layer = low_layer
        self.high_layer = high_layer

        super().__init__(
            model_name=model_name,
            m1=m1,
            m2=m2,
            weight=weight,
            wavlm_layers=high_layer,
            zA_dim=zA_dim,
            wavlm_source=wavlm_source,
            wavlm_ckpt=wavlm_ckpt,
        )

    def _extract_double_ssl_features(self, audio):
        """提取双层WavLM特征：低层用于mdms，高层用于mcos。"""
        audio = audio.squeeze(1)

        with torch.no_grad():
            if self.wavlm_source == "torchaudio" or self.wavlm_source is None:
                features, _ = self.model.extract_features(audio, num_layers=self.high_layer)
                low_feature = features[self.low_layer - 1]
                high_feature = features[self.high_layer - 1]
            else:
                (_, layer_results), _ = self.model.extract_features(
                    audio,
                    output_layer=self.high_layer,
                    ret_layer_results=True
                )
                low_feature = layer_results[self.low_layer - 1][0].transpose(0, 1)
                high_feature = layer_results[self.high_layer - 1][0].transpose(0, 1)

        return low_feature, high_feature

    def forward(self, audio, x):
        """低层计算mdms，高层计算mcos。"""
        x = x.permute(0, 2, 1)
        x_proj = self.proj(x)

        target_length = x.shape[1]

        f_low, f_high = self._extract_double_ssl_features(audio)
        f_low = self._interpolate_features(f_low, target_length)
        f_high = self._interpolate_features(f_high, target_length)

        loss_mdms = self.marginal_distance_matrix_loss(x_proj, f_low)
        loss_mcos = self.marginal_cosine_loss(x_proj, f_high)

        return loss_mcos, loss_mdms


class DOUBLEDISTILL(Bottleneck):
    def __init__(
        self,
        ssl_model_name: str = "WAVLM_LARGE",
        ssl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(is_discrete=False)
        self.wavlm_layers = kwargs.get('wavlm_layers')
        self.use_latent_subspace = kwargs.get('use_latent_subspace')
        self.ssl_ratio = kwargs.get('ssl_ratio')
        self.latent_dim = kwargs.get('latent_dim')
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.wavlm_source = kwargs.get('wavlm_source')
        self.wavlm_ckpt = kwargs.get('wavlm_ckpt')

        if self.use_latent_subspace:
            zA_dim = int(self.latent_dim * self.ssl_ratio)
        elif self.latent_dim is None:
            zA_dim = 64
        else:
            zA_dim = self.latent_dim

        self.ssl_loss = DOUBLEDISTILL_loss(
            model_name=ssl_model_name,
            weight=ssl_weight,
            wavlm_layers=self.wavlm_layers,
            zA_dim=zA_dim,
            m1=self.m1,
            m2=self.m2,
            wavlm_source=self.wavlm_source,
            wavlm_ckpt=self.wavlm_ckpt
        )

    def encode(self, x, return_info=False, inference_only = False, plot_mask=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl

        audio = kwargs.get("audio")
        if self.use_latent_subspace:
            total_dim = x.shape[1]
            zA_dim = int(total_dim * self.ssl_ratio)
            x_ssl = x[:, :zA_dim, :]
        else:
            x_ssl = x

        if not inference_only and audio is not None:
            ssl_loss1, ssl_loss2 = self.ssl_loss(audio, x_ssl)
            info["ssl_1"] = ssl_loss1
            info["ssl_2"] = ssl_loss2
        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x


class JMASVAESSL_new_Bottleneck(Bottleneck):
    def __init__(self, 
        ssl_model_name: str = "WAVLM_LARGE",
        ssl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(is_discrete=False)
        self.wavlm_layers = kwargs.get('wavlm_layers')
        self.use_latent_subspace = kwargs.get('use_latent_subspace')
        self.ssl_ratio = kwargs.get('ssl_ratio')
        self.latent_dim = kwargs.get('latent_dim')
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.align_method = kwargs.get('align_method')
        self.wavlm_source = kwargs.get('wavlm_source')
        self.wavlm_ckpt = kwargs.get('wavlm_ckpt')
        if self.align_method not in (None, "up"):
            raise ValueError("JMAS-VAE release only supports align_method=None or 'up'")

        zA_dim = int(self.latent_dim * self.ssl_ratio) if self.use_latent_subspace else self.latent_dim
        self.ssl_loss = WAVLM_loss(
            model_name=ssl_model_name,
            weight=ssl_weight,
            wavlm_layers=self.wavlm_layers,
            zA_dim=zA_dim or 64,
            m1=self.m1,
            m2=self.m2,
            wavlm_source=self.wavlm_source,
            wavlm_ckpt=self.wavlm_ckpt,
        )


    def encode(self, x, return_info=False, inference_only = False, plot_mask=False, **kwargs):
        info = {}
        original_x = x.clone()

        # VAE采样
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl

        audio = kwargs.get("audio")
        #判断是否使用维度拆分
        if self.use_latent_subspace:
            total_dim = x.shape[1]
            zA_dim = int(total_dim*self.ssl_ratio)
            zA = x[:, :zA_dim, :]
            x_ssl = zA
        else:
            x_ssl = x

        if not inference_only and audio is not None:
            # 计算SSL损失
            ssl_loss1, ssl_loss2 = self.ssl_loss(audio, x_ssl)
            info["ssl_1"] = ssl_loss1
            info["ssl_2"] = ssl_loss2

        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x
