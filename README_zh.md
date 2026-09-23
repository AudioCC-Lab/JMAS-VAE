# JMAS-VAE

JMAS-VAE 是基于 Stable Audio Tools 精简得到的 16 kHz 语音 VAE 开源版本。当前仓库只保留两类 autoencoder 配置所需的训练与推理路径：

- `JMAS-VAE`：DAC encoder + VAE bottleneck + BigVGAN decoder，训练时额外使用 SSL/WavLM bottleneck loss。
- `Vanilla VAE`：相同 encoder/decoder 架构，使用普通 VAE bottleneck 训练。

推理时两者共用同一个轻量配置，因为 SSL/WavLM 只用于训练损失，不参与 latent 提取和重建。

> 论文：[https://arxiv.org/abs/2604.12383](https://arxiv.org/abs/2604.12383)
> 预训练模型：[https://huggingface.co/ch-cheng/JMAS-VAE](https://huggingface.co/ch-cheng/JMAS-VAE)

## 目录说明

- `configs/jmas_vae_train.json`：JMAS-VAE 训练配置，包含 SSL/WavLM bottleneck loss。
- `configs/vanilla_vae_train.json`：vanilla VAE 训练配置。
- `configs/inference.json`：JMAS-VAE 和 vanilla VAE 共用的推理配置，已去掉 SSL/WavLM 字段。
- `configs/local_dataset.json`：本地音频目录数据集模板。
- `train.py`：训练入口。
- `unwrap_model.py`：把 Lightning checkpoint 导出为推理 checkpoint，默认删除训练专用 SSL/WavLM 权重。
- `vae_recon.py`：latent 提取和音频重建入口。
- `stable_audio_tools/`：两类配置需要的模型、数据、损失和训练模块。

## 环境安装

推荐使用 Python 3.10，并根据本机 CUDA 版本先安装匹配的 PyTorch/Torchaudio。

```bash
git clone https://github.com/<your-org>/JMAS-VAE.git
cd JMAS-VAE
pip install -e .
```

如果希望手动控制 PyTorch 版本，先按照 PyTorch 官方说明安装 `torch` 和 `torchaudio`，再运行 `pip install -e .`。

## 训练

默认训练数据集以 LibriLight 为例。若使用本地 LibriLight 音频目录，修改 `configs/local_dataset.json` 中的 `path`：

```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "train",
            "path": "/path/to/librilight/audio_dir"
        }
    ]
}
```

训练 JMAS-VAE：

```bash
python train.py \
  --model-config configs/jmas_vae_train.json \
  --dataset-config configs/local_dataset.json \
  --name jmas_vae \
  --batch-size 8 \
  --num-workers 8 \
  --num-gpus 1 \
  --save-dir checkpoints
```

训练 vanilla VAE：

```bash
python train.py \
  --model-config configs/vanilla_vae_train.json \
  --dataset-config configs/local_dataset.json \
  --name vanilla_vae \
  --batch-size 8 \
  --num-workers 8 \
  --num-gpus 1 \
  --save-dir checkpoints
```

## 导出权重

训练得到的是 PyTorch Lightning checkpoint，推理前需要 unwrap。`unwrap_model.py` 默认会删除 SSL/WavLM 训练专用参数，因此导出的 JMAS-VAE checkpoint 可以直接配合 `configs/inference.json` 推理。

```bash
python unwrap_model.py \
  --model-config configs/jmas_vae_train.json \
  --ckpt-path /path/to/epoch-step.ckpt \
  --output jmas_vae_600k.ckpt
```

vanilla VAE：

```bash
python unwrap_model.py \
  --model-config configs/vanilla_vae_train.json \
  --ckpt-path /path/to/vanilla-lightning.ckpt \
  --output vanilla_vae.ckpt
```

## 提取 Latent

```bash
python vae_recon.py \
  --mode encode \
  --audios /path/to/wavs \
  --model_path jmas_vae_600k.ckpt \
  --model_config configs/inference.json \
  --latent_dir latents \
  --output_dir reconstructions \
  --audio_format all \
  --device cuda
```

`--audios` 支持音频文件、音频目录或 Kaldi 风格 `wav.scp`。目录输入默认递归扫描 `wav/flac/mp3/ogg/m4a/aac`；也可以用 `--audio_format wav,flac,mp3` 指定格式集合。

## 音频重建

```bash
python vae_recon.py \
  --mode decode \
  --model_path jmas_vae_600k.ckpt \
  --model_config configs/inference.json \
  --latent_dir latents \
  --output_dir reconstructions \
  --output_format wav \
  --device cuda
```

`--output_format` 支持 `wav/flac/mp3` 等 `soundfile` 当前环境可写的格式；若输出 MP3，请确保本机 `libsndfile/soundfile` 支持 MP3 编码。

## 致谢

本项目基于 Stable Audio Tools 修改而来，并保留 JMAS-VAE 与 vanilla VAE autoencoder 实验所需的核心代码。

同时感谢 [Semantic-VAE](https://github.com/ZhikangNiu/Semantic-VAE) 项目为本文提供的语义对齐思想启发。
