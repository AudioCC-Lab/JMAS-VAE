# JMAS-VAE

JMAS-VAE is a lightweight release of a 16 kHz speech VAE based on the Stable Audio Tools codebase. This repository keeps only the code path needed to train and run two autoencoder variants:

- `JMAS-VAE`: DAC encoder + VAE bottleneck trained with an additional SSL/WavLM bottleneck loss.
- `Vanilla VAE`: the same encoder/decoder architecture trained with the standard VAE bottleneck.

For inference, both models use the same lightweight configuration because the SSL/WavLM module is only used as a training loss and is not needed for latent extraction or waveform reconstruction.

> Paper: [![arXiv](https://img.shields.io/badge/arXiv-2604.12383-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.12383)
> Pretrained models: [![Hugging Face](https://img.shields.io/badge/HuggingFace-ch--cheng%2FJMAS--VAE-ffd21e.svg?style=flat-square&logo=huggingface)](https://huggingface.co/ch-cheng/JMAS-VAE)

## Repository Layout

- `configs/jmas_vae_train.json`: JMAS-VAE training configuration with the SSL/WavLM bottleneck loss.
- `configs/vanilla_vae_train.json`: vanilla VAE training configuration.
- `configs/inference.json`: shared inference configuration for both released checkpoints.
- `configs/local_dataset.json`: local audio-directory dataset template.
- `train.py`: training entry point.
- `unwrap_model.py`: exports a Lightning checkpoint to an inference checkpoint and removes training-only SSL/WavLM weights by default.
- `vae_recon.py`: extracts latents and reconstructs audio.
- `stable_audio_tools/`: model, data, loss, and training modules required by the two configurations.

## Installation

Python 3.10 and a PyTorch/Torchaudio installation matching your CUDA version are recommended.

```bash
git clone https://github.com/changhao-cheng/JMAS-VAE.git
cd JMAS-VAE
pip install -e .
```

If you prefer to control the PyTorch build manually, install `torch` and `torchaudio` from the official PyTorch index first, then run `pip install -e .`.

## Training

The default training setup is documented with LibriLight-style local audio directories. Edit `configs/local_dataset.json` and point `path` to your local LibriLight audio directory:

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

Train JMAS-VAE:

```bash
python train.py \
  --model-config configs/jmas_vae_train.json \
  --dataset-config configs/local_dataset.json \
  --name jmas_vae \
  --batch-size 2 \
  --num-gpus 8 \
  --save-dir checkpoints
```

Train vanilla VAE:

```bash
python train.py \
  --model-config configs/vanilla_vae_train.json \
  --dataset-config configs/local_dataset.json \
  --name vanilla_vae \
  --batch-size 2 \
  --num-gpus 8 \
  --save-dir checkpoints
```

## Export Checkpoints

Training checkpoints are PyTorch Lightning checkpoints. Export them before inference. By default, `unwrap_model.py` removes SSL/WavLM training-only parameters, so the released JMAS-VAE checkpoint can be loaded with `configs/inference.json`.

```bash
python unwrap_model.py \
  --model-config configs/jmas_vae_train.json \
  --ckpt-path /path/to/epoch-step.ckpt \
  --output jmas_vae_600k.ckpt
```

For vanilla VAE:

```bash
python unwrap_model.py \
  --model-config configs/vanilla_vae_train.json \
  --ckpt-path /path/to/vanilla-lightning.ckpt \
  --output vanilla_vae.ckpt
```

## Latent Extraction

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

`--audios` accepts audio files, audio directories, or Kaldi-style `wav.scp` files. Directory inputs are scanned recursively for `wav/flac/mp3/ogg/m4a/aac` by default. Use `--audio_format wav,flac,mp3` to restrict the accepted formats.

## Reconstruction

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

`--output_format` can be set to formats writable by the local `soundfile` installation, such as `wav`, `flac`, or `mp3`. MP3 output requires a `libsndfile/soundfile` build with MP3 encoding support.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{cheng2026distillation,
  title={On the distillation loss functions of speech vae for unified reconstruction, understanding, and generation},
  author={Cheng, Changhao and Wang, Wei and Zhang, Wangyou and Jia, Dongya and Wu, Jian and Chen, Zhuo and Qian, Yanmin},
  journal={arXiv preprint arXiv:2604.12383},
  year={2026}
}
```

## Acknowledgements

This project is derived from Stable Audio Tools and keeps the pieces needed for the JMAS-VAE and vanilla VAE autoencoder experiments.

We also thank the [Semantic-VAE](https://github.com/ZhikangNiu/Semantic-VAE) project for the semantic alignment idea that inspired this work.

## License
This repository is licensed under the MIT License.
- Code in `stable_audio_tools/` originates from Stable Audio Tools, Copyright (c) 2023 Stability AI.
- All other newly added code (training scripts, inference scripts, configs, README) is Copyright (c) 2026 Changhao Cheng.

See the [LICENSE](LICENSE) file for the full license text.
