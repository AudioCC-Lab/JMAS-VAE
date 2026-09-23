from setuptools import find_packages, setup


setup(
    name="jmas-vae",
    version="0.1.0",
    description="JMAS-VAE training and latent reconstruction tools for 16 kHz speech audio.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "alias-free-torch==0.0.6",
        "auraloss==0.4.0",
        "descript-audio-codec==1.0.0",
        "einops",
        "einops-exts",
        "ema-pytorch==0.2.3",
        "librosa",
        "numpy",
        "pandas",
        "prefigure==0.0.9",
        "pytorch-lightning",
        "safetensors",
        "soundfile",
        "torch",
        "torchaudio",
        "torchmetrics==0.11.4",
        "tqdm",
        "wandb==0.15.4",
        "webdataset==0.2.100",
        "nnAudio",
        "timm",
        "transformers",
    ],
)
