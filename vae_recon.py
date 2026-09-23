import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from stable_audio_tools.models import create_model_from_config


TARGET_FS = 16000
DEFAULT_AUDIO_FORMATS = ("wav", "flac", "mp3", "ogg", "m4a", "aac")


def parse_audio_formats(audio_format):
    if audio_format.lower() in {"all", "auto"}:
        return DEFAULT_AUDIO_FORMATS
    return tuple(fmt.strip().lower().lstrip(".") for fmt in audio_format.split(",") if fmt.strip())


def gather_audios(audios, audio_format="all"):
    """
    Read audio files or directories containing audios.
    If a directory is provided, it reads all audio files in the directory.
    If a scp file is provided, it reads the audio paths from the file.
    """
    if isinstance(audios, str):
        audios = [audios]

    audio_formats = parse_audio_formats(audio_format)
    audio_suffixes = {f".{fmt}" for fmt in audio_formats}
    audio_paths = []
    for audio in audios:
        path = Path(audio)
        if path.is_dir():
            for suffix in audio_suffixes:
                audio_paths.extend(path.rglob(f"*{suffix}"))
        elif path.is_file():
            if path.suffix.lower() in audio_suffixes:
                audio_paths.append(path)
            elif path.suffix == ".scp":
                with path.open("r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        audio_path = parts[-1]
                        audio_path = Path(audio_path)
                        if audio_path.suffix.lower() not in audio_suffixes:
                            raise ValueError(f"Unsupported audio in scp: {audio_path}")
                        audio_paths.append(audio_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Supported: {', '.join(audio_formats)}")
        else:
            raise ValueError(f"Invalid audio path: {audio}")
    names = tuple(p.name for p in audio_paths)
    assert len(names) == len(set(names)), "Audio files must have unique names"
    return [str(p) for p in audio_paths]


def load_vae_model(model_path, model_config, device="cpu"):
    with open(model_config) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.to(device=device)
    model.eval()
    return model

# -------------------------- 拆分：纯编码函数（仅输出隐向量）--------------------------
@torch.no_grad()
def vae_encode(model, audio, fs, device="cpu"):
    """仅编码音频，返回CPU格式的隐向量 + 原始音频长度（用于重建裁剪）"""
    original_length = audio.shape[0]
    # 重采样
    audio = librosa.resample(audio, orig_sr=fs, target_sr=TARGET_FS)
    audio = torch.as_tensor(audio, dtype=torch.float32).to(device=device)
    audio = model.preprocess_audio_for_encoder(audio, TARGET_FS)
    # 编码得到隐向量
    latents = model.encode_audio(
        audio, inference_only=True, chunked=False, overlap=32, chunk_size=128, return_info=False
    )
    # 立即移到CPU保存，释放GPU显存
    latents = latents.cpu()
    return latents, original_length

# -------------------------- 拆分：纯解码函数（仅从隐向量重建）--------------------------
@torch.no_grad()
def vae_decode(model, latents, original_length, device="cpu"):
    """仅从隐向量解码，返回重建后的音频"""
    latent_noise = 0  # 可调整隐空间噪声
    latents = latents.to(device)
    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise
    # 解码
    decoded = model.decode_audio(latents, chunked=False, overlap=32, chunk_size=128)
    reconstructed_audio = decoded.squeeze().cpu().numpy()[:original_length]
    assert reconstructed_audio.shape[0] == original_length, (reconstructed_audio.shape, original_length)
    return reconstructed_audio

# -------------------------- 步骤1：编码并保存隐向量到本地 --------------------------
def process_encode(inf_audio, model, device, latent_dir):
    """编码单条音频，将隐向量+原始长度保存为.pt文件"""
    # 读取音频
    inf, fs = sf.read(inf_audio, dtype="float32")
    # 编码
    latents, original_length = vae_encode(model, inf, fs, device)
    # 生成保存路径（音频名对应隐向量名）
    audio_name = Path(inf_audio).name
    latent_save_path = latent_dir / f"{audio_name}.pt"
    # 保存（包含隐向量+原始长度，缺一不可）
    torch.save({
        "latents": latents,
        "original_length": original_length
    }, latent_save_path)

# -------------------------- 步骤2：加载隐向量并重建音频 --------------------------
def process_decode(latent_file_path, model, device, output_dir, output_format="wav"):
    """加载本地隐向量，解码重建音频"""
    # 加载隐向量和原始音频长度
    latent_data = torch.load(latent_file_path, map_location="cpu")
    latents = latent_data["latents"]
    original_length = latent_data["original_length"]
    # 解码重建
    recon_audio = vae_decode(model, latents, original_length, device)
    # 保存重建音频
    audio_stem = Path(latent_file_path).stem
    output_format = output_format.lower().lstrip(".")
    output_path = output_dir / f"{audio_stem}.{output_format}"
    try:
        sf.write(str(output_path), recon_audio, TARGET_FS)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write {output_format} audio to {output_path}. "
            "If this is mp3, make sure your libsndfile/soundfile build supports MP3 encoding."
        ) from exc

################################################################
# 主函数：根据mode自动切换 编码/解码 模式
################################################################
def main(args):
    latent_dir = Path(args.latent_dir)
    output_dir = Path(args.output_dir)

    # ==================== 模式1：编码音频 → 保存隐向量 ====================
    if args.mode == "encode":
        latent_dir.mkdir(parents=True, exist_ok=True)
        # 读取音频列表
        audio_paths = gather_audios(args.audios, audio_format=args.audio_format)
        total_size = len(audio_paths)
        # 分块处理（兼容原有的多进程/多节点拆分）
        assert 1 <= args.job <= args.nsplits <= total_size
        interval = total_size // args.nsplits
        start_idx = (args.job - 1) * interval
        end_idx = total_size if args.job == args.nsplits else start_idx + interval
        audio_paths = audio_paths[start_idx:end_idx]

        print(f"[encode] job {args.job}/{args.nsplits} | processing {len(audio_paths)}/{total_size} audio file(s)", flush=True)
        # 加载模型并编码
        model = load_vae_model(args.model_path, args.model_config, device=args.device)
        for audio_path in tqdm(audio_paths, desc="Encoding audio"):
            process_encode(audio_path, model, args.device, latent_dir)
        print(f"Latents saved to: {latent_dir}")

    # ==================== 模式2：加载隐向量 → 重建音频 ====================
    elif args.mode == "decode":
        output_dir.mkdir(parents=True, exist_ok=True)
        # 读取所有保存的隐向量文件
        latent_files = list(latent_dir.glob("*.pt"))
        if not latent_files:
            raise ValueError(f"No .pt latent files found in {latent_dir}")
        total_size = len(latent_files)
        # 分块处理
        assert 1 <= args.job <= args.nsplits <= total_size
        interval = total_size // args.nsplits
        start_idx = (args.job - 1) * interval
        end_idx = total_size if args.job == args.nsplits else start_idx + interval
        latent_files = latent_files[start_idx:end_idx]

        print(f"[decode] job {args.job}/{args.nsplits} | processing {len(latent_files)}/{total_size} latent file(s)", flush=True)
        # 加载模型并重建
        model = load_vae_model(args.model_path, args.model_config, device=args.device)
        for latent_path in tqdm(latent_files, desc="Reconstructing audio"):
            process_decode(latent_path, model, args.device, output_dir, output_format=args.output_format)
        print(f"Reconstructed audio saved to: {output_dir}")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}. Choose 'encode' or 'decode'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE audio encoding / reconstruction tool (two-step, GPU-memory friendly)")
    # 核心模式参数
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["encode", "decode"],
        help="Mode: encode = encode audio to latents | decode = reconstruct audio from latents",
    )
    # 音频输入（仅encode模式需要）
    parser.add_argument(
        "--audios",
        type=str,
        nargs="+",
        help="Required for encode: audio file/directory/scp paths",
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default="all",
        help="Input audio formats, comma-separated; default 'all' = wav,flac,mp3,ogg,m4a,aac",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        help="Decode output audio format, e.g. wav/flac/mp3. Default wav",
    )
    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the VAE model checkpoint",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the VAE model config JSON",
    )
    # 路径参数
    parser.add_argument(
        "--latent_dir",
        type=str,
        required=True,
        help="encode: directory to save latents | decode: directory to load latents from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Required for decode: directory for reconstructed audio",
    )
    # 设备/分块参数
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cuda / cpu",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of job splits",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Current split index (1-based)",
    )
    args = parser.parse_args()

    main(args)
