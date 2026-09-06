"""
Modul Evaluasi Standar Kualitas Audio.
Ringkasan Fungsi:
1. convert_tensor_to_numpy : Mengubah tensor PyTorch ComfyUI menjadi array NumPy float32.
2. calculate_lufs_and_lra  : Menghitung Integrated Loudness (LUFS) dan Loudness Range (LRA) via ITU-R BS.1770-4.
3. calculate_true_peak     : Menghitung True Peak (dBTP) menggunakan 4x oversampling polifase.
4. calculate_noise_floor   : Menghitung Noise Floor (dBFS) dari persentil ke-10 jendela RMS 50ms.
5. calculate_max_dropout   : Mendeteksi durasi hening terpanjang dalam detik di bawah ambang batas.
6. evaluate_audio_quality  : Fungsi orkestrator yang mengevaluasi seluruh parameter dan menyusun laporan.
"""

import math
from typing import Any

import numpy as np
import pyloudnorm as pyln
from scipy import signal
import torch


# 1. Konversi Tensor PyTorch ke Matriks NumPy
# ---
# Mengubah tensor audio [batch, channels, samples] dari ComfyUI menjadi format
# matriks NumPy 2D [samples, channels] bertipe float32 dalam rentang [-1.0, 1.0].
def convert_tensor_to_numpy(
    waveform: torch.Tensor,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    audio_tensor: torch.Tensor = waveform[0]
    if audio_tensor.ndim == 2:
        return audio_tensor.cpu().numpy().T.astype(np.float32)
    audio_np: np.ndarray[Any, np.dtype[np.float32]] = (
        audio_tensor.cpu().numpy().astype(np.float32)
    )
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=1)
    return audio_np


# 2. Perhitungan Integrated Loudness (LUFS) dan Loudness Range (LRA)
# ---
# Mengukur persepsi kenyaringan manusia dengan filter K-weighting (BS.1770-4)
# serta mengukur rentang variasi dinamika volume vokal menggunakan pyloudnorm.
def calculate_lufs_and_lra(
    audio_np: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> tuple[float, float]:
    meter: pyln.Meter = pyln.Meter(sample_rate)
    try:
        measured_lufs: float = float(meter.integrated_loudness(audio_np))
    except (ValueError, RuntimeError):
        measured_lufs = -float("inf")

    try:
        measured_lra: float = float(meter.loudness_range(audio_np))
    except (ValueError, RuntimeError):
        measured_lra = 0.0

    return measured_lufs, measured_lra


# 3. Perhitungan True Peak (dBTP) dengan 4x Oversampling
# ---
# Mendeteksi puncak sinyal antarsampel (inter-sample peak) dengan interpolasi
# 4x oversampling untuk mencegah distorsi kliping saat konversi lossy format.
def calculate_true_peak(
    audio_np: np.ndarray[Any, np.dtype[np.float32]],
) -> float:
    oversampled: np.ndarray[Any, np.dtype[np.float32]] = signal.resample_poly(
        audio_np, up=4, down=1, axis=0
    )
    peak_val: float = float(np.max(np.abs(oversampled)))
    if peak_val > 0.0:
        return 20.0 * math.log10(peak_val)
    return -100.0


# 4. Perhitungan Noise Floor (dBFS) pada Segmen Hening
# ---
# Membagi audio ke jendela 50 milidetik dan mengambil persentil ke-10 RMS
# terendah untuk mengukur dasar desis noise tanpa gangguan sinyal vokal utama.
def calculate_noise_floor(
    audio_np: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> float:
    window_len: int = max(1, int(sample_rate * 0.05))
    mono_audio: np.ndarray[Any, np.dtype[np.float32]] = np.mean(audio_np, axis=1)
    num_windows: int = len(mono_audio) // window_len

    if num_windows == 0:
        return -100.0

    truncated_len: int = num_windows * window_len
    windows: np.ndarray[Any, np.dtype[np.float32]] = mono_audio[
        :truncated_len
    ].reshape(num_windows, window_len)
    rms_per_window: np.ndarray[Any, np.dtype[np.float32]] = np.sqrt(
        np.mean(windows**2, axis=1)
    )
    quietest_rms: float = float(np.percentile(rms_per_window, 10))

    if quietest_rms > 0.0:
        return 20.0 * math.log10(quietest_rms)
    return -100.0


# 5. Perhitungan Durasi Dropout Terpanjang (Detik)
# ---
# Mengukur durasi keheningan terpanjang berturut-turut di bawah ambang -60 dBFS
# untuk mendeteksi adanya audio yang terpotong atau mengalami desinkronisasi.
def calculate_max_dropout(
    audio_np: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    silence_threshold_dbfs: float = -60.0,
) -> float:
    mono_audio: np.ndarray[Any, np.dtype[np.float32]] = np.mean(audio_np, axis=1)
    threshold_amp: float = 10.0 ** (silence_threshold_dbfs / 20.0)
    is_silent: np.ndarray[Any, np.dtype[np.bool_]] = (
        np.abs(mono_audio) < threshold_amp
    )

    max_silent_samples: int = 0
    current_silent_samples: int = 0

    for silent in is_silent:
        if silent:
            current_silent_samples += 1
            max_silent_samples = max(max_silent_samples, current_silent_samples)
        else:
            current_silent_samples = 0

    return max_silent_samples / sample_rate




# 6. Fungsi Utama: Orkestrator Evaluasi Standar Audio
# ------
# Menjalankan seluruh fungsi perhitungan, membandingkan hasil dengan target
# spesifikasi dokumen Fase 2, dan menyusun laporan evaluasi dalam format teks.
def evaluate_audio_quality(
    waveform: torch.Tensor,
    sample_rate: int,
    target_lufs: float,
    lufs_tolerance: float,
    max_true_peak_dbtp: float,
    max_noise_floor_dbfs: float,
    max_lra_lu: float,
    max_dropout_seconds: float,
) -> tuple[str, bool]:
    audio_np: np.ndarray[Any, np.dtype[np.float32]] = convert_tensor_to_numpy(
        waveform
    )

    measured_lufs, measured_lra = calculate_lufs_and_lra(audio_np, sample_rate)
    measured_true_peak: float = calculate_true_peak(audio_np)
    measured_noise_floor: float = calculate_noise_floor(audio_np, sample_rate)
    measured_dropout: float = calculate_max_dropout(audio_np, sample_rate)

    lufs_min: float = target_lufs - lufs_tolerance
    lufs_max: float = target_lufs + lufs_tolerance
    lufs_pass: bool = lufs_min <= measured_lufs <= lufs_max
    tp_pass: bool = measured_true_peak <= max_true_peak_dbtp
    noise_pass: bool = measured_noise_floor <= max_noise_floor_dbfs
    lra_pass: bool = measured_lra <= max_lra_lu
    dropout_pass: bool = measured_dropout <= max_dropout_seconds

    is_valid: bool = (
        lufs_pass and tp_pass and noise_pass and lra_pass and dropout_pass
    )

    lufs_status: str = "PASSED" if lufs_pass else "FAILED"
    tp_status: str = "PASSED" if tp_pass else "FAILED"
    noise_status: str = "PASSED" if noise_pass else "FAILED"
    lra_status: str = "PASSED" if lra_pass else "FAILED"
    dropout_status: str = "PASSED" if dropout_pass else "FAILED"
    overall_status: str = "PASSED" if is_valid else "FAILED"

    report: str = (
        f"=== Audio Quality Evaluation Report ===\n"
        f"Overall Status       : {overall_status}\n"
        f"Sample Rate          : {sample_rate} Hz\n"
        f"Channels             : {audio_np.shape[1]}\n"
        f"----------------------------------------\n"
        f"1. Integrated Loudness: {measured_lufs:.2f} LUFS [{lufs_status}]\n"
        f"   Target            : {target_lufs:.1f} +/- {lufs_tolerance:.1f} LUFS (Range: {lufs_min:.1f} to {lufs_max:.1f})\n"
        f"2. True Peak         : {measured_true_peak:.2f} dBTP [{tp_status}]\n"
        f"   Target Max        : {max_true_peak_dbtp:.1f} dBTP\n"
        f"3. Noise Floor       : {measured_noise_floor:.2f} dBFS [{noise_status}]\n"
        f"   Target Max        : {max_noise_floor_dbfs:.1f} dBFS\n"
        f"4. Loudness Range    : {measured_lra:.2f} LU [{lra_status}]\n"
        f"   Target Max        : {max_lra_lu:.1f} LU\n"
        f"5. Max Dropout       : {measured_dropout:.2f} s [{dropout_status}]\n"
        f"   Target Max        : {max_dropout_seconds:.1f} s\n"
        f"========================================\n"
    )

    return report, is_valid
