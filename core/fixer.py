"""
Modul Perbaikan Standar Kualitas Audio.
Ringkasan Fungsi:
1. apply_gain              : Menggeser level audio secara linear sebesar nilai dB.
2. lookahead_limiter       : Menahan true peak pada 4x oversampling dengan lookahead.
3. determine_feasibility   : Menilai kelayakan perbaikan dari nilai PLR.
4. fix_audio_quality       : Fungsi orkestrator yang memperbaiki lalu mengukur ulang.

Loudness dan true peak bersifat dapat diperbaiki. Noise floor, loudness range,
dan dropout tidak dapat diperbaiki oleh gain maupun limiter, sehingga hanya
dilaporkan beserta saran peninjauan sumber.
"""

import math
from typing import Any

import numpy as np
import torch
from scipy import ndimage, signal

from .audio_io import convert_numpy_to_tensor, convert_tensor_to_numpy
from .evaluator import (
    calculate_lufs_and_lra,
    calculate_max_dropout,
    calculate_noise_floor,
    calculate_true_peak,
)

FloatArray = np.ndarray[Any, np.dtype[np.float32]]

MAX_ITERATIONS: int = 3
ITERATION_TOLERANCE_LU: float = 0.5
PEAK_CONVERGENCE_TOLERANCE_DB: float = 0.1

FEASIBILITY_GAIN_ONLY: str = "gain only"
FEASIBILITY_LIMITER_LIGHT: str = "light limiting"
FEASIBILITY_LIMITER_HEAVY: str = "heavy limiting"
FEASIBILITY_NOT_ACHIEVABLE: str = "not achievable"


# 1. Penggeseran Level Linear
# ---
# Mengalikan amplitudo dengan faktor desibel. Operasi ini tidak mengubah
# bentuk gelombang, hanya levelnya, sehingga PLR tetap sama.
def apply_gain(audio_np: FloatArray, gain_db: float) -> FloatArray:
    if gain_db == 0.0:
        return audio_np
    factor: float = 10.0 ** (gain_db / 20.0)
    scaled: FloatArray = (audio_np * factor).astype(np.float32)
    return scaled


# 2. Limiter Lookahead pada 4x Oversampling
# ---
# Menghitung kurva penguatan dari selubung puncak, mengambil nilai minimum
# pada jendela lookahead agar penguatan turun sebelum puncak tiba, lalu
# menghaluskannya agar pelepasan tidak mendadak. Nilai minimum diterapkan
# kembali sebagai batas atas sehingga tidak ada sampel yang melewati ceiling.
def lookahead_limiter(
    audio_np: FloatArray,
    ceiling_dbtp: float,
    sample_rate: int,
    oversample: int = 4,
    attack_ms: float = 2.0,
    release_ms: float = 50.0,
) -> tuple[FloatArray, float]:
    ceiling_amp: float = 10.0 ** (ceiling_dbtp / 20.0)
    oversampled: FloatArray = signal.resample_poly(
        audio_np, up=oversample, down=1, axis=0
    ).astype(np.float32)

    peak_envelope: FloatArray = np.max(np.abs(oversampled), axis=1)
    required_gain: FloatArray = np.ones_like(peak_envelope, dtype=np.float32)
    active: np.ndarray[Any, np.dtype[np.bool_]] = peak_envelope > ceiling_amp
    required_gain[active] = (ceiling_amp / peak_envelope[active]).astype(np.float32)

    if not bool(np.any(active)):
        return audio_np, 0.0

    oversampled_rate: int = sample_rate * oversample
    attack_samples: int = max(1, int(oversampled_rate * attack_ms / 1000.0))
    release_samples: int = max(1, int(oversampled_rate * release_ms / 1000.0))

    lookahead_gain: FloatArray = ndimage.minimum_filter1d(
        required_gain, size=attack_samples, mode="nearest"
    ).astype(np.float32)
    smoothed_gain: FloatArray = ndimage.uniform_filter1d(
        lookahead_gain, size=release_samples, mode="nearest"
    ).astype(np.float32)
    applied_gain: FloatArray = np.minimum(smoothed_gain, lookahead_gain)

    limited: FloatArray = (
        oversampled * applied_gain[:, np.newaxis]
    ).astype(np.float32)
    restored: FloatArray = signal.resample_poly(
        limited, up=1, down=oversample, axis=0
    ).astype(np.float32)

    min_gain: float = float(np.min(applied_gain))
    reduction_db: float = 0.0
    if min_gain > 0.0:
        reduction_db = -20.0 * math.log10(min_gain)

    return restored, reduction_db


# 3. Penilaian Kelayakan Perbaikan dari PLR
# ---
# PLR adalah selisih true peak terhadap integrated loudness. Nilai ini tidak
# berubah oleh gain linear, sehingga menentukan apakah gain saja cukup atau
# limiter diperlukan. Ambang mengikuti hasil pengukuran pada berkas nyata.
def determine_feasibility(plr_db: float, plr_gain_only_threshold: float) -> str:
    excess: float = plr_db - plr_gain_only_threshold
    if excess <= 1.5:
        return FEASIBILITY_LIMITER_LIGHT
    if excess <= 6.0:
        return FEASIBILITY_LIMITER_HEAVY
    return FEASIBILITY_NOT_ACHIEVABLE


# 4. Fungsi Pembantu Penyusunan Laporan
# ---
def _status(passed: bool) -> str:
    return "PASSED" if passed else "FAILED"


def _criteria_block(
    lufs: float,
    true_peak: float,
    noise_floor: float,
    lra: float,
    dropout: float,
    target_lufs: float,
    lufs_tolerance: float,
    max_true_peak_dbtp: float,
    max_noise_floor_dbfs: float,
    max_lra_lu: float,
    max_dropout_seconds: float,
) -> str:
    lufs_min: float = target_lufs - lufs_tolerance
    lufs_max: float = target_lufs + lufs_tolerance
    return (
        f"  Integrated Loudness: {lufs:.2f} LUFS "
        f"[{_status(lufs_min <= lufs <= lufs_max)}]\n"
        f"  True Peak          : {true_peak:.2f} dBTP "
        f"[{_status(true_peak <= max_true_peak_dbtp)}]\n"
        f"  Noise Floor        : {noise_floor:.2f} dBFS "
        f"[{_status(noise_floor <= max_noise_floor_dbfs)}]\n"
        f"  Loudness Range     : {lra:.2f} LU "
        f"[{_status(lra <= max_lra_lu)}]\n"
        f"  Max Dropout        : {dropout:.2f} s "
        f"[{_status(dropout <= max_dropout_seconds)}]\n"
    )


def _build_notes(
    lra: float,
    noise_floor: float,
    dropout: float,
    max_lra_lu: float,
    max_noise_floor_dbfs: float,
    max_dropout_seconds: float,
) -> list[str]:
    notes: list[str] = []
    if lra > max_lra_lu:
        notes.append(
            f"Loudness range {lra:.2f} LU exceeds {max_lra_lu:.2f} LU. "
            "Gain and limiting cannot reduce dynamic range; review the "
            "source audio or apply compression upstream."
        )
    if noise_floor > max_noise_floor_dbfs:
        notes.append(
            f"Noise floor {noise_floor:.2f} dBFS exceeds "
            f"{max_noise_floor_dbfs:.2f} dBFS. This originates from the "
            "recording source and is not corrected by gain or limiting."
        )
    if dropout > max_dropout_seconds:
        notes.append(
            f"Max dropout {dropout:.2f} s exceeds {max_dropout_seconds:.2f} s. "
            "Missing audio cannot be reconstructed; regenerate the source."
        )
    if not notes:
        notes.append("No further issues detected beyond the fixed criteria.")
    return notes


# 5. Fungsi Utama: Orkestrator Perbaikan Standar Audio
# ------
# Menerapkan gain menuju target loudness, menahan true peak dengan limiter,
# lalu mengukur ulang. Bila limiting menurunkan loudness keluar dari
# toleransi, koreksi gain diulang sampai batas iterasi.
def fix_audio_quality(
    waveform: torch.Tensor,
    sample_rate: int,
    target_lufs: float,
    lufs_tolerance: float,
    max_true_peak_dbtp: float,
    target_true_peak_dbtp: float,
    plr_gain_only_threshold: float,
    max_noise_floor_dbfs: float,
    max_lra_lu: float,
    max_dropout_seconds: float,
    fix_loudness: bool,
    fix_true_peak: bool,
) -> tuple[torch.Tensor, str, bool]:
    audio_np: FloatArray = convert_tensor_to_numpy(waveform)

    before_lufs: float
    before_lra: float
    before_lufs, before_lra = calculate_lufs_and_lra(audio_np, sample_rate)
    before_true_peak: float = calculate_true_peak(audio_np)
    before_noise: float = calculate_noise_floor(audio_np, sample_rate)
    before_dropout: float = calculate_max_dropout(audio_np, sample_rate)

    plr_db: float = before_true_peak - before_lufs
    feasibility: str = determine_feasibility(plr_db, plr_gain_only_threshold)

    effective_ceiling: float = min(target_true_peak_dbtp, max_true_peak_dbtp)
    ceiling_note: str = ""
    if target_true_peak_dbtp > max_true_peak_dbtp:
        ceiling_note = (
            f"Target true peak {target_true_peak_dbtp:.2f} dBTP is above the "
            f"pass limit {max_true_peak_dbtp:.2f} dBTP; the stricter value "
            f"{effective_ceiling:.2f} dBTP was used."
        )

    current: FloatArray = audio_np
    total_gain_db: float = 0.0
    total_limiter_db: float = 0.0
    iterations_used: int = 0

    for iteration in range(MAX_ITERATIONS):
        changed: bool = False

        if fix_loudness:
            current_lufs: float
            current_lufs, _ = calculate_lufs_and_lra(current, sample_rate)
            if math.isfinite(current_lufs):
                needed_db: float = target_lufs - current_lufs
                if iteration == 0 or abs(needed_db) > ITERATION_TOLERANCE_LU:
                    current = apply_gain(current, needed_db)
                    total_gain_db += needed_db
                    changed = True

        if fix_true_peak:
            current_true_peak: float = calculate_true_peak(current)
            if current_true_peak > effective_ceiling + PEAK_CONVERGENCE_TOLERANCE_DB:
                current, reduction_db = lookahead_limiter(
                    current, effective_ceiling, sample_rate
                )
                total_limiter_db += reduction_db
                changed = True

        iterations_used = iteration + 1
        if not changed:
            break

    after_lufs: float
    after_lra: float
    after_lufs, after_lra = calculate_lufs_and_lra(current, sample_rate)
    after_true_peak: float = calculate_true_peak(current)
    after_noise: float = calculate_noise_floor(current, sample_rate)
    after_dropout: float = calculate_max_dropout(current, sample_rate)

    lufs_min: float = target_lufs - lufs_tolerance
    lufs_max: float = target_lufs + lufs_tolerance
    loudness_ok: bool = lufs_min <= after_lufs <= lufs_max
    peak_ok: bool = after_true_peak <= max_true_peak_dbtp
    noise_ok: bool = after_noise <= max_noise_floor_dbfs
    lra_ok: bool = after_lra <= max_lra_lu
    dropout_ok: bool = after_dropout <= max_dropout_seconds

    loudness_settled: bool = loudness_ok or not fix_loudness
    peak_settled: bool = peak_ok or not fix_true_peak
    is_fixed: bool = loudness_settled and peak_settled

    applied_any: bool = total_gain_db != 0.0 or total_limiter_db > 0.0
    if not applied_any:
        final_status = "UNCHANGED"
    elif is_fixed and noise_ok and lra_ok and dropout_ok:
        final_status = "FIXED"
    elif is_fixed:
        final_status = "FIXED WITH NOTES"
    else:
        final_status = "PARTIAL"

    limiter_line: str = "not applied"
    if fix_true_peak:
        limiter_line = (
            f"{total_limiter_db:.2f} dB max reduction"
            if total_limiter_db > 0.0
            else "no reduction needed"
        )

    gain_line: str = "not applied"
    if fix_loudness:
        gain_line = f"{total_gain_db:+.2f} dB"

    notes: list[str] = _build_notes(
        lra=after_lra,
        noise_floor=after_noise,
        dropout=after_dropout,
        max_lra_lu=max_lra_lu,
        max_noise_floor_dbfs=max_noise_floor_dbfs,
        max_dropout_seconds=max_dropout_seconds,
    )
    if ceiling_note:
        notes.insert(0, ceiling_note)

    notes_block: str = "\n".join(f"  - {note}" for note in notes)

    report: str = (
        f"=== Audio Standards Fixer Report ===\n"
        f"Sample Rate          : {sample_rate} Hz\n"
        f"Channels             : {audio_np.shape[1]}\n"
        f"----------------------------------------\n"
        f"BEFORE\n"
        + _criteria_block(
            lufs=before_lufs,
            true_peak=before_true_peak,
            noise_floor=before_noise,
            lra=before_lra,
            dropout=before_dropout,
            target_lufs=target_lufs,
            lufs_tolerance=lufs_tolerance,
            max_true_peak_dbtp=max_true_peak_dbtp,
            max_noise_floor_dbfs=max_noise_floor_dbfs,
            max_lra_lu=max_lra_lu,
            max_dropout_seconds=max_dropout_seconds,
        )
        + "----------------------------------------\n"
        "AFTER\n"
        + _criteria_block(
            lufs=after_lufs,
            true_peak=after_true_peak,
            noise_floor=after_noise,
            lra=after_lra,
            dropout=after_dropout,
            target_lufs=target_lufs,
            lufs_tolerance=lufs_tolerance,
            max_true_peak_dbtp=max_true_peak_dbtp,
            max_noise_floor_dbfs=max_noise_floor_dbfs,
            max_lra_lu=max_lra_lu,
            max_dropout_seconds=max_dropout_seconds,
        )
        + f"----------------------------------------\n"
        f"Actions Applied\n"
        f"  Loudness Gain      : {gain_line}\n"
        f"  True Peak Limiter  : {limiter_line}\n"
        f"  Limiter Ceiling    : {effective_ceiling:.2f} dBTP\n"
        f"  PLR                : {plr_db:.2f} dB\n"
        f"  Feasibility        : {feasibility}\n"
        f"  Iterations Used    : {iterations_used} of {MAX_ITERATIONS}\n"
        f"  Final Status       : {final_status}\n"
        f"----------------------------------------\n"
        f"Notes\n"
        f"{notes_block}\n"
        f"========================================\n"
    )

    output_tensor: torch.Tensor = convert_numpy_to_tensor(current)
    return output_tensor, report, is_fixed
