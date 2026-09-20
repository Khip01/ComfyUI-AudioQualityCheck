from typing import Any

import numpy as np
import pytest
import torch

from core.audio_io import convert_numpy_to_tensor, convert_tensor_to_numpy
from core.fixer import (
    FEASIBILITY_GAIN_ONLY,
    FEASIBILITY_LIMITER_HEAVY,
    FEASIBILITY_LIMITER_LIGHT,
    FEASIBILITY_NOT_ACHIEVABLE,
    apply_gain,
    determine_feasibility,
    fix_audio_quality,
    lookahead_limiter,
)

FloatArray = np.ndarray[Any, np.dtype[np.float32]]
SAMPLE_RATE: int = 24000


def _sine(amplitude: float, seconds: float = 4.0, freq: float = 1000.0) -> FloatArray:
    samples: int = int(SAMPLE_RATE * seconds)
    t: FloatArray = np.linspace(
        0.0, seconds, samples, endpoint=False, dtype=np.float32
    )
    wave: FloatArray = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(
        np.float32
    )
    return wave[:, np.newaxis]


def _fix_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "target_lufs": -14.0,
        "lufs_tolerance": 2.0,
        "max_true_peak_dbtp": -1.0,
        "target_true_peak_dbtp": -2.0,
        "plr_gain_only_threshold": 13.0,
        "max_noise_floor_dbfs": -40.0,
        "max_lra_lu": 3.0,
        "max_dropout_seconds": 2.0,
        "fix_loudness": True,
        "fix_true_peak": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_convert_numpy_to_tensor_round_trip() -> None:
    original: FloatArray = _sine(0.5)
    tensor: torch.Tensor = convert_numpy_to_tensor(original)
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == original.shape[1]
    restored: FloatArray = convert_tensor_to_numpy(tensor)
    assert restored.shape == original.shape
    assert np.allclose(restored, original, atol=1e-6)


def test_apply_gain_zero_returns_same_array() -> None:
    arr: FloatArray = _sine(0.5)
    result: FloatArray = apply_gain(arr, 0.0)
    assert result is arr


def test_apply_gain_scales_amplitude() -> None:
    arr: FloatArray = _sine(0.25)
    result: FloatArray = apply_gain(arr, 6.0)
    expected_factor: float = 10.0 ** (6.0 / 20.0)
    assert float(np.max(np.abs(result))) == pytest.approx(
        float(np.max(np.abs(arr))) * expected_factor, rel=1e-4
    )


def test_apply_gain_negative_reduces_amplitude() -> None:
    arr: FloatArray = _sine(0.5)
    result: FloatArray = apply_gain(arr, -6.0)
    assert float(np.max(np.abs(result))) < float(np.max(np.abs(arr)))


def test_lookahead_limiter_reduces_peak_below_ceiling() -> None:
    arr: FloatArray = _sine(0.9)
    ceiling_dbtp: float = -2.0
    limited: FloatArray
    reduction_db: float
    limited, reduction_db = lookahead_limiter(arr, ceiling_dbtp, SAMPLE_RATE)

    ceiling_amp: float = 10.0 ** (ceiling_dbtp / 20.0)
    assert float(np.max(np.abs(limited))) <= ceiling_amp * 1.02
    assert reduction_db > 0.0


def test_lookahead_limiter_skips_signal_under_ceiling() -> None:
    arr: FloatArray = _sine(0.1)
    limited: FloatArray
    reduction_db: float
    limited, reduction_db = lookahead_limiter(arr, -2.0, SAMPLE_RATE)
    assert limited is arr
    assert reduction_db == 0.0


def test_determine_feasibility_thresholds() -> None:
    assert determine_feasibility(12.0, 13.0) == FEASIBILITY_LIMITER_LIGHT
    assert determine_feasibility(14.0, 13.0) == FEASIBILITY_LIMITER_LIGHT
    assert determine_feasibility(16.0, 13.0) == FEASIBILITY_LIMITER_HEAVY
    assert determine_feasibility(19.0, 13.0) == FEASIBILITY_LIMITER_HEAVY
    assert determine_feasibility(20.0, 13.0) == FEASIBILITY_NOT_ACHIEVABLE


def test_determine_feasibility_gain_only_label_is_exposed() -> None:
    assert FEASIBILITY_GAIN_ONLY == "gain only"


def test_fix_audio_quality_reaches_loudness_and_peak_targets() -> None:
    tensor: torch.Tensor = convert_numpy_to_tensor(_sine(0.9))
    fixed: torch.Tensor
    report: str
    is_fixed: bool
    fixed, report, is_fixed = fix_audio_quality(
        waveform=tensor, **_fix_kwargs()
    )

    from core.evaluator import calculate_lufs_and_lra, calculate_true_peak

    fixed_np: FloatArray = convert_tensor_to_numpy(fixed)
    lufs: float
    lufs, _ = calculate_lufs_and_lra(fixed_np, SAMPLE_RATE)
    true_peak: float = calculate_true_peak(fixed_np)

    assert -16.0 <= lufs <= -12.0
    assert true_peak <= -1.0
    assert is_fixed is True
    assert "BEFORE" in report
    assert "AFTER" in report
    assert "Final Status" in report


def test_fix_audio_quality_skips_when_both_flags_disabled() -> None:
    original: FloatArray = _sine(0.9)
    tensor: torch.Tensor = convert_numpy_to_tensor(original)
    fixed: torch.Tensor
    report: str
    is_fixed: bool
    fixed, report, is_fixed = fix_audio_quality(
        waveform=tensor,
        **_fix_kwargs(fix_loudness=False, fix_true_peak=False),
    )

    fixed_np: FloatArray = convert_tensor_to_numpy(fixed)
    assert np.allclose(fixed_np, original, atol=1e-6)
    assert "UNCHANGED" in report
    assert is_fixed is True


def test_fix_audio_quality_respects_disabled_loudness() -> None:
    original: FloatArray = _sine(0.3)
    tensor: torch.Tensor = convert_numpy_to_tensor(original)
    fixed: torch.Tensor
    report: str
    _: bool
    fixed, report, _ = fix_audio_quality(
        waveform=tensor,
        **_fix_kwargs(fix_loudness=False, fix_true_peak=True),
    )

    assert "not applied" in report
    fixed_np: FloatArray = convert_tensor_to_numpy(fixed)
    assert np.allclose(fixed_np, original, atol=1e-6)


def test_fix_audio_quality_reports_target_peak_conflict() -> None:
    tensor: torch.Tensor = convert_numpy_to_tensor(_sine(0.9))
    _: torch.Tensor
    report: str
    _fixed: bool
    _, report, _fixed = fix_audio_quality(
        waveform=tensor,
        **_fix_kwargs(target_true_peak_dbtp=0.0),
    )
    assert "stricter value" in report


def test_fix_audio_quality_silence_stays_silent() -> None:
    silence: FloatArray = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
    tensor: torch.Tensor = convert_numpy_to_tensor(silence)
    fixed: torch.Tensor
    _report: str
    _is_fixed: bool
    fixed, _report, _is_fixed = fix_audio_quality(
        waveform=tensor, **_fix_kwargs()
    )
    fixed_np: FloatArray = convert_tensor_to_numpy(fixed)
    assert np.allclose(fixed_np, 0.0, atol=1e-6)
