from typing import Any

import numpy as np
import pytest
import torch

from core.evaluator import (
    calculate_max_dropout,
    calculate_true_peak,
    convert_tensor_to_numpy,
    evaluate_audio_quality,
)


def test_convert_tensor_to_numpy_mono() -> None:
    tensor: torch.Tensor = torch.zeros((1, 1, 24000), dtype=torch.float32)
    arr: np.ndarray[Any, np.dtype[np.float32]] = convert_tensor_to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (24000, 1)


def test_convert_tensor_to_numpy_stereo() -> None:
    tensor: torch.Tensor = torch.zeros((1, 2, 24000), dtype=torch.float32)
    arr: np.ndarray[Any, np.dtype[np.float32]] = convert_tensor_to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (24000, 2)


def test_calculate_true_peak_silence() -> None:
    arr: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(
        (24000, 1), dtype=np.float32
    )
    tp: float = calculate_true_peak(arr)
    assert tp == -100.0


def test_calculate_true_peak_sine() -> None:
    sample_rate: int = 24000
    t: np.ndarray[Any, np.dtype[np.float32]] = np.linspace(
        0, 1, sample_rate, endpoint=False, dtype=np.float32
    )
    arr: np.ndarray[Any, np.dtype[np.float32]] = np.sin(2 * np.pi * 1000 * t)[
        :, np.newaxis
    ]
    tp: float = calculate_true_peak(arr)
    assert -0.5 <= tp <= 0.5


def test_calculate_max_dropout() -> None:
    sample_rate: int = 24000
    arr: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(
        (sample_rate * 3, 1), dtype=np.float32
    )
    dropout: float = calculate_max_dropout(arr, sample_rate)
    assert pytest.approx(dropout, rel=1e-2) == 3.0


def test_evaluate_audio_quality_pipeline() -> None:
    sample_rate: int = 24000
    tensor: torch.Tensor = torch.zeros(
        (1, 1, sample_rate * 2), dtype=torch.float32
    )
    report: str
    is_valid: bool
    report, is_valid = evaluate_audio_quality(
        waveform=tensor,
        sample_rate=sample_rate,
        target_lufs=-14.0,
        lufs_tolerance=2.0,
        max_true_peak_dbtp=-1.0,
        max_noise_floor_dbfs=-40.0,
        max_lra_lu=3.0,
        max_dropout_seconds=2.0,
    )
    assert isinstance(report, str)
    assert isinstance(is_valid, bool)
    assert "Audio Quality Evaluation Report" in report
