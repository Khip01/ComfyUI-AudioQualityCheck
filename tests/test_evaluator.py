import math
import numpy as np
import pytest
import torch

from core.evaluator import (
    calculate_lufs_and_lra,
    calculate_max_dropout,
    calculate_noise_floor,
    calculate_true_peak,
    convert_tensor_to_numpy,
    evaluate_audio_quality,
)

def test_convert_tensor_to_numpy_mono():
    tensor = torch.zeros((1, 1, 24000), dtype=torch.float32)
    arr = convert_tensor_to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (24000, 1)

def test_convert_tensor_to_numpy_stereo():
    tensor = torch.zeros((1, 2, 24000), dtype=torch.float32)
    arr = convert_tensor_to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (24000, 2)

def test_calculate_true_peak_silence():
    arr = np.zeros((24000, 1), dtype=np.float32)
    tp = calculate_true_peak(arr)
    assert tp == -100.0

def test_calculate_true_peak_sine():
    sample_rate = 24000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    arr = np.sin(2 * np.pi * 1000 * t)[:, np.newaxis]
    tp = calculate_true_peak(arr)
    assert -0.5 <= tp <= 0.5

def test_calculate_max_dropout():
    sample_rate = 24000
    arr = np.zeros((sample_rate * 3, 1), dtype=np.float32)
    dropout = calculate_max_dropout(arr, sample_rate)
    assert pytest.approx(dropout, rel=1e-2) == 3.0

def test_evaluate_audio_quality_pipeline():
    sample_rate = 24000
    tensor = torch.zeros((1, 1, sample_rate * 2), dtype=torch.float32)
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
