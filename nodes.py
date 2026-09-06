from typing import Any

import torch

from .core.evaluator import evaluate_audio_quality

class AudioQualityEvaluator:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": (
                    "FLOAT",
                    {"default": -14.0, "min": -70.0, "max": 0.0, "step": 0.5},
                ),
                "lufs_tolerance": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "max_true_peak_dbtp": (
                    "FLOAT",
                    {"default": -1.0, "min": -20.0, "max": 0.0, "step": 0.5},
                ),
                "max_noise_floor_dbfs": (
                    "FLOAT",
                    {"default": -40.0, "min": -100.0, "max": 0.0, "step": 1.0},
                ),
                "max_lra_lu": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5},
                ),
                "max_dropout_seconds": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES: tuple[str, str, str] = ("AUDIO", "STRING", "BOOLEAN")
    RETURN_NAMES: tuple[str, str, str] = ("audio", "report", "is_valid")
    FUNCTION: str = "evaluate"
    CATEGORY: str = "audio/quality"

    def evaluate (
        self,
        audio: dict[str, Any],
        target_lufs: float,
        lufs_tolerance: float,
        max_true_peak_dbtp: float,
        max_noise_floor_dbfs: float,
        max_lra_lu: float,
        max_dropout_seconds: float
    ) -> dict[str, Any]:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        report, is_valid = evaluate_audio_quality(
            waveform=waveform,
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            lufs_tolerance=lufs_tolerance,
            max_true_peak_dbtp=max_true_peak_dbtp,
            max_noise_floor_dbfs=max_noise_floor_dbfs,
            max_lra_lu=max_lra_lu,
            max_dropout_seconds=max_dropout_seconds,
        )

        return {
            "ui": {
                "text": [report]
            },
            "result": (audio, report, is_valid)
        }
