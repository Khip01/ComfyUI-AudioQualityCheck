from typing import Any

import torch

from .core.fixer import fix_audio_quality


class AudioStandardsFixer:
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
                "target_true_peak_dbtp": (
                    "FLOAT",
                    {"default": -2.0, "min": -20.0, "max": 0.0, "step": 0.5},
                ),
                "plr_gain_only_threshold": (
                    "FLOAT",
                    {"default": 13.0, "min": 0.0, "max": 30.0, "step": 0.5},
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
                "fix_loudness": ("BOOLEAN", {"default": True}),
                "fix_true_peak": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES: tuple[str, str, str] = ("AUDIO", "STRING", "BOOLEAN")
    RETURN_NAMES: tuple[str, str, str] = ("audio", "report", "is_fixed")
    FUNCTION: str = "fix"
    CATEGORY: str = "audio/quality"

    def fix(
        self,
        audio: dict[str, Any],
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
    ) -> dict[str, Any]:
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        fixed_waveform: torch.Tensor
        report: str
        is_fixed: bool
        fixed_waveform, report, is_fixed = fix_audio_quality(
            waveform=waveform,
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            lufs_tolerance=lufs_tolerance,
            max_true_peak_dbtp=max_true_peak_dbtp,
            target_true_peak_dbtp=target_true_peak_dbtp,
            plr_gain_only_threshold=plr_gain_only_threshold,
            max_noise_floor_dbfs=max_noise_floor_dbfs,
            max_lra_lu=max_lra_lu,
            max_dropout_seconds=max_dropout_seconds,
            fix_loudness=fix_loudness,
            fix_true_peak=fix_true_peak,
        )

        fixed_audio: dict[str, Any] = {
            "waveform": fixed_waveform,
            "sample_rate": sample_rate,
        }

        return {
            "ui": {
                "text": [report]
            },
            "result": (fixed_audio, report, is_fixed),
        }
