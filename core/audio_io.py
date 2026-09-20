"""
Modul Konversi Audio antara Tensor PyTorch dan Matriks NumPy.
Ringkasan Fungsi:
1. convert_tensor_to_numpy : Mengubah tensor ComfyUI menjadi matriks NumPy float32.
2. convert_numpy_to_tensor : Mengubah matriks NumPy menjadi tensor ComfyUI.
"""

from typing import Any, cast

import numpy as np
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


# 2. Konversi Matriks NumPy ke Tensor PyTorch
# ---
# Membalik konversi di atas agar hasil pemrosesan dapat diteruskan ke node
# berikutnya. Dimensi batch ditambahkan kembali sesuai format ComfyUI.
def convert_numpy_to_tensor(
    audio_np: np.ndarray[Any, np.dtype[np.float32]],
) -> torch.Tensor:
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=1)
    transposed: np.ndarray[Any, np.dtype[np.float32]] = np.ascontiguousarray(
        audio_np.T, dtype=np.float32
    )
    return cast(torch.Tensor, torch.from_numpy(transposed).unsqueeze(0))
