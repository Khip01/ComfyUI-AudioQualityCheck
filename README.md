# ComfyUI-AudioQualityCheck

Automated audio quality evaluation custom node for ComfyUI for thesis research on broadcast-style audio quality thresholds.

## Features

- **Integrated Loudness (LUFS):** Evaluates overall loudness using ITU-R BS.1770-4 standard (Target: -14.0 LUFS +/- 2.0 LU).
- **True Peak (dBTP):** Measures inter-sample peak amplitude using 4x oversampling (Target: <= -1.0 dBTP).
- **Noise Floor (dBFS):** Evaluates background hiss in 50ms quiet segments (Target: <= -40.0 dBFS).
- **Loudness Range (LRA):** Measures vocal dynamic variation (Target: <= 3.0 LU).
- **Dropout Detection:** Detects maximum consecutive silence duration (Target: <= 2.0s).
- **Dual Output:** Outputs `STRING` report, `BOOLEAN` status, and on-node UI text panel.

## Installation

### Manual Installation
Copy or clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Khip01/ComfyUI-AudioQualityCheck.git
pip install -r ComfyUI-AudioQualityCheck/requirements.txt
```

## Node Specification

- **Category:** `audio/quality`
- **Node Name:** `Audio Quality Evaluator`
- **Inputs:** `audio` (`AUDIO`), `target_lufs` (`FLOAT`), `lufs_tolerance` (`FLOAT`), `max_true_peak_dbtp` (`FLOAT`), `max_noise_floor_dbfs` (`FLOAT`), `max_lra_lu` (`FLOAT`), `max_dropout_seconds` (`FLOAT`)
- **Outputs:** `audio` (`AUDIO`), `report` (`STRING`), `is_valid` (`BOOLEAN`)

## License

Apache-2.0 - Copyright (c) 2026 khip01. See LICENSE for details.
