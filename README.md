# ComfyUI-AudioQualityCheck

Automated audio quality evaluation custom node for ComfyUI.

The node inspects an `AUDIO` input, measures five quality criteria, and returns a human readable report plus a boolean verdict. It does not modify the audio, so it can be inserted anywhere in an existing audio pipeline as a quality gate.

## Criteria

| Criterion | Default threshold | Measurement method |
|---|---|---|
| Integrated Loudness | -14.0 LUFS +/- 2.0 | ITU-R BS.1770-4 gated loudness via `pyloudnorm` |
| True Peak | <= -1.0 dBTP | 4x polyphase oversampling (`scipy.signal.resample_poly`) |
| Noise Floor | <= -40.0 dBFS | 10th percentile of 50 ms RMS windows |
| Loudness Range | <= 3.0 LU | EBU Tech 3342 short-term loudness distribution |
| Dropout | <= 2.0 s | Longest consecutive run below -60 dBFS |

All thresholds are exposed as node widgets, so they can be tuned per project without editing code.

## Installation

### Manual Installation
Copy or clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Khip01/ComfyUI-AudioQualityCheck.git
pip install -r ComfyUI-AudioQualityCheck/requirements.txt
```

Restart ComfyUI after installation. The node appears under the `audio/quality` category.

## Usage

Connect any node that outputs `AUDIO` to the node input:

```
LoadAudio -> Audio Quality Evaluator -> SaveAudio
                        |
                        +-> Preview Any (report)
```

- `audio` output passes the original audio through unchanged, so it can feed a downstream save or processing node.
- `report` output is a `STRING` containing the full measurement report.
- `is_valid` output is a `BOOLEAN` that is `true` only when every criterion passes.

To read the report inside the graph, connect the `report` output to a core `Preview Any` node. ComfyUI does not render custom node text output on the node body by default, so a preview node is required.

Example workflows are provided in the `workflows/` directory:

- `audio_quality_evaluator_ui.json` for the ComfyUI graph editor.
- `audio_quality_evaluator_api.json` for API format submissions.

Both examples reference `example_audio.wav`. Replace it with your own file after loading the workflow.

## Node Specification

- **Category:** `audio/quality`
- **Node Name:** `Audio Quality Evaluator`
- **Inputs:** `audio` (`AUDIO`), `target_lufs` (`FLOAT`), `lufs_tolerance` (`FLOAT`), `max_true_peak_dbtp` (`FLOAT`), `max_noise_floor_dbfs` (`FLOAT`), `max_lra_lu` (`FLOAT`), `max_dropout_seconds` (`FLOAT`)
- **Outputs:** `audio` (`AUDIO`), `report` (`STRING`), `is_valid` (`BOOLEAN`)

## Report Format

```
=== Audio Quality Evaluation Report ===
Overall Status       : FAILED
Sample Rate          : 44100 Hz
Channels             : 2
----------------------------------------
1. Integrated Loudness: -13.33 LUFS [PASSED]
   Target            : -14.0 +/- 2.0 LUFS (Range: -16.0 to -12.0)
2. True Peak         : 0.02 dBTP [FAILED]
   Target Max        : -1.0 dBTP
3. Noise Floor       : -96.31 dBFS [PASSED]
   Target Max        : -40.0 dBFS
4. Loudness Range    : 3.15 LU [FAILED]
   Target Max        : 3.0 LU
5. Max Dropout       : 1.05 s [PASSED]
   Target Max        : 2.0 s
========================================
```

## Limitations

- The node is a pass-through quality gate. It reports a verdict but does not correct audio that fails.
- Decode integrity of the source file is not checked, because the node receives already decoded audio.
- Audio and video synchronization is out of scope, as the node only receives audio.

## Development

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -v --confcutdir=tests
```

## License

Apache-2.0 - Copyright (c) 2026 Khip01. See LICENSE for details.
