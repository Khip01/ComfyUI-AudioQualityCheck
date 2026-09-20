# ComfyUI-AudioQualityCheck

Automated audio quality evaluation and correction custom nodes for ComfyUI.

The pack provides two nodes. The evaluator inspects an `AUDIO` input, measures five quality criteria, and returns a human readable report plus a boolean verdict. The fixer corrects the criteria that can be corrected, measures the result again, and reports both the before and after state. The evaluator never modifies audio, the fixer only modifies loudness and true peak.

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

### Via ComfyUI Manager

Search for `comfyui-audio-quality-check` in ComfyUI Manager and install. Restart ComfyUI afterwards.

### Manual Installation

Copy or clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Khip01/ComfyUI-AudioQualityCheck.git
pip install -r ComfyUI-AudioQualityCheck/requirements.txt
```

Restart ComfyUI after installation. Both nodes appear under the `audio/quality` category.

## Compatibility

| Requirement | Version |
|---|---|
| ComfyUI | Any version with the `AUDIO` type, `PreviewAny`, and `SaveAudio` nodes |
| Python | 3.9 or newer |
| PyTorch | Provided by ComfyUI |

Dependencies are limited to `numpy`, `scipy`, and `pyloudnorm`, all pure Python wheels that need no build step and no GPU.

The nodes never move audio to the GPU. They call `.cpu().numpy()` on the incoming tensor and do all measurement and processing on the CPU, so they run on CPU-only ComfyUI installations as well.

Audio decoding is handled by ComfyUI itself through `LoadAudio`, which relies on PyAV and torchaudio. This pack does not decode files, so no extra codec library is required for MP3 or other lossy formats.

## Nodes

| Node | Purpose | Modifies audio |
|---|---|---|
| Audio Quality Evaluator | Measure and report the five criteria | No |
| Audio Standards Fixer | Correct loudness and true peak, then measure again | Yes |

## Usage

### Evaluator only

```
LoadAudio -> Audio Quality Evaluator -> SaveAudio
                        |
                        +-> Preview Any (report)
```

### Evaluate, fix, then verify

```
LoadAudio -> Audio Quality Evaluator -> Audio Standards Fixer -> Audio Quality Evaluator -> SaveAudio
                                                |                          |
                                                +-> Preview Any            +-> Preview Any
                                                   (fix report)               (final report)
```

- `audio` output passes the audio through, so it can feed a downstream save or processing node.
- `report` output is a `STRING` containing the full measurement report.
- `is_valid` and `is_fixed` are `BOOLEAN` outputs.

To read a report inside the graph, connect the `report` output to a core `Preview Any` node. ComfyUI does not render custom node text output on the node body by default, so a preview node is required.

## Example Workflows

Four workflows are provided in the `workflows/` directory. The UI variants can be loaded through the ComfyUI graph editor. The API variants are prompt-format JSON for programmatic submission.

| File | Format | What it does |
|---|---|---|
| `audio_quality_evaluator_ui.json` | Graph editor | Loads one audio file, measures it, and previews the report |
| `audio_quality_evaluator_api.json` | API prompt | Same as above, for scripted submission |
| `audio_quality_fixer_ui.json` | Graph editor | Measures, fixes, measures again, and previews all three reports |
| `audio_quality_fixer_api.json` | API prompt | Same as above, for scripted submission |

The fixer examples also include a `SaveAudio` node so the corrected result can be written to disk.

All examples reference `example_audio.wav`. Replace it with your own file after loading the workflow.

## Troubleshooting

### The report does not appear on the node

This is expected. ComfyUI only renders text returned through the `ui` channel for nodes that ship a matching frontend extension, and this pack ships none. Connect the `report` output to a core `Preview Any` node to read it.

### Nodes do not appear in the Add Node menu

Check the ComfyUI startup log for a line mentioning `ComfyUI-AudioQualityCheck`. If the module failed to import, the log shows the traceback. The most common causes are a missing dependency and a Python version older than 3.9.

### Nodes show as red or missing on a cloud platform

Managed platforms such as Comfy Cloud, RunningHub, and ComfyICU run a curated set of custom nodes. A node published to the Comfy Registry is not automatically available on those platforms. Self-hosted ComfyUI installations and platforms that allow arbitrary node installation are unaffected.

### The workflow loads but audio cannot be found

The example workflows reference `example_audio.wav`, which is not bundled. Select your own file in the `LoadAudio` node after loading a workflow.

## Node Specification

### Audio Quality Evaluator

- **Category:** `audio/quality`
- **Inputs:** `audio` (`AUDIO`), `target_lufs` (`FLOAT`), `lufs_tolerance` (`FLOAT`), `max_true_peak_dbtp` (`FLOAT`), `max_noise_floor_dbfs` (`FLOAT`), `max_lra_lu` (`FLOAT`), `max_dropout_seconds` (`FLOAT`)
- **Outputs:** `audio` (`AUDIO`), `report` (`STRING`), `is_valid` (`BOOLEAN`)

### Audio Standards Fixer

- **Category:** `audio/quality`
- **Inputs:** `audio` (`AUDIO`), `target_lufs` (`FLOAT`), `lufs_tolerance` (`FLOAT`), `max_true_peak_dbtp` (`FLOAT`), `target_true_peak_dbtp` (`FLOAT`), `plr_gain_only_threshold` (`FLOAT`), `max_noise_floor_dbfs` (`FLOAT`), `max_lra_lu` (`FLOAT`), `max_dropout_seconds` (`FLOAT`), `fix_loudness` (`BOOLEAN`), `fix_true_peak` (`BOOLEAN`)
- **Outputs:** `audio` (`AUDIO`), `report` (`STRING`), `is_fixed` (`BOOLEAN`)

The fixer exposes one toggle per correctable criterion. A criterion whose toggle is off is left untouched and only reported.

## How the Fixer Works

1. Measure the input and compute the peak to loudness ratio (PLR).
2. Apply a linear gain toward `target_lufs` when `fix_loudness` is on.
3. Apply a lookahead limiter toward `target_true_peak_dbtp` when `fix_true_peak` is on.
4. Measure again. If limiting pushed loudness outside `lufs_tolerance`, correct the gain and limit again. The loop stops after three passes.
5. Report the before and after measurements, the applied actions, and a final status.

The limiter works on 4x oversampled audio so that inter-sample peaks are held under the ceiling. It uses a lookahead minimum filter to drop the gain before a peak arrives, then smooths the release.

`target_true_peak_dbtp` defaults to `-2.0` dBTP, which is stricter than the `-1.0` dBTP pass limit, so the result stays compliant after lossy re-encoding. If it is set above the pass limit, the stricter of the two values is used and the report notes the substitution.

### Feasibility

PLR is the difference between true peak and integrated loudness. A linear gain shifts both by the same amount, so PLR does not change and determines how much limiting is required.

| PLR above `plr_gain_only_threshold` | Feasibility |
|---|---|
| No limiting required | gain only |
| Up to 1.5 dB | light limiting |
| 1.5 to 6.0 dB | heavy limiting |
| Above 6.0 dB | not achievable |

## What the Fixer Can and Cannot Correct

| Criterion | Corrected | Method | Note |
|---|---|---|---|
| Integrated Loudness | Yes | Linear gain | High precision |
| True Peak | Yes | Lookahead limiter on 4x oversampling | Verified by re-measurement |
| Noise Floor | No | Reported only | Originates from the recording source |
| Loudness Range | No | Reported only | Gain and limiting cannot reduce dynamic range |
| Dropout | No | Reported only | Missing audio cannot be reconstructed |

A final status of `FIXED WITH NOTES` means the correctable criteria now pass while one or more reported-only criteria still fail.

## Report Formats

Evaluator report:

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

Fixer report:

```
=== Audio Standards Fixer Report ===
Sample Rate          : 44100 Hz
Channels             : 2
----------------------------------------
BEFORE
  Integrated Loudness: -13.33 LUFS [PASSED]
  True Peak          : 0.02 dBTP [FAILED]
  Noise Floor        : -96.31 dBFS [PASSED]
  Loudness Range     : 3.15 LU [FAILED]
  Max Dropout        : 1.05 s [PASSED]
----------------------------------------
AFTER
  Integrated Loudness: -14.10 LUFS [PASSED]
  True Peak          : -1.99 dBTP [PASSED]
  Noise Floor        : -96.98 dBFS [PASSED]
  Loudness Range     : 3.08 LU [FAILED]
  Max Dropout        : 1.05 s [PASSED]
----------------------------------------
Actions Applied
  Loudness Gain      : -0.67 dB
  True Peak Limiter  : 1.35 dB max reduction
  Limiter Ceiling    : -2.00 dBTP
  PLR                : 13.35 dB
  Feasibility        : light limiting
  Iterations Used    : 2 of 3
  Final Status       : FIXED WITH NOTES
----------------------------------------
Notes
  - Loudness range 3.08 LU exceeds 3.00 LU. Gain and limiting cannot reduce dynamic range; review the source audio or apply compression upstream.
========================================
```

## Limitations

- The evaluator is a pass-through quality gate. It reports a verdict but does not correct audio.
- The fixer corrects loudness and true peak only. Noise floor, loudness range, and dropout are reported without modification.
- Decode integrity of the source file is not checked, because the nodes receive already decoded audio.
- Audio and video synchronization is out of scope, as the nodes only receive audio.

## Development

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -v --confcutdir=tests
```

## License

Apache-2.0 - Copyright (c) 2026 Khip01. See LICENSE for details.
