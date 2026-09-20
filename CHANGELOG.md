# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-09-20

### Added

- Compatibility section documenting ComfyUI, Python, and dependency requirements.
- Troubleshooting section covering report visibility, import failures, and cloud platform availability.
- Example workflow table distinguishing UI and API formats.

### Changed

- Removed `soundfile` from dependencies. The pack does not decode audio files, so the library was never used at runtime.
- Clarified that both nodes perform all processing on the CPU and do not require a GPU.

## [0.3.0] - 2026-09-20

### Added

- `Audio Standards Fixer` node with per-criterion fix toggles.
- `core/audio_io.py` with shared tensor and NumPy conversion helpers.
- `core/fixer.py` with linear gain, a 4x oversampled lookahead limiter, and PLR based feasibility assessment.
- Before and after report with actions applied and a final status.
- Thirteen unit tests for the fixer.
- Example UI and API workflows for the fixer.

### Changed

- Split the evaluator into `nodes_evaluator.py` and removed `nodes.py`.
- Renamed the registry display name to `Audio Quality Check`.
- Documented both nodes, the fix capability matrix, and the feasibility table.

## [0.2.1] - 2026-09-20

### Added

- Criteria table, usage flow, sample report, and limitations in the README.
- Example UI and API workflows for the evaluator.

### Fixed

- Corrected the claim that reports render on the node body. A `Preview Any` node is required.
- Fixed the sample report overall status so it matches the failed criteria.

## [0.2.0] - 2026-09-06

### Added

- `requires-python` constraint, set to 3.9 or newer.
- Explicit type annotations across the evaluator and its tests.
- `logs/` to `.gitignore`.

### Changed

- Relicensed from a proprietary notice to Apache-2.0.
- Updated the author, repository URL, and publisher ID to `khip01`.
- Upgraded `actions/checkout` and `actions/setup-python` to v7.

### Removed

- Empty `Icon` field from the registry metadata.

## [0.1.0] - 2026-09-06

### Added

- Initial release.
- `Audio Quality Evaluator` node measuring integrated loudness, true peak, noise floor, loudness range, and dropout.
- `core/evaluator.py` with modular measurement functions.
- Six unit tests and a GitHub Actions test workflow.
- Apache-2.0 license.
