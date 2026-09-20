from .nodes_evaluator import AudioQualityEvaluator
from .nodes_fixer import AudioStandardsFixer

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "AudioQualityEvaluator": AudioQualityEvaluator,
    "AudioStandardsFixer": AudioStandardsFixer,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "AudioQualityEvaluator": "Audio Quality Evaluator",
    "AudioStandardsFixer": "Audio Standards Fixer",
}

__all__: list[str] = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
