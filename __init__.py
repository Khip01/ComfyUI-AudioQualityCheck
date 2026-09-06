from .nodes import AudioQualityEvaluator

NODE_CLASS_MAPPINGS: dict[str, type[AudioQualityEvaluator]] = {
    "AudioQualityEvaluator": AudioQualityEvaluator
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "AudioQualityEvaluator": "Audio Quality Evaluator"
}

__all__: list[str] = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
