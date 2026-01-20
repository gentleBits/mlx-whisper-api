"""Configuration settings for the MLX Whisper API."""

import os

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mlx-community/whisper-large-v3-mlx")
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "100"))
HUGGINGFACE_CACHE = os.getenv("HUGGINGFACE_CACHE", os.path.expanduser("~/.cache/huggingface"))

# Supported models (from mlx-community)
SUPPORTED_MODELS = [
    "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-large-v3-mlx",
]
