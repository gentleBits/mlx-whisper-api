"""Input validation utilities for the MLX Whisper API.

Provides validation for:
- Language codes (ISO 639-1)
- Prompt text (length limits)
- Audio file formats
- Model IDs
"""

import re
from typing import Optional

from app.errors import (
    InvalidLanguageError,
    PromptTooLongError,
    UnsupportedFormatError,
)

# Maximum prompt length in characters
MAX_PROMPT_LENGTH = 1000

# ISO 639-1 language codes supported by Whisper
# This is the subset commonly used; Whisper supports more
VALID_LANGUAGE_CODES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
}

# Supported audio formats with their file extensions
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def validate_language(language: Optional[str]) -> Optional[str]:
    """Validate and normalize a language code.

    Args:
        language: Two-letter ISO 639-1 language code (e.g., "en", "fr")

    Returns:
        Normalized lowercase language code, or None if input is None

    Raises:
        InvalidLanguageError: If the language code is invalid
    """
    if language is None:
        return None

    # Normalize to lowercase and strip whitespace
    normalized = language.strip().lower()

    # Check if empty after normalization
    if not normalized:
        return None

    # Validate format (should be 2-3 letters)
    if not re.match(r"^[a-z]{2,3}$", normalized):
        raise InvalidLanguageError(language)

    # Check if it's a known language code
    if normalized not in VALID_LANGUAGE_CODES:
        raise InvalidLanguageError(language)

    return normalized


def validate_prompt(prompt: Optional[str]) -> Optional[str]:
    """Validate and sanitize a transcription prompt.

    Args:
        prompt: Text prompt to guide transcription

    Returns:
        Sanitized prompt, or None if input is None/empty

    Raises:
        PromptTooLongError: If the prompt exceeds the maximum length
    """
    if prompt is None:
        return None

    # Strip whitespace
    sanitized = prompt.strip()

    # Return None if empty after stripping
    if not sanitized:
        return None

    # Check length
    if len(sanitized) > MAX_PROMPT_LENGTH:
        raise PromptTooLongError(len(sanitized), MAX_PROMPT_LENGTH)

    return sanitized


def get_file_extension(filename: str) -> str:
    """Extract and normalize file extension from a filename.

    Args:
        filename: The filename to extract extension from

    Returns:
        Lowercase extension including the dot (e.g., ".wav")
    """
    # Find the last dot in the filename
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        return ext
    return ""


def validate_audio_format(filename: str) -> str:
    """Validate that the audio format is supported.

    Args:
        filename: The filename to validate

    Returns:
        The file extension if valid

    Raises:
        UnsupportedFormatError: If the format is not supported
    """
    ext = get_file_extension(filename)

    if not ext or ext not in SUPPORTED_AUDIO_FORMATS:
        raise UnsupportedFormatError(
            format=ext or "unknown",
            supported_formats=sorted(SUPPORTED_AUDIO_FORMATS),
        )

    return ext


def sanitize_filename(filename: Optional[str]) -> str:
    """Sanitize a filename for safe use.

    Removes path traversal attempts and normalizes the filename.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename, defaults to "audio.wav" if invalid
    """
    if not filename:
        return "audio.wav"

    # Remove path separators to prevent path traversal
    sanitized = filename.replace("/", "_").replace("\\", "_")

    # Remove any null bytes
    sanitized = sanitized.replace("\x00", "")

    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip(".")

    # If empty after sanitization, use default
    if not sanitized:
        return "audio.wav"

    return sanitized
