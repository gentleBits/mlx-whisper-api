"""Unit tests for input validation functions."""

import pytest

from app.validation import (
    validate_language,
    validate_prompt,
    validate_audio_format,
    sanitize_filename,
    get_file_extension,
    MAX_PROMPT_LENGTH,
    VALID_LANGUAGE_CODES,
    SUPPORTED_AUDIO_FORMATS,
)
from app.errors import (
    InvalidLanguageError,
    PromptTooLongError,
    UnsupportedFormatError,
)


class TestValidateLanguage:
    """Tests for validate_language function."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert validate_language(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert validate_language("") is None
        assert validate_language("  ") is None

    def test_valid_language_codes(self):
        """Valid ISO 639-1 codes should pass."""
        valid_codes = ["en", "fr", "de", "es", "ja", "zh", "ko", "ru"]
        for code in valid_codes:
            assert validate_language(code) == code

    def test_uppercase_normalized(self):
        """Uppercase codes should be normalized to lowercase."""
        assert validate_language("EN") == "en"
        assert validate_language("FR") == "fr"
        assert validate_language("De") == "de"

    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert validate_language("  en  ") == "en"
        assert validate_language("\ten\n") == "en"

    def test_invalid_language_code_raises(self):
        """Invalid language codes should raise InvalidLanguageError."""
        invalid_codes = ["xx", "invalid", "123", "e", "english", "!@#"]
        for code in invalid_codes:
            with pytest.raises(InvalidLanguageError) as exc_info:
                validate_language(code)
            assert code in str(exc_info.value)

    def test_three_letter_codes(self):
        """Some three-letter codes are valid (e.g., 'yue' for Cantonese)."""
        assert validate_language("yue") == "yue"

    def test_all_whisper_languages_valid(self):
        """All languages in VALID_LANGUAGE_CODES should be accepted."""
        for code in VALID_LANGUAGE_CODES:
            assert validate_language(code) == code


class TestValidatePrompt:
    """Tests for validate_prompt function."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert validate_prompt(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert validate_prompt("") is None
        assert validate_prompt("   ") is None

    def test_valid_prompt_passes(self):
        """Valid prompts should pass through."""
        prompt = "This is a meeting about Kubernetes and Docker."
        assert validate_prompt(prompt) == prompt

    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert validate_prompt("  hello  ") == "hello"
        assert validate_prompt("\n\ttest\n\t") == "test"

    def test_prompt_at_max_length(self):
        """Prompt at max length should pass."""
        prompt = "a" * MAX_PROMPT_LENGTH
        assert validate_prompt(prompt) == prompt

    def test_prompt_exceeds_max_length_raises(self):
        """Prompt exceeding max length should raise PromptTooLongError."""
        prompt = "a" * (MAX_PROMPT_LENGTH + 1)
        with pytest.raises(PromptTooLongError) as exc_info:
            validate_prompt(prompt)
        assert str(MAX_PROMPT_LENGTH) in str(exc_info.value)

    def test_unicode_prompt(self):
        """Unicode prompts should work."""
        prompt = "日本語のテキスト"
        assert validate_prompt(prompt) == prompt


class TestGetFileExtension:
    """Tests for get_file_extension function."""

    def test_standard_extensions(self):
        """Standard extensions should be extracted correctly."""
        assert get_file_extension("audio.wav") == ".wav"
        assert get_file_extension("music.mp3") == ".mp3"
        assert get_file_extension("recording.m4a") == ".m4a"

    def test_uppercase_normalized(self):
        """Extensions should be normalized to lowercase."""
        assert get_file_extension("audio.WAV") == ".wav"
        assert get_file_extension("audio.MP3") == ".mp3"

    def test_no_extension(self):
        """Files without extension should return empty string."""
        assert get_file_extension("noextension") == ""

    def test_multiple_dots(self):
        """Only the last extension should be returned."""
        assert get_file_extension("file.backup.wav") == ".wav"
        assert get_file_extension("my.audio.file.mp3") == ".mp3"

    def test_hidden_files(self):
        """Hidden files (starting with dot) should be handled."""
        assert get_file_extension(".hidden") == ".hidden"
        assert get_file_extension(".hidden.wav") == ".wav"


class TestValidateAudioFormat:
    """Tests for validate_audio_format function."""

    def test_valid_formats(self):
        """All supported formats should pass."""
        for fmt in SUPPORTED_AUDIO_FORMATS:
            filename = f"audio{fmt}"
            assert validate_audio_format(filename) == fmt

    def test_uppercase_formats_pass(self):
        """Uppercase extensions should pass."""
        assert validate_audio_format("audio.WAV") == ".wav"
        assert validate_audio_format("audio.MP3") == ".mp3"

    def test_unsupported_format_raises(self):
        """Unsupported formats should raise UnsupportedFormatError."""
        unsupported = ["audio.txt", "audio.pdf", "audio.aac", "audio.wma"]
        for filename in unsupported:
            with pytest.raises(UnsupportedFormatError) as exc_info:
                validate_audio_format(filename)
            error = exc_info.value
            assert error.details["supported_formats"] == sorted(SUPPORTED_AUDIO_FORMATS)

    def test_no_extension_raises(self):
        """Files without extension should raise UnsupportedFormatError."""
        with pytest.raises(UnsupportedFormatError):
            validate_audio_format("noextension")


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_none_returns_default(self):
        """None should return default filename."""
        assert sanitize_filename(None) == "audio.wav"

    def test_empty_returns_default(self):
        """Empty string should return default filename."""
        assert sanitize_filename("") == "audio.wav"
        assert sanitize_filename("   ") == "audio.wav"

    def test_normal_filename_unchanged(self):
        """Normal filenames should pass through."""
        assert sanitize_filename("audio.wav") == "audio.wav"
        assert sanitize_filename("my_recording.mp3") == "my_recording.mp3"

    def test_path_traversal_removed(self):
        """Path traversal attempts should be sanitized."""
        assert "/" not in sanitize_filename("../../../etc/passwd")
        assert "\\" not in sanitize_filename("..\\..\\windows\\system32")

    def test_null_bytes_removed(self):
        """Null bytes should be removed."""
        assert "\x00" not in sanitize_filename("audio\x00.wav")

    def test_leading_dots_stripped(self):
        """Leading dots should be stripped."""
        assert not sanitize_filename("...hidden").startswith(".")

    def test_slashes_replaced_with_underscores(self):
        """Slashes should be replaced with underscores."""
        assert sanitize_filename("path/to/file.wav") == "path_to_file.wav"
        assert sanitize_filename("path\\to\\file.wav") == "path_to_file.wav"

    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert sanitize_filename("  audio.wav  ") == "audio.wav"
