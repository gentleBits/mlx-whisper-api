"""Generate test audio fixtures for integration tests.

This script creates minimal audio files for testing the transcription API.
The files are small to keep tests fast.
"""

import wave
import struct
import math
import os
from pathlib import Path


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> list[int]:
    """Generate a sine wave as 16-bit PCM samples.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        List of 16-bit integer samples
    """
    samples = []
    num_samples = int(sample_rate * duration)
    max_amplitude = 32767 * amplitude

    for i in range(num_samples):
        t = i / sample_rate
        value = max_amplitude * math.sin(2 * math.pi * frequency * t)
        samples.append(int(value))

    return samples


def generate_silence(duration: float, sample_rate: int = 16000) -> list[int]:
    """Generate silence as 16-bit PCM samples.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        List of zero-valued 16-bit integer samples
    """
    num_samples = int(sample_rate * duration)
    return [0] * num_samples


def write_wav_file(
    filepath: str,
    samples: list[int],
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write samples to a WAV file.

    Args:
        filepath: Output file path
        samples: List of 16-bit integer samples
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
    """
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)

        # Pack samples as 16-bit signed integers
        packed_samples = struct.pack(f"<{len(samples)}h", *samples)
        wav_file.writeframes(packed_samples)


def main():
    """Generate all test audio fixtures."""
    # Get the audio fixtures directory
    fixtures_dir = Path(__file__).parent / "audio"
    fixtures_dir.mkdir(exist_ok=True)

    # Parameters
    sample_rate = 16000  # 16kHz - standard for Whisper
    duration = 2.0  # 2 seconds

    # Generate test files
    print("Generating test audio fixtures...")

    # 1. sample_en.wav - A simple tone (440Hz A note)
    # This file is for testing the basic transcription pipeline
    # Note: Since this is just a tone, Whisper may not produce meaningful text
    # but it will test that the pipeline works
    samples = generate_sine_wave(440, duration, sample_rate)
    filepath = fixtures_dir / "sample_en.wav"
    write_wav_file(str(filepath), samples, sample_rate)
    print(f"  Created: {filepath} ({os.path.getsize(filepath)} bytes)")

    # 2. silence.wav - Silent audio for edge case testing
    samples = generate_silence(duration, sample_rate)
    filepath = fixtures_dir / "silence.wav"
    write_wav_file(str(filepath), samples, sample_rate)
    print(f"  Created: {filepath} ({os.path.getsize(filepath)} bytes)")

    # 3. short.wav - Very short audio (0.5 seconds)
    samples = generate_sine_wave(440, 0.5, sample_rate)
    filepath = fixtures_dir / "short.wav"
    write_wav_file(str(filepath), samples, sample_rate)
    print(f"  Created: {filepath} ({os.path.getsize(filepath)} bytes)")

    print("\nDone! Test audio fixtures generated successfully.")


if __name__ == "__main__":
    main()
