"""Shared test fixtures."""

import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from app.main import app


# Test model (smallest available for fast tests)
TEST_MODEL = "mlx-community/whisper-tiny-mlx"


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client for FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def fixtures_path():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_audio_path(fixtures_path):
    """Path to sample audio file."""
    path = fixtures_path / "audio" / "sample_en.wav"
    if not path.exists():
        pytest.skip(f"Test audio file not found: {path}")
    return path


@pytest.fixture
def silence_audio_path(fixtures_path):
    """Path to silent audio file."""
    path = fixtures_path / "audio" / "silence.wav"
    if not path.exists():
        pytest.skip(f"Silent audio file not found: {path}")
    return path


@pytest.fixture
def short_audio_path(fixtures_path):
    """Path to short audio file."""
    path = fixtures_path / "audio" / "short.wav"
    if not path.exists():
        pytest.skip(f"Short audio file not found: {path}")
    return path


@pytest.fixture
def sample_audio_bytes(sample_audio_path):
    """Raw bytes of sample audio for upload testing."""
    return sample_audio_path.read_bytes()


@pytest.fixture
def harvard_audio_path(fixtures_path):
    """Path to Harvard sentences audio file (real speech)."""
    path = fixtures_path / "audio" / "harvard_sample.wav"
    if not path.exists():
        pytest.skip(f"Harvard audio file not found: {path}")
    return path
