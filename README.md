# MLX Whisper API

REST API for audio-to-text transcription using MLX-optimized Whisper models on Apple Silicon.

## Features

- **Audio file transcription** with multiple format support (wav, mp3, m4a, flac, ogg)
- **MLX-optimized Whisper models** including tiny, small, and large-v3
- **Model download management** with progress tracking
- **Prompt support** for guided transcription
- **Runs entirely offline** on Apple Silicon Macs (M1/M2/M3/M4)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10-3.12

## Installation

### Via Homebrew (Recommended)

```bash
brew tap gentleBits/mlx-whisper-api
brew install mlx-whisper-api
```

### Via pip

Install directly from GitHub:
```bash
pip install git+https://github.com/gentleBits/mlx-whisper-api.git
```

Or clone and install locally:
```bash
git clone https://github.com/gentleBits/mlx-whisper-api.git
cd mlx-whisper-api
pip install .
```

After installation, start the server with:
```bash
mlx-whisper-api
```

### From Source (Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/gentleBits/mlx-whisper-api.git
   cd mlx-whisper-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the API server:
```bash
python -m app
```

With custom host and port:
```bash
python -m app --host 127.0.0.1 --port 8080
```

Development mode with auto-reload:
```bash
python -m app --reload
```

Production mode with multiple workers:
```bash
python -m app --workers 4
```

View all CLI options:
```bash
python -m app --help
```

The API will be available at `http://localhost:1738`. Interactive documentation is available at `http://localhost:1738/docs`.

## API Endpoints

### Health Check

```bash
curl http://localhost:1738/health
```

Response:
```json
{"status": "healthy"}
```

### Transcribe Audio

Transcribe an audio file to text.

```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@audio.wav"
```

With optional parameters:
```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@meeting.mp3" \
  -F "model=mlx-community/whisper-large-v3-mlx" \
  -F "language=en" \
  -F "prompt=Meeting attendees: Alice, Bob, Charlie. Technical discussion about Kubernetes."
```

**Parameters:**
- `file` (required): Audio file (wav, mp3, m4a, flac, ogg)
- `model` (optional): Model identifier, defaults to `mlx-community/whisper-large-v3-mlx`
- `language` (optional): Two-letter language code (e.g., "en", "fr")
- `prompt` (optional): Text prompt to guide transcription

**Response:**
```json
{
  "text": "Transcribed text here",
  "language": "en",
  "model": "mlx-community/whisper-large-v3-mlx"
}
```

### List Models

List all supported models and their download status.

```bash
curl http://localhost:1738/models
```

Response:
```json
{
  "models": [
    {
      "id": "mlx-community/whisper-large-v3-mlx",
      "name": "Whisper Large V3",
      "size": "large",
      "quantization": null,
      "english_only": false,
      "status": "downloaded",
      "size_bytes": 3100000000,
      "download_progress": null
    },
    {
      "id": "mlx-community/whisper-tiny-mlx",
      "name": "Whisper Tiny",
      "size": "tiny",
      "quantization": null,
      "english_only": false,
      "status": "not_downloaded",
      "size_bytes": null,
      "download_progress": null
    }
  ]
}
```

### Model Status

Get detailed status of a specific model.

```bash
curl http://localhost:1738/models/mlx-community%2Fwhisper-large-v3-mlx/status
```

Response (downloaded):
```json
{
  "id": "mlx-community/whisper-large-v3-mlx",
  "status": "downloaded",
  "path": "/Users/.../.cache/huggingface/hub/models--mlx-community--whisper-large-v3-mlx",
  "size_bytes": 3100000000
}
```

Response (downloading):
```json
{
  "id": "mlx-community/whisper-large-v3-mlx",
  "status": "downloading",
  "progress": 0.67,
  "downloaded_bytes": 2077000000,
  "total_bytes": 3100000000
}
```

### Download Model

Initiate download of a model (non-blocking).

```bash
curl -X POST http://localhost:1738/models/mlx-community%2Fwhisper-tiny-mlx/download
```

Response:
```json
{
  "id": "mlx-community/whisper-tiny-mlx",
  "status": "download_started",
  "message": "Model download initiated. Check /models/{model_id}/status for progress."
}
```

### Delete Model

Remove a downloaded model from local cache.

```bash
curl -X DELETE http://localhost:1738/models/mlx-community%2Fwhisper-tiny-mlx
```

Response:
```json
{
  "id": "mlx-community/whisper-tiny-mlx",
  "status": "deleted"
}
```

## Configuration

### CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--host` | | `0.0.0.0` | Server bind address |
| `--port` | `-p` | `1738` | Server port |
| `--reload` | `-r` | off | Enable auto-reload for development |
| `--workers` | `-w` | `1` | Number of worker processes |
| `--log-level` | `-l` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--version` | `-V` | | Show version and exit |

### Environment Variables

Environment variables are used as defaults when CLI arguments are not provided.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `1738` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `DEFAULT_MODEL` | `mlx-community/whisper-large-v3-mlx` | Default model for transcription |
| `MAX_AUDIO_SIZE_MB` | `100` | Maximum upload file size in MB |
| `HUGGINGFACE_CACHE` | `~/.cache/huggingface` | HuggingFace cache directory |

## Examples

### Basic Transcription
```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@recording.wav"
```

### Transcription with Language Hint
```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@french_audio.mp3" \
  -F "language=fr"
```

### Transcription with Context Prompt
```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@medical_recording.m4a" \
  -F "prompt=Medical consultation about diabetes treatment. Patient name: John Smith."
```

### Using a Smaller Model for Speed
```bash
curl -X POST http://localhost:1738/transcribe \
  -F "file=@quick_note.wav" \
  -F "model=mlx-community/whisper-tiny-mlx"
```

## Supported Models

All models are from the [mlx-community](https://huggingface.co/mlx-community) on HuggingFace.

| Model ID | Size | Description |
|----------|------|-------------|
| `mlx-community/whisper-tiny-mlx` | Tiny | Fastest, lowest accuracy |
| `mlx-community/whisper-small-mlx` | Small | Balanced speed and accuracy |
| `mlx-community/whisper-large-v3-mlx` | Large | Best accuracy (default) |

**Notes:**
- Larger models provide better accuracy but require more memory and processing time
- The default model is `whisper-large-v3-mlx` for best quality

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only fast tests (skip slow model inference tests)
pytest -m "not slow"
```

## License

MIT License
