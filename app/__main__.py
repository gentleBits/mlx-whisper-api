"""Command-line interface for MLX Whisper API.

Run the server with: mlx-whisper-api [OPTIONS]

Examples:
    mlx-whisper-api
    mlx-whisper-api --port 8080
    mlx-whisper-api --host 127.0.0.1 --port 8080 --reload
    mlx-whisper-api --log-level DEBUG

Alternative (without installation):
    python -m app [OPTIONS]
"""

import argparse
import os
import sys


def get_version() -> str:
    """Get the API version from main module."""
    from app.main import API_VERSION
    return API_VERSION


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        prog="mlx-whisper-api",
        description="MLX Whisper API - REST API for audio-to-text transcription "
                    "using MLX-optimized Whisper models on Apple Silicon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlx-whisper-api                          Start with defaults (0.0.0.0:1738)
  mlx-whisper-api --port 8080              Start on port 8080
  mlx-whisper-api --host 127.0.0.1         Bind to localhost only
  mlx-whisper-api --reload                 Enable auto-reload for development
  mlx-whisper-api --workers 4              Run with 4 worker processes

Environment variables:
  HOST              Server bind address (default: 0.0.0.0)
  PORT              Server port (default: 1738)
  LOG_LEVEL         Logging level (default: INFO)
  DEFAULT_MODEL     Default Whisper model
  MAX_AUDIO_SIZE_MB Maximum upload size in MB (default: 100)
""",
    )

    # Server options
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: %(default)s)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("PORT", "1738")),
        help="Port to bind the server to (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: 1, incompatible with --reload)",
    )

    # Development options
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        default=False,
        help="Enable auto-reload on code changes (development mode)",
    )

    # Logging options
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Set the logging level (default: %(default)s)",
    )

    # Version
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate arguments
    if args.reload and args.workers and args.workers > 1:
        parser.error("--reload cannot be used with multiple workers")

    # Set log level environment variable so it's picked up by the app
    os.environ["LOG_LEVEL"] = args.log_level

    # Import uvicorn here to avoid slow startup for --help/--version
    import uvicorn

    # Build uvicorn config
    uvicorn_kwargs = {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "log_level": args.log_level.lower(),
    }

    if args.workers:
        uvicorn_kwargs["workers"] = args.workers

    print(f"Starting MLX Whisper API v{get_version()}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs:   http://{args.host}:{args.port}/docs")
    print()

    try:
        uvicorn.run(**uvicorn_kwargs)
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
