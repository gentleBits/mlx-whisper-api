# typed: false
# frozen_string_literal: true

# Homebrew formula for MLX Whisper API
class MlxWhisperApi < Formula
  include Language::Python::Virtualenv

  desc "REST API for audio-to-text transcription using MLX Whisper on Apple Silicon"
  homepage "https://github.com/gentleBits/mlx-whisper-api"
  url "https://github.com/gentleBits/mlx-whisper-api/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "b4d426598d4d7c1ad054f403cbc36a8c85995ae4e2a4efc5092404cd3bfaa95c"
  license "MIT"
  head "https://github.com/gentleBits/mlx-whisper-api.git", branch: "main"

  # MLX requires Apple Silicon
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  def install
    # Create a virtualenv and install via pip (allows binary wheels)
    venv = virtualenv_create(libexec, "python3.12")

    # Use system pip to install dependencies into the venv
    system Formula["python@3.12"].opt_bin/"python3.12", "-m", "pip", "install",
           "--target=#{libexec}/lib/python3.12/site-packages",
           "mlx-whisper==0.4.2",
           "fastapi==0.115.7",
           "uvicorn[standard]==0.34.0",
           "python-multipart==0.0.20",
           "huggingface-hub==0.27.1"

    # Install the package itself
    venv.pip_install_and_link buildpath
  end

  def caveats
    <<~EOS
      MLX Whisper API has been installed!

      To start the server:
        mlx-whisper-api

      To start on a custom port:
        mlx-whisper-api --port 8080

      For all options:
        mlx-whisper-api --help

      API documentation will be available at:
        http://localhost:1738/docs

      Note: The first transcription may take longer as the Whisper model
      needs to be downloaded from HuggingFace Hub.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/mlx-whisper-api --version")
  end
end
