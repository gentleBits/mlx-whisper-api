# typed: false
# frozen_string_literal: true

# Homebrew formula for MLX Whisper API
#
# To use this formula:
#   1. Create a tap repository: github.com/gentleBits/homebrew-mlx-whisper-api
#   2. Copy this file to Formula/mlx-whisper-api.rb in that repository
#   3. Update the url, sha256, and resource blocks with actual values
#   4. Users can then install with:
#        brew tap gentleBits/mlx-whisper-api
#        brew install mlx-whisper-api
#
# To generate resource blocks for dependencies, use homebrew-pypi-poet:
#   pip install homebrew-pypi-poet
#   poet mlx-whisper-api > resources.rb
#
class MlxWhisperApi < Formula
  include Language::Python::Virtualenv

  desc "REST API for audio-to-text transcription using MLX Whisper on Apple Silicon"
  homepage "https://github.com/gentleBits/mlx-whisper-api"
  # Option 1: Install from PyPI (preferred, after publishing)
  # url "https://files.pythonhosted.org/packages/source/m/mlx-whisper-api/mlx_whisper_api-0.1.0.tar.gz"
  # Option 2: Install from GitHub release tarball
  url "https://github.com/gentleBits/mlx-whisper-api/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "b4d426598d4d7c1ad054f403cbc36a8c85995ae4e2a4efc5092404cd3bfaa95c"
  license "MIT"
  head "https://github.com/gentleBits/mlx-whisper-api.git", branch: "main"

  # MLX requires Apple Silicon
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  # Core dependencies
  # Generate these with: poet mlx-whisper-api
  # Or manually add each dependency

  resource "mlx" do
    url "https://files.pythonhosted.org/packages/source/m/mlx/mlx-0.21.0.tar.gz"
    sha256 "REPLACE_WITH_ACTUAL_SHA256"
  end

  resource "mlx-whisper" do
    url "https://files.pythonhosted.org/packages/source/m/mlx-whisper/mlx_whisper-0.4.2.tar.gz"
    sha256 "REPLACE_WITH_ACTUAL_SHA256"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.2.tar.gz"
    sha256 "ed6906f61834d687738d25988ae117683705636936cc605be0bb208b23df4d8f"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/source/f/fastapi/fastapi-0.115.7.tar.gz"
    sha256 "0f106da6c01d88a6786b3248fb3404179a2907fc98a3a12d652ccfe46e0f6dc7"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/source/u/uvicorn/uvicorn-0.34.0.tar.gz"
    sha256 "404051050cd7e905de2c9a7e61790943440b3416f49cb409f965d9dcd0fa73e9"
  end

  resource "starlette" do
    url "https://files.pythonhosted.org/packages/source/s/starlette/starlette-0.45.3.tar.gz"
    sha256 "2cbcba2a75806f8a41c722141486f37c28e30a0921c5f6fe4346cb0dcee1302f"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic/pydantic-2.10.5.tar.gz"
    sha256 "278b38dbbaec562011d659ee05f63346951b3a248a6f3642e1bc68894ea2b4ff"
  end

  resource "pydantic-core" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic-core/pydantic_core-2.27.2.tar.gz"
    sha256 "eb026e5a4c1fee05726072337ff51d1efb6f59090b7da90d30ea58625b1ffb39"
  end

  resource "python-multipart" do
    url "https://files.pythonhosted.org/packages/source/p/python-multipart/python_multipart-0.0.20.tar.gz"
    sha256 "8dd0cab45b8e23064ae09147625994d090fa46f5b0d1e13af944c331a7fa9571"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/source/h/huggingface-hub/huggingface_hub-0.27.1.tar.gz"
    sha256 "c004463ca870283909d715d20f066ebd6434799f3b7c9c8a710ade7effd87c20"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/source/c/click/click-8.1.8.tar.gz"
    sha256 "ed53c9d8990d83c2a27deae68e4ee337473f6330c040a31d4225c9574d16096a"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/source/h/h11/h11-0.14.0.tar.gz"
    sha256 "8f19fbbe99e72420ff35c00b27a34cb9937e902a8b810e2c88300c6f0a3b699d"
  end

  resource "httptools" do
    url "https://files.pythonhosted.org/packages/source/h/httptools/httptools-0.6.4.tar.gz"
    sha256 "4e93eee4f0f8ff98299efdb35b8a4703a7a2d326498b5bd37bb36ef81377864f"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/source/t/typing-extensions/typing_extensions-4.12.2.tar.gz"
    sha256 "1a7ead55c7e559dd4dee8856e3a88b41225abfe1ce8df57b7c13915fe121ffb8"
  end

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/source/a/annotated-types/annotated_types-0.7.0.tar.gz"
    sha256 "aff07c09a53a08bc8cfccb9c85b05f1aa9a2a6f23728d790723543408344ce89"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/source/a/anyio/anyio-4.8.0.tar.gz"
    sha256 "1d9fe889df5212298c0c0723fa20479d1b94883a2df44f529e519fecdc43def9"
  end

  resource "sniffio" do
    url "https://files.pythonhosted.org/packages/source/s/sniffio/sniffio-1.3.1.tar.gz"
    sha256 "f4324edc670a0f49750a81b895f35c3adb843cca46f0530f79fc1babb23789dc"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/source/i/idna/idna-3.10.tar.gz"
    sha256 "12f65c9b470abda6dc35cf8e63cc574b1c52b11df2c86030af0ac09b01b13ea9"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/source/c/certifi/certifi-2024.12.14.tar.gz"
    sha256 "b650d30f370c2b724812bee08008be0c4163b163ddaec3f2546c1caf65f191db"
  end

  resource "charset-normalizer" do
    url "https://files.pythonhosted.org/packages/source/c/charset-normalizer/charset_normalizer-3.4.1.tar.gz"
    sha256 "44251f18cd68a75b56585dd00dae26183e102cd5e0f9f1466e6df5da2ed64ea3"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/source/u/urllib3/urllib3-2.3.0.tar.gz"
    sha256 "f8c5449b3cf0861679ce7e0503c7b44b5ec981bec0d1d3795a07f1ba96f0204d"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/source/r/requests/requests-2.32.3.tar.gz"
    sha256 "55365417734eb18255590a9ff9eb97e9e1da868d4ccd6402399eaf68af20a760"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/source/f/filelock/filelock-3.16.1.tar.gz"
    sha256 "c249fbfcd5db47e5e2d6d62198e565475ee65e4831e2561c8e313fa7eb961435"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/source/f/fsspec/fsspec-2024.12.0.tar.gz"
    sha256 "670700c977ed2fb51e0f9f1c40343a2426ad92b28da8837482fbde3da2f82f31"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/p/pyyaml/pyyaml-6.0.2.tar.gz"
    sha256 "d584d9ec91ad65861cc08d42e834324ef890a082e591037abe114850ff7bbc3e"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/source/t/tqdm/tqdm-4.67.1.tar.gz"
    sha256 "f8aef9c52c08c13a65f30ea34f4e5aac3fd1a34959b7f74571d160e2d0f7f7a5"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/source/p/packaging/packaging-24.2.tar.gz"
    sha256 "c228a6dc5e932d346bc5739379109d49e8853dd8223571c7c5b55260edc0b97f"
  end

  resource "uvloop" do
    url "https://files.pythonhosted.org/packages/source/u/uvloop/uvloop-0.21.0.tar.gz"
    sha256 "3bf12b0fda68447806a7ad847bfa591613177275f35b6724b1ee573faa3704e3"
  end

  resource "watchfiles" do
    url "https://files.pythonhosted.org/packages/source/w/watchfiles/watchfiles-1.0.4.tar.gz"
    sha256 "6ba473efd11062d73e4f00c2b730255f9c1bdd73cd5f9fe5b5da8dbd4a717205"
  end

  resource "websockets" do
    url "https://files.pythonhosted.org/packages/source/w/websockets/websockets-14.2.tar.gz"
    sha256 "5059ed9c54945efb321f097084b4c7e52c246f2c869815876a69d1efc4ad6eb5"
  end

  def install
    # Create virtualenv and install dependencies
    virtualenv_install_with_resources

    # Generate shell completions (optional)
    # generate_completions_from_executable(bin/"mlx-whisper-api", shells: [:bash, :zsh, :fish])
  end

  def caveats
    <<~EOS
      MLX Whisper API has been installed!

      To start the server:
        mlx-whisper-api

      To start on a custom port:
        mlx-whisper-api --port 8080

      To enable development mode with auto-reload:
        mlx-whisper-api --reload

      For all options:
        mlx-whisper-api --help

      API documentation will be available at:
        http://localhost:8000/docs

      Note: The first transcription may take longer as the Whisper model
      needs to be downloaded from HuggingFace Hub.
    EOS
  end

  test do
    # Test that the CLI runs and shows version
    assert_match version.to_s, shell_output("#{bin}/mlx-whisper-api --version")

    # Test that the server starts (briefly)
    require "timeout"
    begin
      pid = fork do
        exec bin/"mlx-whisper-api", "--port", "58432"
      end
      sleep 3
      output = shell_output("curl -s http://127.0.0.1:58432/health")
      assert_match "healthy", output
    ensure
      Process.kill("TERM", pid) if pid
      Process.wait(pid) if pid
    end
  end
end
