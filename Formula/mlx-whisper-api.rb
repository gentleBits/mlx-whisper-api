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
  depends_on "mlx"
  depends_on "python@3.12"

  # Core dependencies
  # Generate these with: poet mlx-whisper-api
  # Or manually add each dependency

  # Note: mlx is installed via Homebrew core dependency, not as a Python resource
  # because the PyPI version is outdated and mlx builds require special tooling

  resource "mlx-whisper" do
    url "https://files.pythonhosted.org/packages/4b/be/da654e46741fb07aafbfe44b8e8227f890a6874f17dd068db310d75d491b/mlx_whisper-0.4.2.tar.gz"
    sha256 "5487d967245291fd45d5f11c98a69da1130b9304d0767663412d56fffd71e088"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/ec/d0/c12ddfd3a02274be06ffc71f3efc6d0e457b0409c4481596881e748cb264/numpy-2.2.2.tar.gz"
    sha256 "ed6906f61834d687738d25988ae117683705636936cc605be0bb208b23df4d8f"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/a2/f5/3f921e59f189e513adb9aef826e2841672d50a399fead4e69afdeb808ff4/fastapi-0.115.7.tar.gz"
    sha256 "0f106da6c01d88a6786b3248fb4d7a940d071f6f488488898ad5d354b25ed015"
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
    url "https://files.pythonhosted.org/packages/6a/c7/ca334c2ef6f2e046b1144fe4bb2a5da8a4c574e7f2ebf7e16b34a6a2fa92/pydantic-2.10.5.tar.gz"
    sha256 "278b38dbbaec562011d659ee05f63346951b3a248a6f3642e1bc68894ea2b4ff"
  end

  resource "pydantic-core" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic-core/pydantic_core-2.27.2.tar.gz"
    sha256 "eb026e5a4c1fee05726072337ff51d1efb6f59090b7da90d30ea58625b1ffb39"
  end

  resource "python-multipart" do
    url "https://files.pythonhosted.org/packages/f3/87/f44d7c9f274c7ee665a29b885ec97089ec5dc034c7f3fafa03da9e39a09e/python_multipart-0.0.20.tar.gz"
    sha256 "8dd0cab45b8e23064ae09147625994d090fa46f5b0d1e13af944c331a7fa9d13"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/e1/d2/d6976de7542792fc077b498d64af64882b6d8bb40679284ec0bff77d5929/huggingface_hub-0.27.1.tar.gz"
    sha256 "c004463ca870283909d715d20f066ebd6968c2207dae9393fdffb3c1d4d8f98b"
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
    url "https://files.pythonhosted.org/packages/a7/9a/ce5e1f7e131522e6d3426e8e7a490b3a01f39a6696602e1c4f33f9e94277/httptools-0.6.4.tar.gz"
    sha256 "4e93eee4add6493b59a5c514da98c939b244fce4a0d8879cd3f466562f4b7d5c"
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
    url "https://files.pythonhosted.org/packages/a3/73/199a98fc2dae33535d6b8e8e6ec01f8c1d76c9adb096c6b7d64823038cde/anyio-4.8.0.tar.gz"
    sha256 "1d9fe889df5212298c0c0723fa20479d1b94883a2df44bd3897aa91083316f7a"
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
    url "https://files.pythonhosted.org/packages/aa/63/e53da845320b757bf29ef6a9062f5c669fe997973f966045cb019c3f4b66/urllib3-2.3.0.tar.gz"
    sha256 "f8c5449b3cf0861679ce7e0503c7b44b5ec981bec0d1d3795a07f1ba96f0204d"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/source/r/requests/requests-2.32.3.tar.gz"
    sha256 "55365417734eb18255590a9ff9eb97e9e1da868d4ccd6402399eaf68af20a760"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/9d/db/3ef5bb276dae18d6ec2124224403d1d67bccdbefc17af4cc8f553e341ab1/filelock-3.16.1.tar.gz"
    sha256 "c249fbfcd5db47e5e2d6d62198e565475ee65e4831e2561c8e313fa7eb961435"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/ee/11/de70dee31455c546fbc88301971ec03c328f3d1138cfba14263f651e9551/fsspec-2024.12.0.tar.gz"
    sha256 "670700c977ed2fb51e0d9f9253177ed20cbde4a3e5c0283cc5385b5870c8533f"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/p/pyyaml/pyyaml-6.0.2.tar.gz"
    sha256 "d584d9ec91ad65861cc08d42e834324ef890a082e591037abe114850ff7bbc3e"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/a8/4b/29b4ef32e036bb34e4ab51796dd745cdba7ed47ad142a9f4a1eb8e0c744d/tqdm-4.67.1.tar.gz"
    sha256 "f8aef9c52c08c13a65f30ea34f4e5aac3fd1a34959879d7e59e63027286627f2"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/source/p/packaging/packaging-24.2.tar.gz"
    sha256 "c228a6dc5e932d346bc5739379109d49e8853dd8223571c7c5b55260edc0b97f"
  end

  resource "uvloop" do
    url "https://files.pythonhosted.org/packages/af/c0/854216d09d33c543f12a44b393c402e89a920b1a0a7dc634c42de91b9cf6/uvloop-0.21.0.tar.gz"
    sha256 "3bf12b0fda68447806a7ad847bfa591613177275d35b6724b1ee573faa3704e3"
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
        http://localhost:1738/docs

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
