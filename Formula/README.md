# Homebrew Formula for MLX Whisper API

This directory contains the Homebrew formula for distributing MLX Whisper API via Homebrew.

## Setting Up Your Homebrew Tap

### Step 1: Create the Tap Repository

1. Create a new GitHub repository named `homebrew-mlx-whisper-api`
   - The `homebrew-` prefix is **required** for Homebrew taps
   - Example: `https://github.com/YOURUSERNAME/homebrew-mlx-whisper-api`

2. Clone the repository:
   ```bash
   git clone https://github.com/YOURUSERNAME/homebrew-mlx-whisper-api.git
   cd homebrew-mlx-whisper-api
   ```

3. Create the Formula directory and copy the formula:
   ```bash
   mkdir -p Formula
   cp /path/to/mlx-whisper-api/Formula/mlx-whisper-api.rb Formula/
   ```

### Step 2: Create a Release

Before the formula will work, you need a release:

**Option A: GitHub Release (Recommended)**
```bash
# In your main mlx-whisper-api repository
git tag v0.1.0
git push origin v0.1.0
# Then create a release on GitHub from this tag
```

**Option B: Publish to PyPI**
```bash
# Build and publish
poetry build
poetry publish
```

### Step 3: Update the Formula with Real Values

1. **Get the SHA256 hash** of your release tarball:
   ```bash
   # For GitHub release:
   curl -sL https://github.com/YOURUSERNAME/mlx-whisper-api/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256

   # For PyPI:
   curl -sL https://files.pythonhosted.org/packages/source/m/mlx-whisper-api/mlx_whisper_api-0.1.0.tar.gz | shasum -a 256
   ```

2. **Generate dependency resource blocks** using `poet`:
   ```bash
   pip install homebrew-pypi-poet
   poet mlx-whisper-api
   ```
   This outputs the exact `resource` blocks with correct URLs and SHA256 hashes.

3. **Update the formula** with:
   - Your GitHub username in URLs
   - Correct SHA256 hash for the main package
   - Updated resource blocks from `poet` output

### Step 4: Test the Formula Locally

```bash
# Test installation
brew install --build-from-source ./Formula/mlx-whisper-api.rb

# Run the formula tests
brew test mlx-whisper-api

# Audit the formula for issues
brew audit --strict mlx-whisper-api
```

### Step 5: Push and Use

```bash
# In your homebrew-mlx-whisper-api tap repository
git add Formula/mlx-whisper-api.rb
git commit -m "Add mlx-whisper-api formula v0.1.0"
git push origin main
```

Users can now install:
```bash
brew tap YOURUSERNAME/mlx-whisper-api
brew install mlx-whisper-api
```

## Updating the Formula

When you release a new version:

1. Create a new release/tag in the main repository
2. Get the new SHA256 hash
3. Update the formula:
   ```ruby
   url "https://github.com/YOURUSERNAME/mlx-whisper-api/archive/refs/tags/v0.2.0.tar.gz"
   sha256 "NEW_SHA256_HASH"
   ```
4. Update any changed dependency versions
5. Push to the tap repository

## Troubleshooting

### "mlx" or "mlx-whisper" fails to build

MLX requires compilation. Ensure users have:
- Xcode Command Line Tools: `xcode-select --install`
- CMake (if needed): `brew install cmake`

You may need to add to the formula:
```ruby
depends_on "cmake" => :build
```

### Missing dependencies

If `poet` misses some dependencies, check the actual installed packages:
```bash
pip install mlx-whisper-api
pip freeze | grep -v mlx-whisper-api
```

### Architecture errors

The formula enforces Apple Silicon with:
```ruby
depends_on arch: :arm64
```

This will show a clear error on Intel Macs.

## Alternative: Direct pip/pipx Installation

If users don't want to use the tap, they can install directly:

```bash
# Using pipx (recommended for CLI tools)
brew install pipx
pipx install mlx-whisper-api

# Or using pip
pip install mlx-whisper-api
```
