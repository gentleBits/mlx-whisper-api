#!/bin/bash
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the server in the background
python3 -m app &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
until curl -s http://localhost:1738/health > /dev/null 2>&1; do
    sleep 1
done
echo "Server ready!"

# Transcribe using Q8 model
curl -X POST http://localhost:1738/transcribe \
    -F "file=@tests/fixtures/audio/harvard_sample.wav" \
    -F "model=mlx-community/whisper-large-v3-mlx-8bit"

echo ""

# Cleanup
kill $SERVER_PID 2>/dev/null
