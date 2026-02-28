#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== claude-ai-memory setup ==="

# 1. Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# 2. Create host data dir
echo "Creating /var/lib/openviking/data..."
sudo mkdir -p /var/lib/openviking/data
sudo chown "$USER:$USER" /var/lib/openviking/data

# 3. Check config
if [ ! -f "$PROJECT_DIR/config/ov.conf" ]; then
    echo "Creating config from example..."
    cp "$PROJECT_DIR/config/ov.conf.example" "$PROJECT_DIR/config/ov.conf"
    echo "IMPORTANT: Edit config/ov.conf and set your MiniMax API key"
fi

# 4. Build and start
echo "Building and starting services..."
cd "$PROJECT_DIR"
docker compose build embedding
docker compose up -d

# 5. Wait for health
echo "Waiting for services..."
for i in {1..30}; do
    if curl -fsS http://localhost:1933/health &>/dev/null; then
        echo "OpenViking: healthy"
        break
    fi
    sleep 2
done

for i in {1..30}; do
    if curl -fsS http://localhost:8100/health &>/dev/null; then
        echo "Embedding: healthy"
        break
    fi
    sleep 2
done

# 6. Pull Ollama model
echo "Pulling Ollama fallback model..."
docker compose exec ollama ollama pull qwen3:0.6b || echo "WARNING: Ollama model pull failed"

# 7. Install Python deps
echo "Installing Python dependencies..."
pip install -e ".[dev]"

echo ""
echo "=== Setup complete ==="
echo "Add to ~/.claude/settings.json:"
echo '  "mcpServers": {'
echo '    "claude-ai-memory": {'
echo "      \"command\": \"python\","
echo "      \"args\": [\"$PROJECT_DIR/mcp_server/server.py\"]"
echo '    }'
echo '  }'
