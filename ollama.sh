#!/bin/bash
# Ollama Server Manager
# Usage:
#   ./ollama.sh start     - Start Ollama server
#   ./ollama.sh stop      - Stop Ollama server
#   ./ollama.sh restart   - Restart Ollama server
#   ./ollama.sh status    - Show server status
#   ./ollama.sh list      - List installed models
#   ./ollama.sh pull <model>    - Install a model
#   ./ollama.sh delete <model>  - Remove a model
#   ./ollama.sh recommended     - Show recommended models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/hunyuan_env/bin/activate"

python "$SCRIPT_DIR/ollama_manager.py" "$@"
