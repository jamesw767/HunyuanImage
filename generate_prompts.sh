#!/bin/bash
# Generate creative prompts for a theme using Ollama
# Usage: ./generate_prompts.sh "theme" [count] [--style cinematic] [--output prompts.json]
#
# Examples:
#   ./generate_prompts.sh "cyberpunk cities" 20
#   ./generate_prompts.sh "underwater scenes" 10 --style cinematic --output batch_underwater.json
#   ./generate_prompts.sh --character "A young woman with silver hair and blue eyes" -n 5
#   ./generate_prompts.sh "fantasy" --fusion "art nouveau" "steampunk" -n 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/hunyuan_env/bin/activate"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is not running. Starting it..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Handle the case where count is provided as second positional arg
if [[ $# -ge 2 && $2 =~ ^[0-9]+$ ]]; then
    THEME="$1"
    COUNT="$2"
    shift 2
    python "$SCRIPT_DIR/prompt_generator.py" "$THEME" --count "$COUNT" "$@"
else
    python "$SCRIPT_DIR/prompt_generator.py" "$@"
fi
