#!/bin/bash
# Enhance a single prompt using Ollama
# Usage: ./enhance_prompt.sh "your prompt here" [--style cinematic] [--model qwen2.5:7b-instruct]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/hunyuan_env/bin/activate"

python "$SCRIPT_DIR/ollama_prompts.py" "$@"
