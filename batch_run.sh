#!/bin/bash
# Run batch image generation
# Usage:
#   ./batch_run.sh create prompts.csv              # Create batch from file
#   ./batch_run.sh create prompts.json --enhance   # Create with Ollama enhancement
#   ./batch_run.sh run                             # Run latest pending batch
#   ./batch_run.sh run batch_20231230_123456       # Run specific batch
#   ./batch_run.sh list                            # List all batches
#   ./batch_run.sh resume batch_20231230_123456    # Resume interrupted batch
#
# Pipeline example:
#   ./generate_prompts.sh "fantasy castles" 50 -o batch_castles.json
#   ./batch_run.sh create batch_castles.json --enhance
#   ./batch_run.sh run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/hunyuan_env/bin/activate"

# Set CUDA device for RTX PRO 6000 Blackwell
export CUDA_VISIBLE_DEVICES=1

# Check if reading from stdin
if [[ "$1" == "--stdin" || "$1" == "-" ]]; then
    # Read prompts from stdin, save to temp file, then process
    TEMP_FILE=$(mktemp --suffix=.json)

    # Read all input and format as JSON
    PROMPTS=()
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            PROMPTS+=("$line")
        fi
    done

    # Build JSON
    echo "{" > "$TEMP_FILE"
    echo '  "batch_name": "stdin_batch",' >> "$TEMP_FILE"
    echo '  "prompts": [' >> "$TEMP_FILE"

    for i in "${!PROMPTS[@]}"; do
        if [[ $i -gt 0 ]]; then
            echo "," >> "$TEMP_FILE"
        fi
        echo "    {\"prompt\": \"${PROMPTS[$i]}\"}" >> "$TEMP_FILE"
    done

    echo "  ]" >> "$TEMP_FILE"
    echo "}" >> "$TEMP_FILE"

    shift  # Remove --stdin from args
    python "$SCRIPT_DIR/batch_generator.py" create "$TEMP_FILE" "$@"
    BATCH_ID=$(python "$SCRIPT_DIR/batch_generator.py" list 2>/dev/null | tail -1 | awk '{print $1}')

    echo ""
    echo "Running batch..."
    python "$SCRIPT_DIR/batch_generator.py" run "$BATCH_ID" "$@"

    rm -f "$TEMP_FILE"
else
    python "$SCRIPT_DIR/batch_generator.py" "$@"
fi
