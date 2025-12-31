#!/bin/bash
#
# HunyuanImage-3.0 Launcher
# Launches the Web UI with Ollama support
#

SCRIPT_DIR="/media/james/DataDrive/jamesw767/Hun3d"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     ${GREEN}HunyuanImage-3.0 AI Image Generator${NC}           ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}     with Ollama LLM Prompt Enhancement            ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment
source "$SCRIPT_DIR/hunyuan_env/bin/activate"

# Check Ollama status
echo -e "${YELLOW}Checking Ollama status...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓ Ollama running with $MODELS model(s)${NC}"
else
    echo -e "${YELLOW}Starting Ollama server...${NC}"
    python3 "$SCRIPT_DIR/ollama_manager.py" start --no-wait
    sleep 2
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama started${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama not available (prompt enhancement will be disabled)${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}Loading HunyuanImage-3.0 model...${NC}"
echo -e "${YELLOW}This may take 1-2 minutes on first load.${NC}"
echo ""

# Check if UI is already running
if lsof -i :7860 > /dev/null 2>&1; then
    echo -e "${GREEN}UI already running!${NC}"
    echo -e "Opening browser to ${BLUE}http://localhost:7860${NC}"
    sleep 1
    xdg-open "http://localhost:7860" 2>/dev/null &
    exit 0
fi

# Open browser after a delay (background)
(sleep 30 && xdg-open "http://localhost:7860" 2>/dev/null) &

# Set CUDA device for Blackwell GPU
export CUDA_VISIBLE_DEVICES=1

# Run the UI
python3 "$SCRIPT_DIR/hunyuan_ui.py"
